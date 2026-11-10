"""
PySR (symbolic regression) as a second model family.

PySR is an optional dependency (it needs a Julia runtime, installed on
first import). This module imports pysr only inside fit(), so importing
the engine never starts Julia; candidates.py registers the PySR candidate
only when the package is installed.

Fairness choices, so differences come from the model and not the setup:
  - PySR gets exactly the same derivative estimates as SINDy
    (preprocess.estimate_derivatives) and regresses dX/dt on X.
  - Default operators are + - * only: a polynomial search space directly
    comparable to SINDy-poly.
  - n_terms is the number of additive terms after sympy expansion, the same
    unit as SINDy's nonzero-coefficient count (PySR's own tree "complexity"
    is kept in extra). Polynomial results are also converted to
    {term: coef} in PySINDy term names, so structure recovery applies.

Runs are multithreaded and time-budgeted, so they are NOT bit-reproducible
even with a fixed seed. The seed, PySR version and `reproducible: False`
are stored in the config, and therefore in every logged run.
"""
import contextlib
import io
import tempfile
import warnings

import numpy as np
import sympy

from engine.candidates import Candidate, FittedModel
from pipeline.preprocess import estimate_derivatives

COEF_TOL = 1e-10  # expansion can leave float dust like 1e-17*x; drop it


# ---------------------------------------------- sympy -> engine formats ---

def expand(expr):
    return sympy.expand(sympy.sympify(expr))


def count_terms(expr):
    """Additive terms after expansion: '2*x*y - y' -> 2."""
    e = expand(expr)
    return 0 if e == 0 else len(sympy.Add.make_args(e))


def to_coefficients(expr, var_names):
    """
    Expanded polynomial -> {term: coef} in PySINDy names ('1', 'x', 'x^2 y').
    Returns None if the expression is not a polynomial in the variables
    (e.g. if a future PySR config allows sin/cos or division).
    """
    syms = sympy.symbols(var_names)
    e = expand(expr)
    if not e.is_polynomial(*syms):
        return None
    out = {}
    for powers, c in sympy.Poly(e, *syms).terms():
        c = float(c)
        if abs(c) < COEF_TOL:
            continue
        factors = [v if p == 1 else f"{v}^{p}" for v, p in zip(var_names, powers) if p]
        out[" ".join(factors) or "1"] = c
    return out


def format_equation(expr, digits=4):
    return str(sympy.N(expand(expr), digits))


# ------------------------------------------------------------ candidate ---

class PySRCandidate(Candidate):
    """
    Config:
      binary_operators / unary_operators   PySR search space
      timeout_per_equation   seconds of search per state variable
      maxsize                max expression tree size
      model_selection        how PySR picks one equation from its Pareto front
      deriv_method           same options as SINDy
      random_state           seed (logged; runs are still not bit-reproducible)
    """
    family = "PySR"

    DEFAULTS = dict(binary_operators=["+", "-", "*"], unary_operators=[],
                    timeout_per_equation=60, maxsize=30, model_selection="best",
                    deriv_method="finite_difference", random_state=0,
                    parallelism="multithreading", reproducible=False)

    def __init__(self, name, **config):
        super().__init__(name, **{**self.DEFAULTS, **config})
        try:
            from importlib.metadata import version
            self.config["pysr_version"] = version("pysr")
        except Exception:  # noqa: BLE001
            self.config["pysr_version"] = None

    def fit(self, t, X, var_names):
        from pysr import PySRRegressor  # lazy: starts Julia on first use

        c = self.config
        with contextlib.redirect_stdout(io.StringIO()):
            X_dot = estimate_derivatives(X, t, method=c["deriv_method"])

        model = PySRRegressor(
            binary_operators=list(c["binary_operators"]),
            unary_operators=list(c["unary_operators"]),
            maxsize=c["maxsize"],
            model_selection=c["model_selection"],
            niterations=1_000_000,                      # the timeout is the real budget
            timeout_in_seconds=c["timeout_per_equation"] * X.shape[1],
            random_state=c["random_state"],
            parallelism=c["parallelism"],
            deterministic=False,
            verbosity=0,
            progress=False,
            output_directory=tempfile.mkdtemp(prefix="pysr_"),
        )
        with warnings.catch_warnings():
            # Known and recorded (config["reproducible"] = False); don't repeat it.
            warnings.filterwarnings("ignore", message=".*random_state.*deterministic")
            model.fit(X, X_dot, variable_names=list(var_names))

        exprs = model.sympy()
        exprs = exprs if isinstance(exprs, list) else [exprs]
        syms = sympy.symbols(var_names)
        funcs = [sympy.lambdify(syms, expand(e), "numpy") for e in exprs]

        def rhs(x):
            x = np.asarray(x, float)
            return np.array([float(f(*x)) for f in funcs])

        # Safety check, as for SINDy: the compiled sympy functions must
        # reproduce PySR's own predictions.
        probe = X[np.linspace(0, len(X) - 1, 20, dtype=int)]
        pred = np.asarray(model.predict(probe)).reshape(len(probe), -1)
        if not np.allclose([rhs(z) for z in probe], pred, rtol=1e-6, atol=1e-8):
            raise RuntimeError("lambdified PySR equations disagree with model.predict")

        coefs = [to_coefficients(e, var_names) for e in exprs]
        coefficients = (dict(zip(var_names, coefs))
                        if all(cf is not None for cf in coefs) else None)

        eqs = model.equations_ if isinstance(model.equations_, list) else [model.equations_]
        best = model.get_best()
        best = best if isinstance(best, list) else [best]
        return FittedModel(
            rhs=rhs,
            n_terms=sum(count_terms(e) for e in exprs),
            symbolic=True,
            equations=[format_equation(e) for e in exprs],
            coefficients=coefficients,
            extra={"pysr_complexity": [int(b["complexity"]) for b in best],
                   "pareto_sizes": [len(e) for e in eqs]},
        )

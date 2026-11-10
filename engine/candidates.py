"""
Model candidates: the plug-in point of the comparison engine.

The harness never knows which model it is scoring. Every model family is
wrapped in a Candidate whose fit() returns a FittedModel answering one
question: "given a state x, what is dx/dt?" (FittedModel.rhs). The harness
then simulates, forecasts and scores every model the same way.

To add a model family (e.g. PySR later):
  1. subclass Candidate and implement fit(t, X, var_names) -> FittedModel
  2. add an instance to CANDIDATES
Nothing in the harness, metrics or store needs to change.
"""
import contextlib
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pysindy as ps

from engine.terms import compile_rhs
from pipeline.models import run_sindy
from pipeline.preprocess import estimate_derivatives, scale_data


@dataclass
class FittedModel:
    rhs: Callable[[np.ndarray], np.ndarray]  # x (n_vars,) -> dx/dt (n_vars,)
    n_terms: int                     # complexity: nonzero terms / expression size
    symbolic: bool                   # closed-form equations (True) or black box
    equations: Optional[list[str]] = None
    # {var: {term: coef}} if the model is a linear combination of named terms;
    # used for structure recovery against ground truth. None for other families.
    coefficients: Optional[dict] = None
    extra: dict = field(default_factory=dict)  # anything family-specific


class Candidate(ABC):
    family: str = "base"

    def __init__(self, name, **config):
        self.name = name
        self.config = config

    @abstractmethod
    def fit(self, t, X, var_names) -> FittedModel:
        """Fit on (t, X) and return a FittedModel. Must not see test data."""

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, {self.config})"


def _build_library(kind, degree, n_frequencies):
    if kind == "poly":
        return ps.PolynomialLibrary(degree=degree)
    if kind == "trig":
        return ps.FourierLibrary(n_frequencies=n_frequencies)
    if kind == "poly+trig":
        return ps.PolynomialLibrary(degree=degree) + ps.FourierLibrary(n_frequencies=n_frequencies)
    raise ValueError(f"Unknown library {kind!r}; use 'poly', 'trig' or 'poly+trig'.")


def _fast_rhs(model, coef, names, var_names, X_fit):
    """
    model.predict() validates its input on every call (~3 ms), which makes
    simulating chaotic systems take minutes. Compile the active terms into a
    plain numpy function instead, verify it reproduces predict() on fitted
    states, and fall back to predict() if the term format is unsupported or
    the check fails.
    """
    slow = lambda z: model.predict(z.reshape(1, -1))[0]  # noqa: E731
    try:
        fast = compile_rhs(coef, names, var_names)
        probe = X_fit[np.linspace(0, len(X_fit) - 1, 20, dtype=int)]
        ok = np.allclose([fast(z) for z in probe], model.predict(probe),
                         rtol=1e-8, atol=1e-10)
        return fast if ok else slow
    except ValueError:
        return slow


class SindyCandidate(Candidate):
    """
    SINDy through the existing pipeline (preprocess.estimate_derivatives +
    models.run_sindy). Config:
      library        'poly' | 'trig' | 'poly+trig'
      degree         polynomial degree (poly libraries)
      n_frequencies  number of sin/cos harmonics (trig libraries)
      threshold      STLSQ threshold
      alpha          ridge strength
      deriv_method   'finite_difference' | 'smoothed_fd'
      scale          z-score first. OFF by default: independent per-variable
                     scaling rescales the library terms inconsistently and
                     distorts STLSQ's magnitude-based pruning. Opt-in only.
    """
    family = "SINDy"

    DEFAULTS = dict(library="poly", degree=3, n_frequencies=1, threshold=0.1,
                    alpha=0.05, deriv_method="finite_difference", scale=False)

    def __init__(self, name, **config):
        super().__init__(name, **{**self.DEFAULTS, **config})

    def fit(self, t, X, var_names):
        c = self.config
        mu, sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        if c["scale"]:
            X, scaler = scale_data(X)
            mu, sd = scaler.mean_, scaler.scale_

        library = _build_library(c["library"], c["degree"], c["n_frequencies"])

        # run_sindy prints progress; silence it inside benchmark loops
        with contextlib.redirect_stdout(io.StringIO()):
            X_dot = estimate_derivatives(X, t, method=c["deriv_method"])
            result = run_sindy(X, t, X_dot=X_dot, feature_names=list(var_names),
                               threshold=c["threshold"], alpha=c["alpha"],
                               library=library)

        model = result["model"]
        coef = np.asarray(result["coefficients"])
        names = result["feature_library_names"]

        f = _fast_rhs(model, coef, names, var_names, X)

        def rhs(x):
            # The model lives in (possibly scaled) z-space: dz/dt = f(z) with
            # z = (x - mu) / sd, so dx/dt = sd * f((x - mu) / sd). With scaling
            # off this is just f(x). The harness always works in data units.
            return sd * f((np.asarray(x, float) - mu) / sd)

        coefficients = None
        if not c["scale"]:  # scaled coefficients aren't comparable to ground truth
            coefficients = {
                v: {names[j]: float(coef[i, j]) for j in np.flatnonzero(coef[i])}
                for i, v in enumerate(var_names)
            }

        return FittedModel(
            rhs=rhs,
            n_terms=int(np.count_nonzero(coef)),
            symbolic=True,
            equations=list(result["equations"]),
            coefficients=coefficients,
            extra={"library_size": len(names)},
        )


# Registry: the candidates the comparison runs by default. A fixed config is
# used across all systems so no candidate is tuned to a particular dataset.
#
# alpha=0: with a degree-3 library the columns differ in norm by >100x
# (cond(Theta) ~ 1e6 on Lotka-Volterra). A ridge penalty on unnormalized
# columns then shrinks the small-norm true terms (x, y) under the threshold,
# and recovery fails even on clean data. threshold=0.15 removes the small
# finite-difference artefacts seen on Lorenz. With these, all four systems
# recover exactly on clean and 1%-noise data (see tests/test_candidates.py).
CANDIDATES = {
    "SINDy-poly": SindyCandidate("SINDy-poly", library="poly", degree=3,
                                 threshold=0.15, alpha=0.0),
}

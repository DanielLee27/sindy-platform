"""
Evaluate PySINDy-style library term names as numpy functions.

Shared by metrics.py (ground-truth derivatives) and candidates.py (fast
compiled rhs for fitted SINDy models). Supported term formats:
  '1'                 constant
  'x', 'x^2 y', ...   polynomial monomials
  'sin(1 x)', 'cos(2 y)'   Fourier terms
Works on a single state (shape [n_vars]) or many (shape [n, n_vars]).
"""
import re

import numpy as np

_TRIG = re.compile(r"^(sin|cos)\(([0-9.]+) (\w+)\)$")
_MONOMIAL = re.compile(r"^\w+(\^\d+)?( \w+(\^\d+)?)*$")


def term_function(term, var_names):
    """Return f(X) -> values of `term`, or raise ValueError if unsupported."""
    idx = {v: i for i, v in enumerate(var_names)}

    if term == "1":
        return lambda X: np.ones(np.shape(X)[:-1]) if np.ndim(X) > 1 else 1.0

    m = _TRIG.match(term)
    if m:
        fn, k, var = np.sin if m[1] == "sin" else np.cos, float(m[2]), idx[m[3]]
        return lambda X: fn(k * X[..., var])

    if _MONOMIAL.match(term):
        factors = []
        for f in term.split():
            var, _, p = f.partition("^")
            if var not in idx:
                raise ValueError(f"Unknown variable in term {term!r}")
            factors.append((idx[var], int(p) if p else 1))

        def mono(X):
            out = 1.0
            for i, p in factors:
                out = out * X[..., i] ** p
            return out
        return mono

    raise ValueError(f"Unsupported term format: {term!r}")


def compile_rhs(coef, term_names, var_names):
    """
    Build rhs(x) = coef @ theta(x), evaluating only the active terms.
    coef: [n_vars, n_terms] coefficient matrix.
    """
    coef = np.asarray(coef, float)
    active = np.flatnonzero(np.any(coef != 0, axis=0))
    funcs = [term_function(term_names[j], var_names) for j in active]
    C = coef[:, active]

    def rhs(x):
        x = np.asarray(x, float)
        return C @ np.array([f(x) for f in funcs])
    return rhs

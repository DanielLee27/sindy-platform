"""
Scoring functions for the comparison engine.

Pure functions: arrays in, numbers out. Nothing here knows which model
produced the arrays, which is what keeps scores comparable across families.

Axes
  reconstruction  vector-field error on the training states:
                  ||f_model(x) - f_true(x)|| / ||f_true(x)||. Instantaneous, so
                  chaos cannot distort it (a trajectory comparison over a long
                  window would measure chaotic divergence, not model error).
  forecast        simulate from the true state at the train/test split:
                  normalized RMSE over the test window + valid prediction time
                  (time until normalized error first exceeds 0.5), the standard
                  fair measure for chaotic systems.
  sparsity        number of nonzero terms (reported by the model).
  stability       does the model stay bounded when simulated over a long
                  horizon: 'stable' / 'diverged' / 'failed' + fraction completed.
  runtime         fit wall-clock seconds (measured by the harness).
  interpretability  fixed-rule category from symbolic + terms per equation.
  structure       (ground truth only) precision / recall / F1 of the active
                  terms and relative coefficient error.
"""
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from engine.terms import term_function

VALID_TIME_THRESHOLD = 0.5
DIVERGENCE_FACTOR = 10.0
PARSIMONIOUS_TERMS_PER_EQ = 4


# ------------------------------------------------------- ground-truth f(x) ---

def evaluate_terms(coefficients, X, var_names):
    """dX/dt from {var: {term: coef}} evaluated at the rows of X."""
    idx = {v: i for i, v in enumerate(var_names)}
    X = np.asarray(X, float)
    dX = np.zeros_like(X)
    for v, terms in coefficients.items():
        for term, c in terms.items():
            dX[:, idx[v]] += c * term_function(term, var_names)(X)
    return dX


# ----------------------------------------------------------- simulation ---

def simulate(rhs, x0, t, bound, timeout=30.0):
    """
    Integrate a model's rhs from x0 over t, the same way for every model.

    Stops early if any |x| exceeds `bound` (divergence) or wall-clock time
    exceeds `timeout` (badly behaved models can make the solver crawl).

    Returns (X_sim, status, fraction_of_horizon_completed), where status is
    'ok', 'diverged' or 'failed: <reason>'.
    """
    start = time.perf_counter()

    def f(_t, x):
        if time.perf_counter() - start > timeout:
            raise TimeoutError(f"simulation exceeded {timeout:.0f}s")
        return rhs(x)

    def blow_up(_t, x):
        return bound - np.max(np.abs(x))
    blow_up.terminal = True

    t = np.asarray(t, float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(f, (t[0], t[-1]), np.asarray(x0, float), t_eval=t,
                            method="LSODA", rtol=1e-8, atol=1e-8, events=blow_up)
    except Exception as exc:  # noqa: BLE001 - any model failure is a result
        return None, f"failed: {exc}", 0.0

    X_sim = sol.y.T
    frac = (sol.t[-1] - t[0]) / (t[-1] - t[0]) if len(sol.t) else 0.0
    if not np.all(np.isfinite(X_sim)) or len(X_sim) < len(t):
        return X_sim, "diverged", float(frac)
    return X_sim, "ok", 1.0


# --------------------------------------------------------------- scores ---

def vector_field_error(f_pred, f_true):
    """Relative error between predicted and true derivatives (0 = perfect)."""
    return float(np.linalg.norm(f_pred - f_true) / max(np.linalg.norm(f_true), 1e-12))


def nrmse(X_pred, X_true, scale):
    """RMSE per variable divided by that variable's spread, averaged."""
    rmse = np.sqrt(np.mean((X_pred - X_true) ** 2, axis=0))
    return float(np.mean(rmse / scale))


def valid_time(X_pred, X_true, t, scale, threshold=VALID_TIME_THRESHOLD):
    """Time until the normalized error first exceeds `threshold`."""
    err = np.sqrt(np.mean(((X_pred - X_true) / scale) ** 2, axis=1))
    bad = np.flatnonzero(err > threshold)
    end = bad[0] if len(bad) else len(t) - 1
    return float(t[end] - t[0])


def interpretability(symbolic, n_terms, n_vars, max_per_eq=PARSIMONIOUS_TERMS_PER_EQ):
    """Fixed rule applied identically to every model family."""
    if not symbolic:
        return "black-box"
    return "parsimonious" if n_terms / n_vars <= max_per_eq else "complex"


def structure_recovery(recovered, truth):
    """
    Compare active terms with the ground truth.
    recovered / truth: {var: {term: coef}}. Returns precision, recall, F1
    over (var, term) pairs and relative coefficient error over their union.
    """
    if recovered is None:
        return dict(term_precision=np.nan, term_recall=np.nan, term_f1=np.nan,
                    coef_error=np.nan)
    rec = {(v, k) for v, terms in recovered.items() for k in terms}
    tru = {(v, k) for v, terms in truth.items() for k in terms}
    tp = len(rec & tru)
    precision = tp / len(rec) if rec else 0.0
    recall = tp / len(tru) if tru else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0

    pairs = rec | tru
    c_hat = np.array([recovered.get(v, {}).get(k, 0.0) for v, k in pairs])
    c_true = np.array([truth.get(v, {}).get(k, 0.0) for v, k in pairs])
    coef_err = np.linalg.norm(c_hat - c_true) / max(np.linalg.norm(c_true), 1e-12)
    return dict(term_precision=precision, term_recall=recall, term_f1=f1,
                coef_error=float(coef_err))

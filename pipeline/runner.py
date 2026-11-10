"""
Run the full pipeline: (scale) -> derivatives -> fit -> simulate.

No Streamlit imports here. This module is plain Python so it can be tested,
scripted, or reused by a different front end.
"""
import warnings

import numpy as np

from pipeline.preprocess import estimate_derivatives, scale_data
from pipeline.registry import MODELS


def run_pipeline(t, X, var_names, model_name="SINDy (STLSQ)", model_params=None,
                 scale=False, deriv_method="finite_difference", simulate=True):
    """
    Parameters
      t, X          — time array [n] and state matrix [n, n_vars]
      var_names     — names for each column of X
      model_name    — key in registry.MODELS
      model_params  — dict of settings for that model (defaults used if missing)
      scale         — z-score each variable first. Off by default: for SINDy,
                      independent per-variable scaling rescales the nonlinear
                      library terms inconsistently, which distorts STLSQ's
                      magnitude-based pruning.
      deriv_method  — passed to preprocess.estimate_derivatives
      simulate      — also integrate the recovered model forward

    Returns a dict with everything the UI needs. "X" is the data in the
    space the model was fit in (scaled if scale=True).
    """
    spec = MODELS[model_name]
    params = {name: p.default for name, p in spec.params.items()}
    params.update(model_params or {})

    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    notes = []

    scaler = None
    if scale:
        X, scaler = scale_data(X)
        notes.append("Data was z-scored: equations are in scaled variables.")

    X_dot = estimate_derivatives(X, t, method=deriv_method)

    fit = spec.fn(X, t, X_dot=X_dot, feature_names=list(var_names), **params)

    # Re-score against the same X_dot the model was fit on. Without x_dot,
    # PySINDy re-differentiates the raw data, so on noisy data the R² mostly
    # measures differentiation noise rather than how good the equations are.
    if "model" in fit and hasattr(fit["model"], "score"):
        fit["score"] = fit["model"].score(X, t=t, x_dot=X_dot)

    X_sim = None
    if simulate and spec.simulate is not None:
        X_sim, sim_note = _safe_simulate(spec.simulate, fit, X[0], t)
        if sim_note:
            notes.append(sim_note)

    return {
        **fit,
        "t": t,
        "X": X,
        "X_dot": X_dot,
        "X_sim": X_sim,
        "var_names": list(var_names),
        "model_name": model_name,
        "params": params,
        "scaled": scale,
        "scaler": scaler,
        "notes": notes,
    }


def _safe_simulate(simulate_fn, fit, x0, t):
    """
    A poorly identified model can blow up when integrated. Return
    (X_sim or None, note) instead of letting that crash the caller.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_sim = simulate_fn(fit, x0, t)
    except Exception as exc:
        return None, f"Simulation of the recovered model failed: {exc}"

    X_sim = np.asarray(X_sim)
    if not np.all(np.isfinite(X_sim)):
        return None, "Simulation of the recovered model diverged (non-finite values)."
    if len(X_sim) < len(t):
        return X_sim, f"Simulation stopped early ({len(X_sim)}/{len(t)} steps)."
    return X_sim, None


if __name__ == "__main__":
    from pipeline.systems import DEMO_SYSTEMS

    for name, s in DEMO_SYSTEMS.items():
        t, X = s.generate(s.t_end, s.dt, 0.0, 0, **s.params)
        r = run_pipeline(t, X, s.var_names, model_params=s.suggested)
        print(f"\n=== {name}  (R² = {r['score']:.4f})")
        for v, eq, true in zip(r["var_names"], r["equations"], s.true_equations(**s.params)):
            print(f"  recovered d{v}/dt = {eq}")
            print(f"  true      {true}")
        print("  notes:", r["notes"] or "none")

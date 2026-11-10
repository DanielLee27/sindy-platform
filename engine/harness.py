"""
Evaluation harness: scores any Candidate on any Dataset on fixed axes.

evaluate(candidate, dataset) -> one flat record (dict)
compare(candidates, datasets, store=None) -> DataFrame, one row per pair

The harness only touches the Candidate / FittedModel interface: it fits,
then simulates and scores using FittedModel.rhs with ONE integrator and ONE
set of rules for every model family. A model that crashes is a result
(status='failed'), not a reason to stop the run.
"""
import time
import traceback

import numpy as np
import pandas as pd

from engine import metrics

RECON_POINTS = 500  # training states at which the vector field is compared


def _base_record(candidate, dataset):
    return {
        "dataset_id": dataset.dataset_id,
        "system": dataset.system,
        "model_name": candidate.name,
        "family": candidate.family,
        "model_config": dict(candidate.config),
        "spec": dict(dataset.spec),
        "features": dict(dataset.features),
        "status": "ok",
        "error": None,
        "fit_seconds": np.nan,
        "n_terms": np.nan,
        "interpretability": None,
        "reconstruction_error": np.nan,
        "forecast_nrmse": np.nan,
        "valid_time": np.nan,
        "valid_time_frac": np.nan,
        "stability": "failed",
        "stability_frac": 0.0,
        "term_precision": np.nan,
        "term_recall": np.nan,
        "term_f1": np.nan,
        "coef_error": np.nan,
        "equations": None,
    }


def evaluate(candidate, dataset, sim_timeout=30.0):
    """Fit `candidate` on the training split of `dataset` and score it."""
    d = dataset
    rec = _base_record(candidate, d)
    Xc_train = d.X_clean[: d.split]
    scale = Xc_train.std(axis=0)
    bound = metrics.DIVERGENCE_FACTOR * np.abs(Xc_train).max()

    # 1. Fit (runtime axis). The model only ever sees noisy training data.
    t0 = time.perf_counter()
    try:
        fitted = candidate.fit(d.t_train, d.X_train, d.var_names)
    except Exception as exc:  # noqa: BLE001
        rec.update(status="failed", error=_short_error(exc),
                   fit_seconds=time.perf_counter() - t0)
        return rec
    rec["fit_seconds"] = time.perf_counter() - t0
    rec["n_terms"] = fitted.n_terms
    rec["equations"] = fitted.equations
    rec["interpretability"] = metrics.interpretability(
        fitted.symbolic, fitted.n_terms, len(d.var_names))
    rec.update(metrics.structure_recovery(fitted.coefficients, d.true_coefficients))

    try:
        # 2. Reconstruction: vector-field error at clean training states
        idx = np.linspace(0, d.split - 1, min(RECON_POINTS, d.split), dtype=int)
        f_true = metrics.evaluate_terms(d.true_coefficients, Xc_train[idx], d.var_names)
        f_pred = np.array([fitted.rhs(x) for x in Xc_train[idx]])
        rec["reconstruction_error"] = metrics.vector_field_error(f_pred, f_true)

        # 3. Forecast: from the true state at the split, over the test window
        t_test = d.t[d.split:]
        X_test = d.X_clean[d.split:]
        X_sim, status, _ = metrics.simulate(fitted.rhs, X_test[0], t_test, bound, sim_timeout)
        if X_sim is not None and len(X_sim):
            n = len(X_sim)
            rec["valid_time"] = metrics.valid_time(X_sim, X_test[:n], t_test[:n], scale)
            rec["valid_time_frac"] = rec["valid_time"] / (t_test[-1] - t_test[0])
            if status == "ok":
                rec["forecast_nrmse"] = metrics.nrmse(X_sim, X_test, scale)

        # 4. Stability: stay bounded over a long horizon (full data duration)
        horizon = d.t[-1] - d.t[0]
        t_long = np.linspace(0, horizon, 1000)
        _, long_status, frac = metrics.simulate(fitted.rhs, X_test[0], t_long, bound,
                                                sim_timeout)
        worst = status if status != "ok" else long_status
        rec["stability"] = {"ok": "stable", "diverged": "diverged"}.get(worst, "failed")
        rec["stability_frac"] = 1.0 if worst == "ok" else min(frac, 1.0)
        if worst.startswith("failed"):
            rec["error"] = worst
    except Exception as exc:  # noqa: BLE001
        rec.update(status="failed", error=_short_error(exc))

    return rec


def compare(candidates, datasets, store=None, verbose=True):
    """
    Evaluate every candidate on every dataset. If `store` is given, each
    record is logged as soon as it is produced (so a crash mid-run keeps
    everything finished so far). Returns a DataFrame, one row per pair.
    """
    candidates = list(candidates.values()) if isinstance(candidates, dict) else list(candidates)
    records = []
    total = len(candidates) * len(datasets)
    for i, d in enumerate(datasets):
        for j, c in enumerate(candidates):
            rec = evaluate(c, d)
            records.append(rec)
            if store is not None:
                store.log(rec)
            if verbose:
                k = i * len(candidates) + j + 1
                print(f"[{k}/{total}] {d.system:15s} noise={d.spec['noise']:<5} "
                      f"{c.name:12s} -> {rec['stability']:8s} "
                      f"recon={rec['reconstruction_error']:.3g} "
                      f"valid_t={rec['valid_time']:.3g}")
    return pd.DataFrame(records)


# Score columns shown in the comparison table, and which direction is better
TABLE_METRICS = {
    "reconstruction_error": "lower",
    "forecast_nrmse": "lower",
    "valid_time": "higher",
    "n_terms": "lower",
    "stability": None,
    "fit_seconds": "lower",
    "interpretability": None,
    "term_f1": "higher",
}


def comparison_table(df, metrics_=None):
    """
    Side-by-side table: one row per dataset (system, noise, dt, n_samples),
    one column group per metric, one sub-column per model.
    """
    metrics_ = list(metrics_ or TABLE_METRICS)
    df = df.copy()
    for key in ("noise", "dt", "n_samples"):
        df[key] = df["spec"].map(lambda s, k=key: s[k])
    index = ["system", "noise", "dt", "n_samples"]
    return df.pivot_table(index=index, columns="model_name", values=metrics_,
                          aggfunc="first")[metrics_]


def _short_error(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    where = f" ({tb[-1].filename.split('/')[-1].split(chr(92))[-1]}:{tb[-1].lineno})" if tb else ""
    return f"{type(exc).__name__}: {exc}{where}"[:500]

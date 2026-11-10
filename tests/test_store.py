import sqlite3

import numpy as np

from engine.benchmarks import make_dataset
from engine.candidates import CANDIDATES
from engine.harness import compare, evaluate
from engine.store import RunStore
from tests.test_harness import Crashing

SINDY = {k: CANDIDATES[k] for k in ("SINDy-poly", "SINDy-trig")}


def test_every_run_is_logged_and_round_trips(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    datasets = [make_dataset("Van der Pol", n_samples=1500),
                make_dataset("Lotka-Volterra", noise=0.05, n_samples=1500)]
    df = compare(SINDY, datasets, store=store, verbose=False)
    assert len(store) == len(df) == 4

    runs = store.load_runs()
    assert set(runs["model_name"]) == set(SINDY)
    assert runs["batch_id"].nunique() == 1 and runs["git_commit"].notna().all()
    # measured features and generation settings come back as separate columns
    assert "feat.noise_rel_mean" in runs and "spec.noise" in runs
    assert "spec.params.mu" in runs  # nested settings are flattened
    # scores survive unchanged
    a = df.sort_values(["dataset_id", "model_name"])["reconstruction_error"].to_numpy()
    b = runs.sort_values(["dataset_id", "model_name"])["reconstruction_error"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_failures_and_nan_scores_are_stored(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    store.log(evaluate(Crashing("crash"), make_dataset("Van der Pol", n_samples=500)))
    row = store.load_runs().iloc[0]
    assert row["status"] == "failed" and "boom" in row["error"]
    assert np.isnan(row["reconstruction_error"])  # NaN -> NULL -> NaN


def test_store_persists_across_sessions_and_filters(tmp_path):
    path = tmp_path / "runs.sqlite"
    d = make_dataset("Van der Pol", n_samples=800)
    RunStore(path).log(evaluate(CANDIDATES["SINDy-poly"], d))
    s2 = RunStore(path)  # new session = new batch, same file
    s2.log(evaluate(CANDIDATES["SINDy-trig"], d))
    assert len(s2) == 2
    assert s2.load_runs()["batch_id"].nunique() == 2
    only = s2.load_runs(where="model_name = ?", params=("SINDy-trig",))
    assert list(only["model_name"]) == ["SINDy-trig"]


def test_new_score_column_is_migrated_into_old_file(tmp_path, monkeypatch):
    path = tmp_path / "runs.sqlite"
    RunStore(path)
    import engine.store as st
    monkeypatch.setitem(st.SCORE_COLUMNS, "new_axis", "REAL")
    RunStore(path)
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(runs)")}
    assert "new_axis" in cols


def test_csv_export(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    store.log(evaluate(CANDIDATES["SINDy-poly"], make_dataset("Van der Pol", n_samples=800)))
    out = store.export_csv(tmp_path / "runs.csv")
    assert out.read_text().startswith("run_id,")

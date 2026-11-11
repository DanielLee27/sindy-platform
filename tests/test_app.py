"""
Headless tests of the Streamlit pages (streamlit.testing.v1.AppTest).

They run every page's script without a browser, click its buttons and
check that nothing raised. SINDY_RUNS_DB points the app at a temporary
store so the tests never touch results/runs.sqlite.
"""
import time

import pytest
from streamlit.testing.v1 import AppTest

from engine.benchmarks import make_dataset
from engine.candidates import CANDIDATES
from engine.harness import compare
from engine.store import RunStore

TIMEOUT = 120


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    path = tmp_path / "runs.sqlite"
    monkeypatch.setenv("SINDY_RUNS_DB", str(path))
    return path


def _no_errors(at):
    assert not at.exception, [e.value for e in at.exception]


def test_explorer_default_run():
    at = AppTest.from_file("views/explorer.py", default_timeout=TIMEOUT).run()
    _no_errors(at)
    at.button[0].click().run()
    _no_errors(at)
    assert any("True system" in m.value for m in at.markdown)


def test_explorer_pendulum_defaults_to_poly_trig_and_recovers_sin():
    at = AppTest.from_file("views/explorer.py", default_timeout=TIMEOUT).run()
    at.selectbox[0].select("Pendulum").run()
    model_box = next(s for s in at.selectbox if s.label == "Model")
    assert model_box.value == "SINDy (poly + trig)"
    at.button[0].click().run()
    _no_errors(at)
    assert any("sin(1 theta)" in c.value for c in at.code)


def test_history_page_empty_and_with_runs(tmp_store):
    at = AppTest.from_file("views/history.py", default_timeout=TIMEOUT).run()
    _no_errors(at)
    assert any("No runs logged yet" in i.value for i in at.info)

    sindy = {k: CANDIDATES[k] for k in ("SINDy-poly", "SINDy-poly+trig")}
    datasets = [make_dataset("Pendulum", n_samples=1000),
                make_dataset("Pendulum", noise=0.01, n_samples=1000)]
    compare(sindy, datasets, store=RunStore(tmp_store), verbose=False)

    at = AppTest.from_file("views/history.py", default_timeout=TIMEOUT).run()
    _no_errors(at)
    for metric in ("term_f1", "stable", "fit_seconds"):  # linear, derived, log scale
        at.selectbox(key="hist_metric").select(metric).run()
        _no_errors(at)


def test_comparison_page_runs_a_background_job(tmp_store):
    at = AppTest.from_file("views/comparison.py", default_timeout=TIMEOUT).run()
    _no_errors(at)
    at.multiselect[0].set_value(["Van der Pol"]).run()                 # systems
    at.multiselect[1].set_value([0.0]).run()                           # noise
    at.number_input[1].set_value(800).run()                            # length
    models = next(m for m in at.multiselect if m.label == "Candidates")
    models.set_value(["SINDy-poly", "SINDy-trig"]).run()
    next(b for b in at.button if b.label == "Run comparison").click().run()
    _no_errors(at)

    deadline = time.time() + TIMEOUT  # the job is a separate Python process
    while time.time() < deadline and not any("runs in batch" in s.value for s in at.success):
        time.sleep(1)
        at.run()
        _no_errors(at)
    assert any("2 runs in batch" in s.value for s in at.success)
    assert RunStore(tmp_store).count() == 2

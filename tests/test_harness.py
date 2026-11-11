"""
Harness and metrics tests. The dummy candidates below are also a proof of
the plug-in design: none of them is SINDy, and the harness scores them
without any changes.
"""
import numpy as np
import pytest

from engine import metrics
from engine.benchmarks import make_dataset
from engine.candidates import CANDIDATES, Candidate, FittedModel

SINDY = {k: CANDIDATES[k] for k in ("SINDy-poly", "SINDy-trig")}
from engine.harness import compare, evaluate
from engine.terms import compile_rhs, term_function


class Oracle(Candidate):
    """Returns the exact true equations: the best score achievable."""
    family = "test"

    def __init__(self, truth):
        super().__init__("oracle")
        self.truth = truth

    def fit(self, t, X, var_names):
        c = self.truth
        return FittedModel(
            rhs=lambda x: metrics.evaluate_terms(c, np.atleast_2d(x), var_names)[0],
            n_terms=sum(len(v) for v in c.values()), symbolic=True, coefficients=c)


class Exploding(Candidate):
    """dx/dt = 5x: grows without bound."""
    family = "test"

    def fit(self, t, X, var_names):
        return FittedModel(rhs=lambda x: 5.0 * np.asarray(x), n_terms=X.shape[1], symbolic=True)


class Crashing(Candidate):
    family = "test"

    def fit(self, t, X, var_names):
        raise RuntimeError("boom")


class BlackBox(Candidate):
    family = "test"

    def fit(self, t, X, var_names):
        return FittedModel(rhs=lambda x: np.zeros_like(x), n_terms=10_000, symbolic=False)


# ---------------------------------------------------------------- terms ---

def test_term_functions_poly_trig_and_constant():
    X = np.array([[2.0, 3.0]])
    names = ["x", "y"]
    assert term_function("x^2 y", names)(X)[0] == pytest.approx(12.0)
    assert term_function("sin(2 y)", names)(X)[0] == pytest.approx(np.sin(6.0))
    assert term_function("1", names)(X)[0] == 1.0
    with pytest.raises(ValueError):
        term_function("tanh(x)", names)


def test_compile_rhs_matches_matrix_product():
    names, terms = ["x", "y"], ["1", "x", "x y", "cos(1 y)"]
    coef = np.array([[0.5, -1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0]])
    x = np.array([1.5, -0.5])
    theta = np.array([1.0, 1.5, -0.75, np.cos(-0.5)])
    np.testing.assert_allclose(compile_rhs(coef, terms, names)(x), coef @ theta)


# -------------------------------------------------------------- metrics ---

def test_structure_recovery_counts():
    truth = {"x": {"x": 1.0, "y": 2.0}}
    rec = {"x": {"x": 1.0, "z": 0.5}}
    s = metrics.structure_recovery(rec, truth)
    assert s["term_precision"] == 0.5 and s["term_recall"] == 0.5
    assert np.isnan(metrics.structure_recovery(None, truth)["term_f1"])


def test_valid_time_and_interpretability():
    t = np.linspace(0, 10, 11)
    true = np.zeros((11, 1))
    pred = np.zeros((11, 1)); pred[6:] = 1.0  # error jumps at t=6
    assert metrics.valid_time(pred, true, t, scale=np.array([1.0])) == 6.0
    assert metrics.interpretability(False, 3, 3) == "black-box"
    assert metrics.interpretability(True, 12, 3) == "parsimonious"
    assert metrics.interpretability(True, 13, 3) == "complex"


# -------------------------------------------------------------- harness ---

@pytest.mark.parametrize("system", ["Lorenz", "Van der Pol"])
def test_oracle_scores_perfectly(system):
    d = make_dataset(system)
    r = evaluate(Oracle(d.true_coefficients), d)
    assert r["status"] == "ok" and r["stability"] == "stable"
    assert r["reconstruction_error"] < 1e-12
    assert r["valid_time_frac"] > 0.99
    assert r["term_f1"] == 1.0


def test_sindy_poly_on_clean_data_is_accurate_and_stable():
    d = make_dataset("Lotka-Volterra")
    r = evaluate(CANDIDATES["SINDy-poly"], d)
    assert r["stability"] == "stable" and r["interpretability"] == "parsimonious"
    assert r["reconstruction_error"] < 1e-3 and r["forecast_nrmse"] < 1e-2
    assert r["n_terms"] == 4 and r["fit_seconds"] > 0


def test_bad_models_are_scored_as_bad():
    d = make_dataset("Van der Pol", n_samples=2000)
    boom = evaluate(Exploding("explode"), d)
    assert boom["stability"] == "diverged" and boom["stability_frac"] < 1.0
    assert np.isnan(boom["forecast_nrmse"])
    assert boom["reconstruction_error"] > 1.0

    bb = evaluate(BlackBox("bb"), d)
    assert bb["interpretability"] == "black-box"
    assert np.isnan(bb["term_f1"])  # no coefficients -> structure undefined


def test_a_crash_is_recorded_and_does_not_stop_the_run():
    d = make_dataset("Van der Pol", n_samples=1000)
    df = compare([Crashing("crash"), CANDIDATES["SINDy-poly"]], [d], verbose=False)
    crash, ok = df.iloc[0], df.iloc[1]
    assert crash["status"] == "failed" and "boom" in crash["error"]
    assert ok["status"] == "ok"


# ----------------------------------------------------------- comparison ---

def test_poly_beats_trig_on_polynomial_systems_and_table_is_side_by_side():
    from engine.benchmarks import make_grid
    from engine.harness import comparison_table

    polynomial = ["Lorenz", "Lotka-Volterra", "Van der Pol", "Rossler"]
    df = compare(SINDY, make_grid(systems=polynomial), verbose=False)
    assert set(df["model_name"]) == {"SINDy-poly", "SINDy-trig"}
    tab = comparison_table(df)
    recon = tab["reconstruction_error"]
    assert (recon["SINDy-poly"] < recon["SINDy-trig"]).all()
    assert len(tab) == 4  # one row per dataset


def test_table_aggregates_seeds_instead_of_dropping_them():
    from engine.benchmarks import make_grid
    from engine.harness import comparison_table

    grid = make_grid(systems=["Van der Pol"], noises=(0.05,), lengths=(1500,), seeds=(0, 1, 2))
    df = compare({"SINDy-poly": CANDIDATES["SINDy-poly"]}, grid, verbose=False)
    tab = comparison_table(df)
    assert len(tab) == 1  # three seeds -> one row
    assert tab[("stability", "SINDy-poly")].iloc[0] == "3/3"
    median = df["reconstruction_error"].median()
    assert tab[("reconstruction_error", "SINDy-poly")].iloc[0] == median


def test_table_works_on_stored_runs(tmp_path):
    from engine.benchmarks import make_dataset
    from engine.harness import comparison_table
    from engine.store import RunStore

    store = RunStore(tmp_path / "r.sqlite")
    compare(SINDY, [make_dataset("Van der Pol", n_samples=800)], store=store, verbose=False)
    tab = comparison_table(store.load_runs())
    assert set(tab["n_terms"].columns) == set(SINDY)

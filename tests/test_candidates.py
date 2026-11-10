import numpy as np
import pytest

from engine.benchmarks import make_dataset
from engine.candidates import CANDIDATES, SindyCandidate
from pipeline.systems import DEMO_SYSTEMS

POLY = CANDIDATES["SINDy-poly"]


@pytest.mark.parametrize("system", list(DEMO_SYSTEMS))
@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_registered_poly_recovers_exact_structure(system, noise):
    d = make_dataset(system, noise=noise)
    m = POLY.fit(d.t_train, d.X_train, d.var_names)
    assert m.symbolic and m.n_terms > 0
    for v in d.var_names:
        assert set(m.coefficients[v]) == set(d.true_coefficients[v]), f"d{v}/dt"


def _rhs_matches_clean_derivative(model, d, tol):
    dX = np.gradient(d.X_clean, d.t, axis=0)
    idx = np.linspace(10, d.split - 10, 25, dtype=int)
    pred = np.array([model.rhs(d.X_clean[k]) for k in idx])
    rel = np.linalg.norm(pred - dX[idx]) / np.linalg.norm(dX[idx])
    assert rel < tol


def test_rhs_returns_derivatives_in_data_units():
    d = make_dataset("Lorenz")
    _rhs_matches_clean_derivative(POLY.fit(d.t_train, d.X_train, d.var_names), d, 0.01)


def test_scaled_fit_still_returns_derivatives_in_data_units():
    # Scaling is opt-in; when used, rhs must undo it so the harness sees
    # data units. Coefficients are withheld (not comparable to ground truth).
    d = make_dataset("Van der Pol")
    m = SindyCandidate("scaled", scale=True, alpha=0.0).fit(d.t_train, d.X_train, d.var_names)
    assert m.coefficients is None
    _rhs_matches_clean_derivative(m, d, 0.05)


def test_scaling_is_off_by_default():
    assert SindyCandidate("x").config["scale"] is False


def test_unknown_library_raises():
    d = make_dataset("Van der Pol", n_samples=500)
    with pytest.raises(ValueError, match="Unknown library"):
        SindyCandidate("bad", library="wavelet").fit(d.t_train, d.X_train, d.var_names)

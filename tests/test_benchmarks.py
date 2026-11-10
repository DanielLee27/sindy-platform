import json

import numpy as np
import pytest

from engine.benchmarks import make_dataset, make_grid
from pipeline.systems import DEMO_SYSTEMS


def test_shape_split_and_determinism():
    a = make_dataset("Lorenz", noise=0.05, n_samples=1000, seed=3, train_frac=0.75)
    b = make_dataset("Lorenz", noise=0.05, n_samples=1000, seed=3, train_frac=0.75)
    assert a.X.shape == (1000, 3) and a.split == 750
    assert a.dataset_id == b.dataset_id
    np.testing.assert_array_equal(a.X, b.X)
    c = make_dataset("Lorenz", noise=0.05, n_samples=1000, seed=4, train_frac=0.75)
    assert c.dataset_id != a.dataset_id


@pytest.mark.parametrize("system", list(DEMO_SYSTEMS))
def test_noise_is_relative_to_each_variable(system):
    d = make_dataset(system, noise=0.1, seed=0)
    realised = (d.X - d.X_clean).std(axis=0) / d.X_clean.std(axis=0)
    np.testing.assert_allclose(realised, 0.1, rtol=0.1)


@pytest.mark.parametrize("system", list(DEMO_SYSTEMS))
@pytest.mark.parametrize("noise", [0.01, 0.05, 0.2])
def test_noise_estimate_tracks_injected_noise_when_well_sampled(system, noise):
    d = make_dataset(system, noise=noise, dt=0.01)
    assert d.features["noise_rel_mean"] == pytest.approx(noise, rel=0.15)


def test_features_are_json_safe_and_resolve_lorenz_timescale():
    d = make_dataset("Lorenz", dt=0.01)
    json.dumps(d.features)
    # Lorenz's fast oscillation is ~1.3 Hz -> ~75 samples per period at dt=0.01
    assert 50 < d.features["samples_per_period"] < 120


def test_grid_is_cross_product():
    grid = make_grid(systems=["Van der Pol", "Rossler"], noises=(0.0, 0.1), lengths=(500,))
    assert len(grid) == 4
    assert len({d.dataset_id for d in grid}) == 4

"""Every demo system simulates cleanly and SINDy recovers its true terms."""
import numpy as np
import pytest

from pipeline.runner import run_pipeline
from pipeline.systems import DEMO_SYSTEMS


@pytest.mark.parametrize("name", list(DEMO_SYSTEMS))
def test_ground_truth_is_recovered(name):
    s = DEMO_SYSTEMS[name]
    t, X = s.generate(s.t_end, s.dt, 0.0, 0, **s.params)
    assert np.isfinite(X).all()

    r = run_pipeline(t, X, s.var_names, model_params=s.suggested, simulate=False)
    names = r["feature_library_names"]
    truth = s.true_coefficients(**s.params)
    for i, v in enumerate(s.var_names):
        recovered = {names[j] for j in np.flatnonzero(r["coefficients"][i])}
        assert recovered == set(truth[v]), f"{name} d{v}/dt"

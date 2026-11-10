"""
Benchmark generator: synthetic datasets with KNOWN ground truth.

make_dataset() builds one Dataset from a system in pipeline.systems with
tunable noise, sampling interval (dt), length, seed and initial condition.
make_grid() builds the cross product of several settings.

Noise is RELATIVE: noise=0.05 adds Gaussian noise with std equal to 5% of
each variable's own standard deviation. That makes a noise level mean the
same thing on Lorenz (range ±20) as on Van der Pol (range ±2).
"""
import hashlib
import itertools
import json
from dataclasses import dataclass, field

import numpy as np

from engine.features import measure_features
from pipeline.systems import DEMO_SYSTEMS


@dataclass
class Dataset:
    system: str
    var_names: list[str]
    t: np.ndarray
    X: np.ndarray            # noisy observations: what models get to see
    X_clean: np.ndarray      # noise-free trajectory: what scores are measured against
    split: int               # index where the test segment starts
    spec: dict               # generation settings (known only for synthetic data)
    true_coefficients: dict  # {var: {term: coef}}
    features: dict = field(default_factory=dict)  # measured from X alone

    @property
    def dataset_id(self):
        """Deterministic id: the same settings always give the same id."""
        blob = json.dumps(self.spec, sort_keys=True).encode()
        return f"{self.system}-{hashlib.sha1(blob).hexdigest()[:10]}"

    @property
    def t_train(self):
        return self.t[: self.split]

    @property
    def X_train(self):
        return self.X[: self.split]


def make_dataset(system, noise=0.0, dt=0.01, n_samples=5000, seed=0,
                 train_frac=0.8, params=None, x0=None):
    """
    system     — key in pipeline.systems.DEMO_SYSTEMS
    noise      — relative noise level (fraction of each variable's std)
    dt         — sampling interval
    n_samples  — number of time points (default 5000: at 2000, Rossler covers
                 under 3 orbits and its z-dynamics are barely excited)
    seed       — RNG seed for the noise
    train_frac — fraction of samples used for fitting; the rest is the
                 held-out forecast segment
    params     — physical parameters (defaults from the system)
    x0         — initial condition (default from the system)
    """
    s = DEMO_SYSTEMS[system]
    params = {**s.params, **(params or {})}
    x0 = tuple(float(v) for v in (x0 if x0 is not None else s.x0))

    # Simulate slightly past the end, then trim to exactly n_samples
    t, X_clean = s.generate((n_samples + 1) * dt, dt, 0.0, 0, x0=x0, **params)
    if len(t) < n_samples or not np.isfinite(X_clean).all():
        raise RuntimeError(f"{system}: simulation failed or ended early")
    t, X_clean = t[:n_samples], X_clean[:n_samples]

    rng = np.random.default_rng(seed)
    sigma = noise * X_clean.std(axis=0)
    X = X_clean + rng.normal(size=X_clean.shape) * sigma

    spec = dict(system=system, noise=float(noise), dt=float(dt), n_samples=int(n_samples),
                seed=int(seed), train_frac=float(train_frac), params=params, x0=list(x0))

    return Dataset(
        system=system,
        var_names=list(s.var_names),
        t=t,
        X=X,
        X_clean=X_clean,
        split=int(round(train_frac * n_samples)),
        spec=spec,
        true_coefficients=s.true_coefficients(**params),
        features=measure_features(t, X),
    )


def make_grid(systems=None, noises=(0.0,), dts=(0.01,), lengths=(5000,), seeds=(0,), **kw):
    """Every combination of the given settings, as a list of Datasets."""
    systems = systems or list(DEMO_SYSTEMS)
    return [
        make_dataset(sys_, noise=nz, dt=dt, n_samples=n, seed=sd, **kw)
        for sys_, nz, dt, n, sd in itertools.product(systems, noises, dts, lengths, seeds)
    ]

"""
Demo dynamical systems for the app.

Each system is one DemoSystem entry in DEMO_SYSTEMS. The app builds its
dropdown and parameter inputs from this dict, so adding a new system means
adding one entry here and nothing else.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from pipeline.loader import generate_lorenz


@dataclass(frozen=True)
class DemoSystem:
    var_names: list[str]
    # generate(t_end, dt, noise_level, seed, **params) -> (t, X)
    generate: Callable
    # true_equations(**params) -> list of right-hand sides, one per variable
    true_equations: Callable
    params: dict                      # physical parameters and their defaults
    t_end: float = 10.0
    dt: float = 0.01
    suggested: dict = field(default_factory=dict)  # suggested model settings
    description: str = ""


def simulate_system(rhs, x0, t_end, dt, noise_level=0.0, seed=0):
    """Integrate dx/dt = rhs(t, x) from x0 and optionally add Gaussian noise."""
    t = np.arange(0, t_end, dt)
    sol = solve_ivp(rhs, (0, t_end), x0, t_eval=t, method="RK45", rtol=1e-10, atol=1e-10)
    X = sol.y.T
    if noise_level > 0:
        X = X + np.random.default_rng(seed).normal(0, noise_level, X.shape)
    return sol.t, X


# --- Lorenz: reuses the existing generator in loader.py -----------------------

def _lorenz_generate(t_end, dt, noise_level, seed, sigma, rho, beta):
    # generate_lorenz uses the global NumPy RNG, so seed it for reproducibility
    np.random.seed(seed)
    return generate_lorenz(t_end=t_end, dt=dt, sigma=sigma, rho=rho, beta=beta,
                           noise_level=noise_level)


def _lorenz_true(sigma, rho, beta):
    return [
        f"{-sigma:g} x + {sigma:g} y",
        f"{rho:g} x + -1 y + -1 x z",
        f"{-beta:.3f} z + 1 x y",
    ]


# --- Lotka-Volterra (predator-prey) -------------------------------------------

def _lv_generate(t_end, dt, noise_level, seed, a, b, d, g):
    def rhs(t, s):
        x, y = s
        return [a * x - b * x * y, d * x * y - g * y]
    return simulate_system(rhs, [6.0, 3.0], t_end, dt, noise_level, seed)


def _lv_true(a, b, d, g):
    return [f"{a:g} x + {-b:g} x y", f"{-g:g} y + {d:g} x y"]


# --- Van der Pol oscillator ---------------------------------------------------

def _vdp_generate(t_end, dt, noise_level, seed, mu):
    def rhs(t, s):
        x, y = s
        return [y, mu * (1 - x**2) * y - x]
    return simulate_system(rhs, [2.0, 0.0], t_end, dt, noise_level, seed)


def _vdp_true(mu):
    return ["1 y", f"-1 x + {mu:g} y + {-mu:g} x^2 y"]


DEMO_SYSTEMS = {
    "Lorenz": DemoSystem(
        var_names=["x", "y", "z"],
        generate=_lorenz_generate,
        true_equations=_lorenz_true,
        params={"sigma": 10.0, "rho": 28.0, "beta": 8 / 3},
        t_end=10.0,
        suggested={"degree": 2, "threshold": 0.2},
        description="Chaotic 3-variable system; the classic SINDy benchmark.",
    ),
    "Lotka-Volterra": DemoSystem(
        var_names=["x", "y"],
        generate=_lv_generate,
        true_equations=_lv_true,
        params={"a": 1.0, "b": 0.5, "d": 0.25, "g": 1.0},
        t_end=20.0,
        suggested={"degree": 2, "threshold": 0.05},
        description="Predator-prey: x = prey, y = predators. Periodic orbits.",
    ),
    "Van der Pol": DemoSystem(
        var_names=["x", "y"],
        generate=_vdp_generate,
        true_equations=_vdp_true,
        params={"mu": 1.0},
        t_end=20.0,
        suggested={"degree": 3, "threshold": 0.1},
        description="Self-sustained oscillator. Has an x²y term, so it needs degree 3.",
    ),
}


if __name__ == "__main__":
    for name, sys_ in DEMO_SYSTEMS.items():
        t, X = sys_.generate(sys_.t_end, sys_.dt, 0.0, 0, **sys_.params)
        print(f"{name}: X shape {X.shape}, true eqs: {sys_.true_equations(**sys_.params)}")

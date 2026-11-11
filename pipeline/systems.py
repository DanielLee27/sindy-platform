"""
Demo dynamical systems with known ground truth.

Each system is one DemoSystem entry in DEMO_SYSTEMS. Both the Streamlit app
and the comparison engine (engine/benchmarks.py) read from this dict, so
there is exactly one definition of what "Lorenz" etc. means.

Ground truth is stored as data: true_coefficients(**params) returns
    {variable: {library_term: coefficient}}
using PySINDy's term names ("1", "x", "x y", "x^2 y", ...), so recovered
models can be compared against it term by term.
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
    # true_coefficients(**params) -> {var: {term: coef}} in PySINDy term names
    true_coefficients: Callable
    params: dict                      # physical parameters and their defaults
    x0: tuple                         # default initial condition
    t_end: float = 10.0
    dt: float = 0.01
    suggested: dict = field(default_factory=dict)  # suggested model settings
    description: str = ""
    suggested_model: str = "SINDy (STLSQ)"         # key in registry.MODELS

    def true_equations(self, **params):
        """Right-hand sides as display strings, one per variable."""
        coefs = self.true_coefficients(**(params or self.params))
        return [format_rhs(coefs[v]) for v in self.var_names]


def format_rhs(terms):
    """{'x': -10, 'y': 10} -> '-10 x + 10 y' (matches PySINDy's style)."""
    return " + ".join(f"{c:.4g} {term}" for term, c in terms.items())


def simulate_system(rhs, x0, t_end, dt, noise_level=0.0, seed=0):
    """Integrate dx/dt = rhs(t, x) from x0 and optionally add Gaussian noise."""
    t = np.arange(0, t_end, dt)
    sol = solve_ivp(rhs, (0, t_end), x0, t_eval=t, method="RK45", rtol=1e-10, atol=1e-10)
    X = sol.y.T
    if noise_level > 0:
        X = X + np.random.default_rng(seed).normal(0, noise_level, X.shape)
    return sol.t, X


# --- Lorenz: reuses the existing generator in loader.py -----------------------

def _lorenz_generate(t_end, dt, noise_level, seed, sigma, rho, beta, x0=None):
    if x0 is not None and tuple(x0) != (1.0, 1.0, 1.0):
        # loader.generate_lorenz has a fixed x0, so use the generic path instead
        def rhs(t, s):
            x, y, z = s
            return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
        return simulate_system(rhs, list(x0), t_end, dt, noise_level, seed)
    # generate_lorenz uses the global NumPy RNG, so seed it for reproducibility
    np.random.seed(seed)
    return generate_lorenz(t_end=t_end, dt=dt, sigma=sigma, rho=rho, beta=beta,
                           noise_level=noise_level)


def _lorenz_true(sigma, rho, beta):
    return {
        "x": {"x": -sigma, "y": sigma},
        "y": {"x": rho, "y": -1.0, "x z": -1.0},
        "z": {"z": -beta, "x y": 1.0},
    }


# --- Lotka-Volterra (predator-prey) -------------------------------------------

def _lv_generate(t_end, dt, noise_level, seed, a, b, d, g, x0=(6.0, 3.0)):
    def rhs(t, s):
        x, y = s
        return [a * x - b * x * y, d * x * y - g * y]
    return simulate_system(rhs, list(x0), t_end, dt, noise_level, seed)


def _lv_true(a, b, d, g):
    return {"x": {"x": a, "x y": -b}, "y": {"y": -g, "x y": d}}


# --- Van der Pol oscillator ---------------------------------------------------

def _vdp_generate(t_end, dt, noise_level, seed, mu, x0=(2.0, 0.0)):
    def rhs(t, s):
        x, y = s
        return [y, mu * (1 - x**2) * y - x]
    return simulate_system(rhs, list(x0), t_end, dt, noise_level, seed)


def _vdp_true(mu):
    return {"x": {"y": 1.0}, "y": {"x": -1.0, "y": mu, "x^2 y": -mu}}


# --- Rössler attractor --------------------------------------------------------

def _rossler_generate(t_end, dt, noise_level, seed, a, b, c, x0=(1.0, 1.0, 0.0)):
    def rhs(t, s):
        x, y, z = s
        return [-y - z, x + a * y, b + z * (x - c)]
    return simulate_system(rhs, list(x0), t_end, dt, noise_level, seed)


def _rossler_true(a, b, c):
    return {
        "x": {"y": -1.0, "z": -1.0},
        "y": {"x": 1.0, "y": a},
        "z": {"1": b, "z": -c, "x z": 1.0},
    }


# --- Pendulum (nonlinear, trigonometric) --------------------------------------
# The only demo system whose true equations are NOT polynomial: the restoring
# force is sin(theta). It gives trig libraries a case they should win; a
# polynomial library can only approximate sin(theta) by a Taylor series. The
# default swing (2.5 rad, about 143 degrees) is far outside the small-angle
# regime where sin(theta) ~ theta.
#
# Why damping is ON by default: an undamped pendulum conserves energy,
# omega^2/2 - (g/L) cos(theta) = const, so the library columns 1, omega^2 and
# cos(theta) are exactly linearly dependent along the trajectory. A poly+trig
# library then has infinitely many equally good fits (unidentifiable).
# Damping 0.2 breaks that and stays above the benchmark's STLSQ threshold
# (0.15), so the damping term is recoverable. Side effect: by the forecast
# window (last 20% of a 50 s run) the swing has decayed to ~0.04 rad, so
# forecast scores separate models less on this system than reconstruction
# and structure recovery do.

def _pendulum_generate(t_end, dt, noise_level, seed, g_over_L, damping, x0=(2.5, 0.0)):
    def rhs(t, s):
        theta, omega = s
        return [omega, -g_over_L * np.sin(theta) - damping * omega]
    return simulate_system(rhs, list(x0), t_end, dt, noise_level, seed)


def _pendulum_true(g_over_L, damping):
    omega_terms = {"sin(1 theta)": -g_over_L}
    if damping:
        omega_terms["omega"] = -damping
    return {"theta": {"omega": 1.0}, "omega": omega_terms}


DEMO_SYSTEMS = {
    "Lorenz": DemoSystem(
        var_names=["x", "y", "z"],
        generate=_lorenz_generate,
        true_coefficients=_lorenz_true,
        params={"sigma": 10.0, "rho": 28.0, "beta": 8 / 3},
        x0=(1.0, 1.0, 1.0),
        t_end=10.0,
        suggested={"degree": 2, "threshold": 0.2},
        description="Chaotic 3-variable system; the classic SINDy benchmark.",
    ),
    "Lotka-Volterra": DemoSystem(
        var_names=["x", "y"],
        generate=_lv_generate,
        true_coefficients=_lv_true,
        params={"a": 1.0, "b": 0.5, "d": 0.25, "g": 1.0},
        x0=(6.0, 3.0),
        t_end=20.0,
        suggested={"degree": 2, "threshold": 0.05},
        description="Predator-prey: x = prey, y = predators. Periodic orbits.",
    ),
    "Van der Pol": DemoSystem(
        var_names=["x", "y"],
        generate=_vdp_generate,
        true_coefficients=_vdp_true,
        params={"mu": 1.0},
        x0=(2.0, 0.0),
        t_end=20.0,
        suggested={"degree": 3, "threshold": 0.1},
        description="Self-sustained oscillator. Has an x²y term, so it needs degree 3.",
    ),
    "Rossler": DemoSystem(
        var_names=["x", "y", "z"],
        generate=_rossler_generate,
        true_coefficients=_rossler_true,
        params={"a": 0.2, "b": 0.2, "c": 5.7},
        x0=(1.0, 1.0, 0.0),
        t_end=50.0,
        suggested={"degree": 2, "threshold": 0.05},
        description="Chaotic attractor with one nonlinear term (xz); spiky z.",
    ),
    "Pendulum": DemoSystem(
        var_names=["theta", "omega"],
        generate=_pendulum_generate,
        true_coefficients=_pendulum_true,
        params={"g_over_L": 4.0, "damping": 0.2},
        x0=(2.5, 0.0),
        t_end=20.0,
        suggested={"degree": 1, "n_frequencies": 1, "threshold": 0.1},
        suggested_model="SINDy (poly + trig)",
        description="Damped large-angle pendulum: d(omega)/dt = -(g/L) sin(theta) "
                    "- c omega. Not polynomial, so it needs a trig library.",
    ),
}


if __name__ == "__main__":
    for name, sys_ in DEMO_SYSTEMS.items():
        t, X = sys_.generate(sys_.t_end, sys_.dt, 0.0, 0, **sys_.params)
        print(f"{name}: X shape {X.shape}, finite={np.isfinite(X).all()}")
        for v, rhs in zip(sys_.var_names, sys_.true_equations()):
            print(f"   d{v}/dt = {rhs}")

"""
Registries of models and plots available in the app.

- To add a model: write a fit function with the signature
      fn(X, t, X_dot, feature_names, **params) -> dict
  that returns at least "equations" and "score", then add a ModelSpec below.
  Its `params` dict is turned into sidebar widgets automatically.

- To add a plot: add a PlotSpec whose `fn` takes the run-result dict and
  returns a matplotlib Figure. The app creates one tab per plot.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from pipeline.models import run_sindy, simulate_model
from pipeline.visualize import (
    plot_phase_portrait,
    plot_recovered_vs_true,
    plot_trajectories,
)


# ----------------------------------------------------------------- models ---

@dataclass(frozen=True)
class Param:
    """Describes one tunable setting so the UI can render a widget for it."""
    kind: str            # "float" or "int"
    default: float
    min: float
    max: float
    step: float = 0.01
    help: str = ""


@dataclass(frozen=True)
class ModelSpec:
    fn: Callable
    params: dict[str, Param]
    # simulate(fit_result, x0, t) -> X_sim; None if the model can't simulate
    simulate: Optional[Callable] = None
    description: str = ""


MODELS = {
    "SINDy (STLSQ)": ModelSpec(
        fn=run_sindy,
        simulate=simulate_model,
        params={
            "degree": Param("int", 2, 1, 5, step=1,
                            help="Max polynomial degree of the candidate library."),
            "threshold": Param("float", 0.1, 0.0, 2.0, step=0.01,
                               help="STLSQ drops coefficients smaller than this."),
            "alpha": Param("float", 0.05, 0.0, 1.0, step=0.01,
                           help="Ridge regularization strength."),
        },
        description="Polynomial library + sequentially thresholded least squares.",
    ),
}


# ------------------------------------------------------------------ plots ---

@dataclass(frozen=True)
class PlotSpec:
    fn: Callable                      # fn(result) -> matplotlib Figure
    requires: tuple = ()              # result keys that must be non-None
    description: str = ""


# Adapters: each wraps an unchanged visualize.py function, passing the
# arguments it needs from the run-result dict.
PLOTS = {
    "Phase portrait": PlotSpec(
        fn=lambda r: plot_phase_portrait(r["X"], title="Phase Portrait (data)",
                                         var_names=r["var_names"]),
        description="Every pair of variables plotted against each other.",
    ),
    "Trajectories": PlotSpec(
        fn=lambda r: plot_trajectories(r["t"], r["X"], title="State Trajectories (data)",
                                       var_names=r["var_names"]),
        description="Each variable over time.",
    ),
    "Recovered vs true": PlotSpec(
        fn=lambda r: plot_recovered_vs_true(r["t"], r["X"], r["X_sim"],
                                            var_names=r["var_names"]),
        requires=("X_sim",),
        description="Data vs. a forward simulation of the recovered equations.",
    ),
}

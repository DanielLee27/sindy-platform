import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_phase_portrait(X, title="Phase Portrait", var_names=None):
    """
    Plot 2D phase portraits for all variable pairs.
    For a 3-variable system, this gives x-y, x-z, y-z.
    """
    n_vars = X.shape[1]
    if var_names is None:
        var_names = [f"x{i}" for i in range(n_vars)]

    pairs = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars)]
    n_plots = len(pairs)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, (i, j) in zip(axes, pairs):
        ax.plot(X[:, i], X[:, j], lw=0.5, alpha=0.7, color='steelblue')
        ax.set_xlabel(var_names[i])
        ax.set_ylabel(var_names[j])
        ax.set_title(f"{var_names[i]} vs {var_names[j]}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_trajectories(t, X, title="State Trajectories", var_names=None):
    """
    Plot each variable over time.
    """
    n_vars = X.shape[1]
    if var_names is None:
        var_names = [f"x{i}" for i in range(n_vars)]

    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    colors = ['steelblue', 'tomato', 'seagreen', 'mediumpurple', 'orange']

    for i, ax in enumerate(axes):
        ax.plot(t, X[:, i], lw=1.2, color=colors[i % len(colors)])
        ax.set_ylabel(var_names[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_recovered_vs_true(t, X_true, X_sim, var_names=None, title="Recovered vs True"):
    """
    Overlay true trajectory with SINDy-simulated trajectory.
    Shows how well the recovered equations reproduce the data.
    """
    n_vars = X_true.shape[1]
    if var_names is None:
        var_names = [f"x{i}" for i in range(n_vars)]

    # Trim to the shorter of the two (simulation may diverge early)
    n = min(len(t), len(X_sim))
    t = t[:n]
    X_true = X_true[:n]
    X_sim = X_sim[:n]

    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, X_true[:, i], lw=1.5, label='True', color='steelblue')
        ax.plot(t, X_sim[:, i], lw=1.5, label='SINDy', linestyle='--', color='tomato')
        ax.set_ylabel(var_names[i])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def save_figure(fig, path):
    """Save a matplotlib figure to disk."""
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {path}")


# Quick test — generates and saves plots to data/
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from pipeline.loader import generate_lorenz
    from pipeline.preprocess import scale_data, estimate_derivatives
    from pipeline.models import run_sindy, simulate_model

    t, X = generate_lorenz(t_end=10.0, dt=0.01)
    X_scaled, scaler = scale_data(X)
    X_dot = estimate_derivatives(X_scaled, t, method='smoothed_fd')
    result = run_sindy(X_scaled, t, X_dot=X_dot, feature_names=['x', 'y', 'z'])
    X_sim = simulate_model(result, X_scaled[0], t)

    fig1 = plot_phase_portrait(X_scaled, title="Lorenz — Phase Portrait", var_names=['x', 'y', 'z'])
    save_figure(fig1, "data/phase_portrait.png")

    fig2 = plot_trajectories(t, X_scaled, title="Lorenz — Trajectories", var_names=['x', 'y', 'z'])
    save_figure(fig2, "data/trajectories.png")

    fig3 = plot_recovered_vs_true(t, X_scaled, X_sim, var_names=['x', 'y', 'z'])
    save_figure(fig3, "data/recovered_vs_true.png")

    print("\nAll plots saved to data/. Open them to inspect.")
    print("visualize.py works correctly.")
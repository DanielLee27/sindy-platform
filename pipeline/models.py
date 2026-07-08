import numpy as np
import pysindy as ps


def run_sindy(X, t, X_dot=None, feature_names=None, threshold=0.1, alpha=0.05):
    """
    Run SINDy on scaled data and return recovered equations + model object.

    Parameters:
      X             — state data, shape [n_timesteps, n_vars]
      t             — time array
      X_dot         — precomputed derivatives (optional; SINDy estimates if None)
      feature_names — list of variable names, e.g. ['x', 'y', 'z']
      threshold     — sparsity threshold (higher = fewer terms kept)
      alpha         — regularization strength

    Returns: dict with model, equations, and score
    """
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    # Build the feature library (polynomial terms up to degree 2)
    library = ps.PolynomialLibrary(degree=2)

    # Optimizer: STLSQ (standard SINDy sparse regression)
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)

    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer
    )
    model.feature_names = feature_names

    if X_dot is not None:
        model.fit(X, t=t, x_dot=X_dot)
    else:
        model.fit(X, t=t)

    # Extract equations as strings
    equations = model.equations(precision=3)

    # Score: R² on derivative prediction
    score = model.score(X, t=t)

    print("\n--- SINDy Results ---")
    model.print()
    print(f"\nR² score: {score:.4f}")
    print("---------------------\n")

    return {
        "model": model,
        "equations": equations,
        "score": score,
        "feature_names": feature_names,
        "coefficients": model.coefficients(),
        "feature_library_names": model.get_feature_names(),
    }


def simulate_model(result, X0, t):
    """
    Simulate the recovered model forward in time from initial condition X0.
    Useful for comparing recovered vs true trajectories.
    
    Returns: X_sim (simulated trajectory)
    """
    model = result["model"]
    X_sim = model.simulate(X0, t)
    return X_sim


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from pipeline.loader import generate_lorenz
    from pipeline.preprocess import scale_data, estimate_derivatives

    t, X = generate_lorenz(t_end=10.0, dt=0.01)
    X_scaled, scaler = scale_data(X)
    X_dot = estimate_derivatives(X_scaled, t, method='smoothed_fd')

    result = run_sindy(
        X_scaled, t, X_dot=X_dot,
        feature_names=['x', 'y', 'z'],
        threshold=0.1
    )

    print("Equations recovered:")
    for i, eq in enumerate(result["equations"]):
        print(f"  d{result['feature_names'][i]}/dt = {eq}")

    print(f"\nCoefficients shape: {result['coefficients'].shape}")
    print("models.py works correctly.")
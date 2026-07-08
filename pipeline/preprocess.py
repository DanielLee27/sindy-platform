import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_data(X):
    """
    Z-score normalize the data (mean=0, std=1 per variable).
    Returns: X_scaled, scaler (keep the scaler to invert later)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Data scaled. Means: {scaler.mean_.round(3)}, Stds: {scaler.scale_.round(3)}")
    return X_scaled, scaler


def estimate_derivatives(X, t, method='finite_difference'):
    """
    Estimate time derivatives of X.
    
    method options:
      'finite_difference' — fast, sensitive to noise (default)
      'smoothed_fd'       — smooth first, then differentiate (better for noisy data)
    
    Returns: X_dot (same shape as X)
    """
    if method == 'finite_difference':
        # Simple central differences
        dt = np.mean(np.diff(t))
        X_dot = np.gradient(X, dt, axis=0)
        print(f"Derivatives estimated via finite difference (dt={dt:.4f})")

    elif method == 'smoothed_fd':
        from scipy.signal import savgol_filter
        # Savitzky-Golay filter: smooths while preserving shape
        # window_length must be odd; 11 is a reasonable default
        dt = np.mean(np.diff(t))
        X_smooth = savgol_filter(X, window_length=11, polyorder=3, axis=0)
        X_dot = np.gradient(X_smooth, dt, axis=0)
        print(f"Derivatives estimated via smoothed finite difference (Savitzky-Golay)")

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'finite_difference' or 'smoothed_fd'.")

    return X_dot


def check_sampling(t):
    """
    Basic diagnostics on the time array.
    Prints whether sampling is regular, and estimates noise level.
    """
    diffs = np.diff(t)
    mean_dt = np.mean(diffs)
    std_dt = np.std(diffs)

    if std_dt / mean_dt < 0.01:
        print(f"Sampling: regular (dt ≈ {mean_dt:.5f})")
    else:
        print(f"Sampling: irregular (mean dt={mean_dt:.5f}, std={std_dt:.5f})")

    return mean_dt


# Quick test
if __name__ == "__main__":
    from loader import generate_lorenz

    t, X = generate_lorenz(t_end=5.0, dt=0.01, noise_level=0.1)

    check_sampling(t)
    X_scaled, scaler = scale_data(X)
    X_dot = estimate_derivatives(X_scaled, t, method='smoothed_fd')

    print(f"X_dot shape: {X_dot.shape}")
    print(f"X_dot max value: {np.abs(X_dot).max():.3f}")
    print("preprocess.py works correctly.")
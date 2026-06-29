import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def generate_lorenz(t_end=10.0, dt=0.01, sigma=10, rho=28, beta=8/3, noise_level=0.0):
    """
    Generate synthetic Lorenz system data.
    Returns: t (time array), X (state matrix of shape [n_timesteps, 3])
    """
    def lorenz(t, state):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)
    x0 = [1.0, 1.0, 1.0]  # initial conditions

    sol = solve_ivp(lorenz, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-10)
    X = sol.y.T  # shape: [n_timesteps, 3]

    if noise_level > 0:
        X += np.random.normal(0, noise_level, X.shape)

    print(f"Generated Lorenz data: {X.shape[0]} timesteps, {X.shape[1]} variables")
    return sol.t, X


def load_csv(filepath, time_column=None):
    """
    Load time-series data from a CSV file.
    - time_column: name of the time column, or None if not present
    Returns: t (time array or None), X (state matrix)
    """
    df = pd.read_csv(filepath)

    # Basic validation
    if df.empty:
        raise ValueError("CSV file is empty.")
    if df.isnull().any().any():
        print("Warning: NaN values found. Rows with NaN will be dropped.")
        df = df.dropna()

    if time_column and time_column in df.columns:
        t = df[time_column].values
        X = df.drop(columns=[time_column]).values
        # Check time is monotonically increasing
        if not np.all(np.diff(t) > 0):
            raise ValueError("Time column is not monotonically increasing.")
    else:
        X = df.values
        t = None
        print("No time column specified. SINDy will assume uniform sampling.")

    print(f"Loaded data: {X.shape[0]} timesteps, {X.shape[1]} variables")
    return t, X


# Quick test — run this file directly to check it works
if __name__ == "__main__":
    t, X = generate_lorenz(t_end=5.0, dt=0.01)
    print(f"t shape: {t.shape}")
    print(f"X shape: {X.shape}")
    print(f"X first row: {X[0]}")
    print("loader.py works correctly.")
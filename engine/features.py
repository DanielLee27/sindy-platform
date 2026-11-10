"""
Dataset features measured from the observed data ALONE.

Nothing here may use ground truth or generation settings. These are the
features a future recommender would see for a real uploaded CSV, so they
must be computable from (t, X) only. Generation settings (true noise,
system name, ...) live separately in Dataset.spec.
"""
import numpy as np
from scipy.signal import savgol_coeffs, savgol_filter

NOISE_WINDOW = 11
NOISE_POLY = 3


def estimate_noise(X, window=NOISE_WINDOW, poly=NOISE_POLY):
    """
    Per-variable noise std, relative to each variable's std.

    Smooth with a Savitzky-Golay filter and look at the residual. The filter
    keeps part of the noise, so the residual underestimates it. For this
    filter, residual variance = sigma² · (1 - h0), where h0 is the filter's
    centre weight, so we divide that factor back out.

    Limitation: when the signal is under-resolved (few samples per period,
    comparable to the window), unresolved dynamics also land in the residual
    and read as noise. Data alone cannot fully separate the two; read this
    together with samples_per_period.
    """
    if len(X) < window:
        return np.full(X.shape[1], np.nan)
    resid = X - savgol_filter(X, window, poly, axis=0)
    h0 = savgol_coeffs(window, poly)[window // 2]
    sigma = resid.std(axis=0) / np.sqrt(1 - h0)
    return sigma / np.maximum(X.std(axis=0), 1e-12)


def dominant_frequency(x, dt):
    """Frequency (cycles per time unit) of the largest non-zero FFT peak."""
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), dt)
    if len(power) < 2:
        return np.nan
    return freqs[1:][np.argmax(power[1:])]


def measure_features(t, X):
    """Return a flat, JSON-safe dict of features for one dataset."""
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    n, d = X.shape
    dts = np.diff(t)
    dt = dts.mean()

    noise = estimate_noise(X)
    # Use the FASTEST variable's peak: sampling must resolve the fastest
    # timescale, and slow variables can put their FFT peak in the lowest bin
    # (e.g. Lorenz x/y lobe switching).
    freqs = np.array([dominant_frequency(X[:, i], dt) for i in range(d)])
    f_dom = np.nanmax(freqs)
    stds = X.std(axis=0)

    feats = {
        "n_vars": d,
        "n_samples": n,
        "dt": dt,
        "duration": t[-1] - t[0],
        "dt_cv": dts.std() / dt,                    # 0 = perfectly regular sampling
        "noise_rel_mean": np.nanmean(noise),        # estimated noise / signal std
        "noise_rel_max": np.nanmax(noise),
        "snr_db": -20 * np.log10(max(np.nanmean(noise), 1e-12)),
        "dominant_freq": f_dom,
        "samples_per_period": 1 / (f_dom * dt) if f_dom > 0 else np.nan,
        "n_periods": (t[-1] - t[0]) * f_dom,        # oscillations observed
        "std_min": stds.min(),
        "std_max": stds.max(),
        "scale_ratio": stds.max() / max(stds.min(), 1e-12),  # spread disparity
    }
    return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in feats.items()}

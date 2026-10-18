import numpy as np


def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion. Returns (low, high) arrays."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    centre = (k + z * z / 2.0) / (n + z * z)
    half = z * np.sqrt(k * (n - k) / n + z * z / 4.0) / (n + z * z)
    # the interval always contains k/n; clip rounding noise so error bars are never negative
    return np.minimum(centre - half, k / n), np.maximum(centre + half, k / n)

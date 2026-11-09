"""Sliding-window feature extraction.

STATUS: IMPLEMENTED, unit-tested against synthetic data.

Design decision (documented, not dataset-dependent): each window is
reduced to 5 summary statistics — mean, std, min, max, slope — per the
approved proposal's feature_set. This keeps the feature-vector
dimensionality fixed at 5 regardless of the (still TBD, dataset-
dependent) window_size chosen later. window_size itself is a caller-
supplied parameter here, never hardcoded, so this module works
correctly for whatever real value Phase-2 dataset inspection produces.
"""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = ["mean", "std", "min", "max", "slope"]
N_FEATURES = len(FEATURE_NAMES)


def _slope(values: np.ndarray) -> float:
    """Least-squares slope of values against their position index."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    # np.polyfit degree-1 returns [slope, intercept]
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def extract_window_features(series: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Slide a window of `window_size` raw points across `series`,
    producing one 5-dim feature row per window.

    Raises ValueError (not a silent empty result) if window_size is
    invalid for the given series length.
    """
    series = np.asarray(series, dtype=float)
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")
    if window_size > len(series):
        raise ValueError(
            f"window_size ({window_size}) exceeds series length ({len(series)})"
        )

    n_windows = (len(series) - window_size) // stride + 1
    features = np.empty((n_windows, N_FEATURES), dtype=float)

    for i in range(n_windows):
        start = i * stride
        window = series[start:start + window_size]
        features[i, 0] = window.mean()
        features[i, 1] = window.std()
        features[i, 2] = window.min()
        features[i, 3] = window.max()
        features[i, 4] = _slope(window)

    return features


class PrefixFittedScaler:
    """A StandardScaler-equivalent fit ONLY on a prefix of the data
    (intended: the pre-drift region), then applied unchanged afterward.

    This directly implements the locked design decision
    (config: preprocessing.scaler = "fit_on_pre_drift_only") — the
    scaler must never be re-fit after drift, since re-fitting would be
    a second, hidden intervention confounding the retraining effect
    under study.
    """

    def __init__(self):
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fitted = False

    def fit(self, X_prefix: np.ndarray) -> "PrefixFittedScaler":
        X_prefix = np.asarray(X_prefix, dtype=float)
        self._mean = X_prefix.mean(axis=0)
        std = X_prefix.std(axis=0)
        # Avoid division by zero for a constant feature column.
        self._std = np.where(std == 0, 1.0, std)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PrefixFittedScaler.transform called before fit()")
        X = np.asarray(X, dtype=float)
        return (X - self._mean) / self._std

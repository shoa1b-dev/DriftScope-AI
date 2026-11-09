"""Unit tests for src/data/preprocessing.py — synthetic data only.

Written as unittest.TestCase for `python -m unittest discover` compatibility.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocessing import (
    N_FEATURES,
    PrefixFittedScaler,
    extract_window_features,
)


class TestExtractWindowFeatures(unittest.TestCase):
    def test_shape(self):
        series = np.arange(100, dtype=float)
        feats = extract_window_features(series, window_size=10)
        # n_windows = (100 - 10) // 1 + 1 = 91
        self.assertEqual(feats.shape, (91, N_FEATURES))

    def test_known_values_flat_window(self):
        # A flat window: mean == value, std == 0, min == max == value, slope == 0
        series = np.full(20, 7.0)
        feats = extract_window_features(series, window_size=5)
        np.testing.assert_allclose(feats[:, 0], 7.0)   # mean
        np.testing.assert_allclose(feats[:, 1], 0.0, atol=1e-9)   # std
        np.testing.assert_allclose(feats[:, 2], 7.0)   # min
        np.testing.assert_allclose(feats[:, 3], 7.0)   # max
        np.testing.assert_allclose(feats[:, 4], 0.0, atol=1e-9)   # slope (float noise ~1e-16 expected)

    def test_slope_direction(self):
        # Strictly increasing series -> positive slope in every window
        series = np.arange(50, dtype=float)
        feats = extract_window_features(series, window_size=10)
        self.assertTrue(np.all(feats[:, 4] > 0))

    def test_rejects_invalid_window(self):
        series = np.arange(5, dtype=float)
        with self.assertRaises(ValueError):
            extract_window_features(series, window_size=10)

    def test_stride_reduces_window_count(self):
        series = np.arange(100, dtype=float)
        feats_stride1 = extract_window_features(series, window_size=10, stride=1)
        feats_stride5 = extract_window_features(series, window_size=10, stride=5)
        self.assertLess(feats_stride5.shape[0], feats_stride1.shape[0])


class TestPrefixFittedScaler(unittest.TestCase):
    def test_fit_transform_roundtrip(self):
        rng = np.random.default_rng(0)
        X_prefix = rng.normal(loc=5.0, scale=2.0, size=(100, N_FEATURES))
        scaler = PrefixFittedScaler().fit(X_prefix)
        transformed = scaler.transform(X_prefix)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-8)

    def test_never_refit_applies_prefix_stats_to_new_data(self):
        rng = np.random.default_rng(1)
        X_prefix = rng.normal(loc=0.0, scale=1.0, size=(100, N_FEATURES))
        scaler = PrefixFittedScaler().fit(X_prefix)

        # Simulate "post-drift" data with a real shift
        X_post_drift = rng.normal(loc=10.0, scale=1.0, size=(50, N_FEATURES))
        transformed = scaler.transform(X_post_drift)

        # Because the scaler was NOT re-fit, the shift must still be
        # visible (proves no silent re-fit happened)
        self.assertGreater(transformed.mean(), 5.0)

    def test_raises_if_transform_before_fit(self):
        scaler = PrefixFittedScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(np.zeros((5, N_FEATURES)))

    def test_handles_constant_feature_column(self):
        X_prefix = np.zeros((50, N_FEATURES))  # every feature constant (std=0)
        scaler = PrefixFittedScaler().fit(X_prefix)
        transformed = scaler.transform(X_prefix)
        self.assertTrue(np.all(np.isfinite(transformed)))


if __name__ == "__main__":
    unittest.main()

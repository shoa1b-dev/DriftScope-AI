"""Unit tests for src/detectors/ — synthetic data only.

These verify the detector interface contract and basic sanity behavior
(fits without error, produces the right shape, flags obvious outliers
higher than obvious inliers). They are NOT a substitute for real
detection-performance evaluation on NAB, which requires real drift and
real anomaly labels — not available yet.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.detectors.detector_a_isolation_forest import DetectorAIsolationForest
from src.detectors.detector_b_autoencoder import DetectorBAutoencoder


def _make_synthetic_features(n=200, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, n_features))


class TestDetectorAIsolationForest(unittest.TestCase):
    def test_fit_and_score_shape(self):
        X = _make_synthetic_features()
        det = DetectorAIsolationForest(random_state=42).fit(X)
        scores = det.anomaly_score(X)
        self.assertEqual(scores.shape, (X.shape[0],))
        self.assertTrue(det.fitted)

    def test_score_before_fit_raises(self):
        det = DetectorAIsolationForest(random_state=42)
        with self.assertRaises(RuntimeError):
            det.anomaly_score(_make_synthetic_features())

    def test_obvious_outlier_scores_higher_than_inliers(self):
        rng = np.random.default_rng(1)
        inliers = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
        det = DetectorAIsolationForest(random_state=42).fit(inliers)

        far_outlier = np.full((1, 5), 100.0)
        inlier_sample = inliers[:1]

        outlier_score = det.anomaly_score(far_outlier)[0]
        inlier_score = det.anomaly_score(inlier_sample)[0]
        self.assertGreater(outlier_score, inlier_score)

    def test_different_seeds_give_different_random_state(self):
        det_a = DetectorAIsolationForest(random_state=13)
        det_b = DetectorAIsolationForest(random_state=42)
        self.assertNotEqual(det_a.random_state, det_b.random_state)


class TestDetectorBAutoencoder(unittest.TestCase):
    def test_fit_and_score_shape(self):
        X = _make_synthetic_features()
        det = DetectorBAutoencoder(random_state=42, max_iter=200).fit(X)
        scores = det.anomaly_score(X)
        self.assertEqual(scores.shape, (X.shape[0],))
        self.assertTrue(det.fitted)
        self.assertTrue(np.all(scores >= 0))  # MSE is non-negative

    def test_score_before_fit_raises(self):
        det = DetectorBAutoencoder(random_state=42)
        with self.assertRaises(RuntimeError):
            det.anomaly_score(_make_synthetic_features())

    def test_obvious_outlier_scores_higher_than_inliers(self):
        rng = np.random.default_rng(2)
        inliers = rng.normal(loc=0.0, scale=1.0, size=(300, 5))
        det = DetectorBAutoencoder(random_state=42, max_iter=500).fit(inliers)

        far_outlier = np.full((1, 5), 50.0)
        inlier_sample = inliers[:1]

        outlier_score = det.anomaly_score(far_outlier)[0]
        inlier_score = det.anomaly_score(inlier_sample)[0]
        self.assertGreater(outlier_score, inlier_score)

    def test_hidden_layer_sizes_default_is_documented_bottleneck(self):
        from src.detectors.detector_b_autoencoder import DEFAULT_HIDDEN_LAYER_SIZES
        self.assertEqual(DEFAULT_HIDDEN_LAYER_SIZES, (4, 2, 4))


class TestDetectorsShareInterface(unittest.TestCase):
    """RO4 fairness depends on both detectors being usable identically
    by the rest of the pipeline — this test guards that contract."""

    def test_both_detectors_expose_same_interface(self):
        X = _make_synthetic_features()
        detectors = [
            DetectorAIsolationForest(random_state=42),
            DetectorBAutoencoder(random_state=42, max_iter=200),
        ]
        for det in detectors:
            det.fit(X)
            scores = det.anomaly_score(X)
            self.assertEqual(scores.shape, (X.shape[0],))
            self.assertTrue(det.fitted)


if __name__ == "__main__":
    unittest.main()

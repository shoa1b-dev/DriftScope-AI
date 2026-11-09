"""Unit tests for src/explanation/fixed_sample.py — synthetic data only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explanation.fixed_sample import (
    FixedSampleError,
    build_fixed_sample,
    build_fixed_sample_from_full_series,
)


class TestBuildFixedSample(unittest.TestCase):
    def test_creates_sample_of_requested_size(self):
        features = np.arange(100 * 5, dtype=float).reshape(100, 5)
        sample = build_fixed_sample(features, n_samples=10, random_state=42)
        self.assertEqual(sample.n_samples, 10)
        self.assertEqual(len(sample.source_indices), 10)
        self.assertEqual(sample.features.shape, (10, 5))

    def test_same_random_state_gives_same_sample(self):
        features = np.arange(100 * 5, dtype=float).reshape(100, 5)
        sample_1 = build_fixed_sample(features, n_samples=10, random_state=42)
        sample_2 = build_fixed_sample(features, n_samples=10, random_state=42)
        self.assertEqual(sample_1.source_indices, sample_2.source_indices)
        np.testing.assert_array_equal(sample_1.features, sample_2.features)

    def test_different_random_state_gives_different_sample(self):
        features = np.arange(100 * 5, dtype=float).reshape(100, 5)
        sample_1 = build_fixed_sample(features, n_samples=10, random_state=42)
        sample_2 = build_fixed_sample(features, n_samples=10, random_state=99)
        self.assertNotEqual(sample_1.source_indices, sample_2.source_indices)

    def test_rejects_n_samples_exceeding_available_rows(self):
        features = np.arange(5 * 5, dtype=float).reshape(5, 5)
        with self.assertRaises(FixedSampleError):
            build_fixed_sample(features, n_samples=10, random_state=42)

    def test_rejects_zero_or_negative_n_samples(self):
        features = np.arange(50 * 5, dtype=float).reshape(50, 5)
        with self.assertRaises(FixedSampleError):
            build_fixed_sample(features, n_samples=0, random_state=42)


class TestSampleImmutability(unittest.TestCase):
    def test_features_array_is_read_only(self):
        features = np.arange(50 * 5, dtype=float).reshape(50, 5)
        sample = build_fixed_sample(features, n_samples=10, random_state=42)
        with self.assertRaises(ValueError):  # numpy raises ValueError on write to read-only array
            sample.features[0, 0] = 999.0

    def test_mutating_source_array_after_the_fact_does_not_affect_sample(self):
        features = np.arange(50 * 5, dtype=float).reshape(50, 5)
        sample = build_fixed_sample(features, n_samples=10, random_state=42)
        original_first_row = sample.features[0].copy()

        features[:] = -1.0  # mutate the ORIGINAL array after sampling

        np.testing.assert_array_equal(sample.features[0], original_first_row)

    def test_repeated_access_returns_identical_data(self):
        features = np.arange(50 * 5, dtype=float).reshape(50, 5)
        sample = build_fixed_sample(features, n_samples=10, random_state=42)
        snapshot_1 = sample.features.copy()
        snapshot_2 = sample.features.copy()
        np.testing.assert_array_equal(snapshot_1, snapshot_2)


class TestPreDriftOnlyEnforcement(unittest.TestCase):
    def test_full_series_wrapper_never_selects_post_drift_indices(self):
        # Rows 0..49 are "pre-drift", rows 50..99 are "post-drift"
        # (values deliberately distinct so leakage would be detectable
        # even without checking indices directly).
        pre_drift_block = np.zeros((50, 5))
        post_drift_block = np.full((50, 5), 999.0)
        all_features = np.vstack([pre_drift_block, post_drift_block])

        sample = build_fixed_sample_from_full_series(
            all_features, t_inject=50, n_samples=20, random_state=42
        )

        self.assertTrue(all(idx < 50 for idx in sample.source_indices))
        self.assertTrue(np.all(sample.features == 0.0))  # never picked up the 999.0 block

    def test_rejects_t_inject_at_or_below_zero(self):
        features = np.arange(50 * 5, dtype=float).reshape(50, 5)
        with self.assertRaises(FixedSampleError):
            build_fixed_sample_from_full_series(features, t_inject=0, n_samples=5, random_state=42)

    def test_rejects_t_inject_beyond_series_length(self):
        features = np.arange(50 * 5, dtype=float).reshape(50, 5)
        with self.assertRaises(FixedSampleError):
            build_fixed_sample_from_full_series(features, t_inject=200, n_samples=5, random_state=42)


class TestSignature(unittest.TestCase):
    def test_signature_is_deterministic(self):
        features = np.arange(50 * 5, dtype=float).reshape(50, 5)
        sample_1 = build_fixed_sample(features, n_samples=10, random_state=42)
        sample_2 = build_fixed_sample(features, n_samples=10, random_state=42)
        self.assertEqual(sample_1.signature, sample_2.signature)

    def test_signature_differs_for_different_samples(self):
        features = np.arange(50 * 5, dtype=float).reshape(50, 5)
        sample_1 = build_fixed_sample(features, n_samples=10, random_state=42)
        sample_2 = build_fixed_sample(features, n_samples=10, random_state=99)
        self.assertNotEqual(sample_1.signature, sample_2.signature)


if __name__ == "__main__":
    unittest.main()

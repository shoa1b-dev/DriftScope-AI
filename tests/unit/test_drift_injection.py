"""Unit tests for src/data/drift_injection.py — synthetic data only.

No NAB-specific magnitude or injection point appears anywhere here —
all numbers are arbitrary, chosen only to exercise the mechanism.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.drift_injection import DriftInjectionError, inject_offset_scale_drift


def _make_synthetic_series(n=100, value=10.0):
    timestamps = pd.date_range("2020-01-01", periods=n, freq="5min")
    return pd.Series(np.full(n, value), index=timestamps)


class TestOffsetDrift(unittest.TestCase):
    def test_offset_only_shifts_post_injection_values(self):
        series = _make_synthetic_series(n=50, value=10.0)
        modified, record = inject_offset_scale_drift(series, t_inject=20, offset=5.0)

        # Pre-injection unchanged
        self.assertTrue(np.allclose(modified.iloc[:20], 10.0))
        # Post-injection shifted by exactly the offset
        self.assertTrue(np.allclose(modified.iloc[20:], 15.0))
        self.assertEqual(record.t_inject, 20)
        self.assertEqual(record.offset, 5.0)
        self.assertEqual(record.n_points_affected, 30)


class TestScaleDrift(unittest.TestCase):
    def test_scale_only_multiplies_post_injection_values(self):
        series = _make_synthetic_series(n=50, value=10.0)
        modified, record = inject_offset_scale_drift(series, t_inject=10, scale_multiplier=2.0)

        self.assertTrue(np.allclose(modified.iloc[:10], 10.0))
        self.assertTrue(np.allclose(modified.iloc[10:], 20.0))
        self.assertEqual(record.scale_multiplier, 2.0)

    def test_combined_offset_and_scale(self):
        series = _make_synthetic_series(n=50, value=10.0)
        modified, record = inject_offset_scale_drift(
            series, t_inject=10, offset=3.0, scale_multiplier=2.0
        )
        # new_value = old * scale + offset = 10*2 + 3 = 23
        self.assertTrue(np.allclose(modified.iloc[10:], 23.0))


class TestReproducibility(unittest.TestCase):
    def test_identical_params_produce_identical_output(self):
        series = _make_synthetic_series(n=80, value=5.0)
        modified_1, record_1 = inject_offset_scale_drift(
            series, t_inject=30, offset=2.0, scale_multiplier=1.5, random_state=42
        )
        modified_2, record_2 = inject_offset_scale_drift(
            series, t_inject=30, offset=2.0, scale_multiplier=1.5, random_state=42
        )
        pd.testing.assert_series_equal(modified_1, modified_2)
        self.assertEqual(record_1, record_2)

    def test_record_captures_injected_label(self):
        series = _make_synthetic_series(n=50, value=1.0)
        _, record = inject_offset_scale_drift(series, t_inject=25, offset=1.0)
        self.assertEqual(record.injected_at_label, series.index[25])


class TestInvalidParameters(unittest.TestCase):
    def test_t_inject_zero_rejected(self):
        series = _make_synthetic_series(n=50)
        with self.assertRaises(DriftInjectionError):
            inject_offset_scale_drift(series, t_inject=0, offset=1.0)

    def test_t_inject_at_end_rejected(self):
        series = _make_synthetic_series(n=50)
        with self.assertRaises(DriftInjectionError):
            inject_offset_scale_drift(series, t_inject=50, offset=1.0)

    def test_t_inject_beyond_end_rejected(self):
        series = _make_synthetic_series(n=50)
        with self.assertRaises(DriftInjectionError):
            inject_offset_scale_drift(series, t_inject=100, offset=1.0)

    def test_negative_scale_multiplier_rejected(self):
        series = _make_synthetic_series(n=50)
        with self.assertRaises(DriftInjectionError):
            inject_offset_scale_drift(series, t_inject=10, scale_multiplier=-1.0)

    def test_zero_scale_multiplier_rejected(self):
        series = _make_synthetic_series(n=50)
        with self.assertRaises(DriftInjectionError):
            inject_offset_scale_drift(series, t_inject=10, scale_multiplier=0.0)


if __name__ == "__main__":
    unittest.main()

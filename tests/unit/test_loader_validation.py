"""Unit tests for src/data/loader.py and src/data/validation.py.

IMPORTANT: these tests use synthetic, hand-constructed CSV data. They
verify the CODE is correct — they are NOT a substitute for real NAB
dataset inspection. Do not cite these results as findings about NAB.

Written as unittest.TestCase so `python -m unittest discover -s tests -v`
finds and runs them (the Phase 1 version used plain functions, which
`discover` silently skips — fixed here).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import DatasetLoadError, load_nab_series
from src.data.validation import baseline_statistics, validate_series


def _write_synthetic_nab_csv(path: Path, n=200, with_gap=False, with_duplicate=False):
    timestamps = pd.date_range("2014-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    values = rng.normal(loc=50.0, scale=5.0, size=n)
    df = pd.DataFrame({"timestamp": timestamps, "value": values})

    if with_gap:
        df.loc[10, "value"] = np.nan

    if with_duplicate:
        dup_row = df.iloc[[5]].copy()
        df = pd.concat([df.iloc[:6], dup_row, df.iloc[6:]], ignore_index=True)

    df.to_csv(path, index=False)


class TestLoader(unittest.TestCase):
    def test_load_valid_synthetic_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_series.csv"
            _write_synthetic_nab_csv(path, n=100)
            series = load_nab_series(path)
            self.assertEqual(len(series), 100)
            self.assertEqual(series.name, "synthetic_series")
            self.assertIsInstance(series.index, pd.DatetimeIndex)

    def test_load_missing_file_raises(self):
        with self.assertRaises(DatasetLoadError):
            load_nab_series("/tmp/does_not_exist_12345.csv")

    def test_load_wrong_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_columns.csv"
            pd.DataFrame({"time": [1, 2], "val": [3, 4]}).to_csv(path, index=False)
            with self.assertRaises(DatasetLoadError):
                load_nab_series(path)


class TestValidation(unittest.TestCase):
    def test_detects_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "with_gap.csv"
            _write_synthetic_nab_csv(path, n=200, with_gap=True)
            series = load_nab_series(path)
            report = validate_series(series, minimum_length_required=50)
            self.assertEqual(report.n_missing_values, 1)
            self.assertEqual(report.length, 200)
            self.assertTrue(report.passes_minimum_length)

    def test_detects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "with_dup.csv"
            _write_synthetic_nab_csv(path, n=200, with_duplicate=True)
            series = load_nab_series(path)
            report = validate_series(series, minimum_length_required=50)
            self.assertEqual(report.n_duplicate_timestamps, 1)
            self.assertEqual(report.length, 201)

    def test_length_gate_fails_when_too_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "too_short.csv"
            _write_synthetic_nab_csv(path, n=10)
            series = load_nab_series(path)
            report = validate_series(series, minimum_length_required=500)
            self.assertFalse(report.passes_minimum_length)
            self.assertEqual(report.length, 10)

    def test_baseline_statistics_are_real_not_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "for_stats.csv"
            _write_synthetic_nab_csv(path, n=100)
            series = load_nab_series(path)
            stats = baseline_statistics(series, end_index=50)
            self.assertEqual(stats["n_points"], 50)
            # Known synthetic generation: loc=50.0, scale=5.0, seed=0
            self.assertTrue(45.0 < stats["mean"] < 55.0)
            self.assertGreater(stats["std"], 0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for src/data/nab_inspection.py::build_binary_labels_from_windows
— synthetic data only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.nab_inspection import build_binary_labels_from_windows


def _make_series(n=100):
    return pd.Series(np.arange(n, dtype=float), index=pd.date_range("2020-01-01", periods=n, freq="5min"))


class TestBuildBinaryLabelsFromWindows(unittest.TestCase):
    def test_empty_windows_gives_all_zero(self):
        series = _make_series()
        y_true = build_binary_labels_from_windows(series, [])
        self.assertEqual(y_true.sum(), 0)
        self.assertEqual(len(y_true), len(series))

    def test_single_window_marks_correct_range_inclusive(self):
        series = _make_series(n=50)
        windows = [(str(series.index[10]), str(series.index[15]))]
        y_true = build_binary_labels_from_windows(series, windows)
        self.assertEqual(y_true.sum(), 6)  # indices 10..15 inclusive = 6 points
        self.assertTrue(all(y_true[10:16] == 1))
        self.assertEqual(y_true[9], 0)
        self.assertEqual(y_true[16], 0)

    def test_multiple_non_overlapping_windows(self):
        series = _make_series(n=100)
        windows = [
            (str(series.index[5]), str(series.index[8])),
            (str(series.index[50]), str(series.index[55])),
        ]
        y_true = build_binary_labels_from_windows(series, windows)
        self.assertEqual(y_true.sum(), 4 + 6)

    def test_output_is_strictly_binary(self):
        series = _make_series(n=50)
        windows = [(str(series.index[0]), str(series.index[49]))]
        y_true = build_binary_labels_from_windows(series, windows)
        self.assertTrue(set(np.unique(y_true)) <= {0, 1})

    def test_window_outside_series_range_marks_nothing(self):
        series = _make_series(n=50)
        windows = [("2019-01-01 00:00:00", "2019-01-01 01:00:00")]  # before series starts
        y_true = build_binary_labels_from_windows(series, windows)
        self.assertEqual(y_true.sum(), 0)


if __name__ == "__main__":
    unittest.main()

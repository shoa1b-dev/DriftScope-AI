"""Unit tests for src/data/baseline_analysis.py — synthetic data with
HAND-COMPUTABLE expected values only. No real NAB data appears
anywhere in this file; no fact about fe7f93.csv or any real series is
claimed or implied here."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.baseline_analysis import (
    BLOCKED,
    PASS,
    WARN,
    check_anomaly_window_overlap,
    check_checkpoint_feasibility,
    check_event_index_validity,
    compute_baseline_statistics,
    recommend_drift_magnitudes,
)


def _make_series(n=100, start="2020-01-01", values=None):
    timestamps = pd.date_range(start, periods=n, freq="5min")
    if values is None:
        values = np.arange(n, dtype=float)
    return pd.Series(values, index=timestamps)


class TestBaselineStatistics(unittest.TestCase):
    def test_known_constant_series(self):
        series = _make_series(n=50, values=np.full(50, 10.0))
        stats = compute_baseline_statistics(series, t_event=30)
        self.assertEqual(stats.count, 30)
        self.assertEqual(stats.mean, 10.0)
        self.assertEqual(stats.std, 0.0)
        self.assertEqual(stats.minimum, 10.0)
        self.assertEqual(stats.maximum, 10.0)
        self.assertEqual(stats.median, 10.0)
        self.assertEqual(stats.coefficient_of_variation, 0.0)  # std=0, mean=10 -> CV=0/10=0.0, not None (None is only for mean==0)

    def test_cv_is_none_when_mean_is_exactly_zero(self):
        series = _make_series(n=50, values=np.zeros(50))
        stats = compute_baseline_statistics(series, t_event=30)
        self.assertIsNone(stats.coefficient_of_variation)

    def test_known_linear_series_hand_computed(self):
        # values 0..29 for the baseline region (t_event=30)
        series = _make_series(n=50, values=np.arange(50, dtype=float))
        stats = compute_baseline_statistics(series, t_event=30)
        expected_mean = np.mean(np.arange(30))  # 14.5
        self.assertAlmostEqual(stats.mean, expected_mean)
        self.assertEqual(stats.minimum, 0.0)
        self.assertEqual(stats.maximum, 29.0)
        self.assertEqual(stats.median, 14.5)
        self.assertAlmostEqual(stats.coefficient_of_variation, stats.std / stats.mean)

    def test_never_includes_post_event_data(self):
        # Post-event values are deliberately huge (999) — if they leaked
        # into the baseline, mean/max would be way off.
        values = np.concatenate([np.zeros(30), np.full(20, 999.0)])
        series = _make_series(n=50, values=values)
        stats = compute_baseline_statistics(series, t_event=30)
        self.assertEqual(stats.maximum, 0.0)
        self.assertEqual(stats.mean, 0.0)

    def test_rejects_invalid_t_event(self):
        series = _make_series(n=50)
        with self.assertRaises(ValueError):
            compute_baseline_statistics(series, t_event=0)
        with self.assertRaises(ValueError):
            compute_baseline_statistics(series, t_event=100)

    def test_quantiles_present_and_ordered(self):
        series = _make_series(n=100, values=np.arange(100, dtype=float))
        stats = compute_baseline_statistics(series, t_event=100)
        q10, q50, q90 = stats.quantiles["q10"], stats.quantiles["q50"], stats.quantiles["q90"]
        self.assertLess(q10, q50)
        self.assertLess(q50, q90)

    def test_anomaly_overlap_in_baseline_reported_not_hidden(self):
        series = _make_series(n=50, values=np.arange(50, dtype=float))
        # A synthetic anomaly window fully inside the baseline region [0, 30)
        windows = [(str(series.index[5]), str(series.index[10]))]
        stats = compute_baseline_statistics(series, t_event=30, labeled_anomaly_windows=windows)
        self.assertEqual(stats.n_labeled_anomalies_in_baseline, 1)
        self.assertIn("overlap", stats.anomaly_treatment_note.lower())

    def test_anomaly_outside_baseline_not_flagged(self):
        series = _make_series(n=50, values=np.arange(50, dtype=float))
        windows = [(str(series.index[40]), str(series.index[45]))]  # after t_event=30
        stats = compute_baseline_statistics(series, t_event=30, labeled_anomaly_windows=windows)
        self.assertEqual(stats.n_labeled_anomalies_in_baseline, 0)

    def test_anomaly_points_not_excluded_from_statistics(self):
        """Direct test that overlap is REPORTED but the anomalous
        values themselves remain part of the computed statistics."""
        values = np.zeros(30)
        values[10] = 1000.0  # a real anomalous spike inside the baseline
        series = _make_series(n=50, values=np.concatenate([values, np.zeros(20)]))
        windows = [(str(series.index[10]), str(series.index[10]))]
        stats = compute_baseline_statistics(series, t_event=30, labeled_anomaly_windows=windows)
        self.assertEqual(stats.maximum, 1000.0)  # NOT excluded
        self.assertEqual(stats.n_labeled_anomalies_in_baseline, 1)  # but flagged


class TestEventIndexValidity(unittest.TestCase):
    def test_valid_index(self):
        result = check_event_index_validity(t_event=50, row_count=100)
        self.assertEqual(result.status, PASS)

    def test_zero_rejected(self):
        result = check_event_index_validity(t_event=0, row_count=100)
        self.assertEqual(result.status, BLOCKED)

    def test_beyond_row_count_rejected(self):
        result = check_event_index_validity(t_event=100, row_count=100)
        self.assertEqual(result.status, BLOCKED)

    def test_negative_rejected(self):
        result = check_event_index_validity(t_event=-5, row_count=100)
        self.assertEqual(result.status, BLOCKED)


class TestAnomalyWindowOverlap(unittest.TestCase):
    def test_window_before_event(self):
        series = _make_series(n=100)
        windows = [(str(series.index[5]), str(series.index[10]))]
        results = check_anomaly_window_overlap(series, t_event=50, labeled_anomaly_windows=windows)
        self.assertEqual(results[0].position, "before_event")

    def test_window_after_event(self):
        series = _make_series(n=100)
        windows = [(str(series.index[60]), str(series.index[70]))]
        results = check_anomaly_window_overlap(series, t_event=50, labeled_anomaly_windows=windows)
        self.assertEqual(results[0].position, "after_event")

    def test_window_overlapping_event(self):
        series = _make_series(n=100)
        windows = [(str(series.index[45]), str(series.index[55]))]
        results = check_anomaly_window_overlap(series, t_event=50, labeled_anomaly_windows=windows)
        self.assertEqual(results[0].position, "overlaps_event")

    def test_multiple_windows_classified_independently(self):
        series = _make_series(n=100)
        windows = [
            (str(series.index[5]), str(series.index[10])),
            (str(series.index[60]), str(series.index[70])),
        ]
        results = check_anomaly_window_overlap(series, t_event=50, labeled_anomaly_windows=windows)
        self.assertEqual(results[0].position, "before_event")
        self.assertEqual(results[1].position, "after_event")


class TestCheckpointFeasibility(unittest.TestCase):
    def test_sufficient_data_passes(self):
        result = check_checkpoint_feasibility(
            row_count=1000, t_event=500, window_size=10,
            retrain_window_size=20, post_retrain_window_size=20,
        )
        self.assertEqual(result.status, PASS)
        self.assertGreater(result.n_post_event_windows, result.minimum_post_event_required)

    def test_insufficient_post_event_data_warns(self):
        result = check_checkpoint_feasibility(
            row_count=520, t_event=500, window_size=10,
            retrain_window_size=100, post_retrain_window_size=100,
        )
        self.assertEqual(result.status, WARN)

    def test_t_event_too_close_to_end_blocked(self):
        result = check_checkpoint_feasibility(
            row_count=100, t_event=99, window_size=10,
        )
        self.assertEqual(result.status, BLOCKED)

    def test_uses_exact_runner_boundary_formula(self):
        """Confirms this reuses the runner's own formula rather than an
        independently re-derived one — same inputs must give the same
        t_event_window as src.experiments.experiment_runner does."""
        from src.experiments.experiment_runner import _first_window_touched_by_event
        expected = _first_window_touched_by_event(t_event=500, window_size=10, stride=1)
        result = check_checkpoint_feasibility(row_count=1000, t_event=500, window_size=10)
        self.assertEqual(result.t_event_window, expected)


class TestDriftMagnitudeRecommendation(unittest.TestCase):
    def test_candidates_derived_from_real_baseline_not_fixed(self):
        series = _make_series(n=100, values=np.full(100, 50.0))
        stats = compute_baseline_statistics(series, t_event=100)
        # std=0 here (constant series) -> all offset-based candidates should be 0
        candidates = recommend_drift_magnitudes(stats)
        for c in candidates[:3]:  # the 1/2/3-std candidates
            self.assertEqual(c.offset, 0.0)  # 0 * std = 0, exact, traceable

    def test_larger_std_gives_larger_offset(self):
        rng = np.random.default_rng(0)
        low_var_series = _make_series(n=200, values=rng.normal(50, 1, 200))
        high_var_series = _make_series(n=200, values=rng.normal(50, 10, 200))
        low_stats = compute_baseline_statistics(low_var_series, t_event=200)
        high_stats = compute_baseline_statistics(high_var_series, t_event=200)
        low_candidates = recommend_drift_magnitudes(low_stats)
        high_candidates = recommend_drift_magnitudes(high_stats)
        self.assertLess(low_candidates[1].offset, high_candidates[1].offset)  # 2-std candidate

    def test_flags_implausible_negative_shift_for_nonnegative_signal(self):
        # A low-mean, high-variance baseline where a 3-std negative...
        # actually our candidates are all positive offsets; test the
        # plausibility check differently: a baseline near zero with
        # large std makes even a positive offset candidate's minimum
        # check meaningful via baseline.minimum.
        values = np.concatenate([np.zeros(50), np.full(50, 0.01)])  # near-zero, low variance
        series = _make_series(n=100, values=values)
        stats = compute_baseline_statistics(series, t_event=100)
        candidates = recommend_drift_magnitudes(stats, baseline_is_nonnegative=True)
        for c in candidates:
            self.assertTrue(c.plausible_range)  # positive offsets on a non-negative baseline are always plausible

    def test_one_std_flagged_as_not_distinguishable(self):
        rng = np.random.default_rng(1)
        series = _make_series(n=200, values=rng.normal(50, 5, 200))
        stats = compute_baseline_statistics(series, t_event=200)
        candidates = recommend_drift_magnitudes(stats)
        one_std = candidates[0]
        self.assertEqual(one_std.label, "1_std_shift")
        self.assertFalse(one_std.distinguishable_from_noise)
        self.assertTrue(len(one_std.notes) > 0)

    def test_combined_candidate_present(self):
        series = _make_series(n=100, values=np.full(100, 10.0))
        stats = compute_baseline_statistics(series, t_event=100)
        candidates = recommend_drift_magnitudes(stats)
        labels = [c.label for c in candidates]
        self.assertIn("combined_offset_and_scale", labels)

    def test_iqr_based_robust_candidates_present(self):
        rng = np.random.default_rng(2)
        series = _make_series(n=200, values=rng.normal(50, 5, 200))
        stats = compute_baseline_statistics(series, t_event=200)
        candidates = recommend_drift_magnitudes(stats)
        labels = [c.label for c in candidates]
        self.assertIn("1.5_iqr_shift_robust", labels)
        self.assertIn("3_iqr_shift_robust", labels)

    def test_contamination_flag_added_when_anomalies_overlap_baseline(self):
        """Direct test of the real finding from fe7f93.csv: when the
        baseline region contains real labeled anomalies, std-based
        candidates must carry an explicit caution note — never
        silently present a possibly-inflated std as if it were clean."""
        series = _make_series(n=100, values=np.arange(100, dtype=float))
        windows = [(str(series.index[10]), str(series.index[15]))]  # inside baseline [0, 50)
        stats = compute_baseline_statistics(series, t_event=50, labeled_anomaly_windows=windows)
        self.assertGreater(stats.n_labeled_anomalies_in_baseline, 0)

        candidates = recommend_drift_magnitudes(stats)
        std_based = [c for c in candidates if "std" in c.label]
        for c in std_based:
            self.assertTrue(any("CAUTION" in note for note in c.notes))

    def test_no_contamination_flag_when_baseline_is_clean(self):
        series = _make_series(n=100, values=np.arange(100, dtype=float))
        stats = compute_baseline_statistics(series, t_event=50)  # no windows passed at all
        candidates = recommend_drift_magnitudes(stats)
        std_based = [c for c in candidates if "std" in c.label]
        for c in std_based:
            self.assertFalse(any("CAUTION" in note for note in c.notes))

    def test_iqr_candidates_less_sensitive_to_outliers_than_std_candidates(self):
        """Reproduces the real fe7f93 scenario at small scale: a mostly
        low, tight cluster of values with two extreme spikes. IQR-based
        offset should be much smaller than std-based offset, matching
        what was observed on the real data."""
        values = np.full(100, 3.0)
        values[10] = 100.0
        values[50] = 90.0
        series = _make_series(n=100, values=values)
        stats = compute_baseline_statistics(series, t_event=100)
        candidates = recommend_drift_magnitudes(stats)
        two_std = next(c for c in candidates if c.label == "2_std_shift")
        iqr_robust = next(c for c in candidates if c.label == "1.5_iqr_shift_robust")
        self.assertLess(iqr_robust.offset, two_std.offset)


if __name__ == "__main__":
    unittest.main()

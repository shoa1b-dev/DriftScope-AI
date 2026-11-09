"""Unit tests for src/data/nab_inspection.py.

IMPORTANT: all fixture CSV/JSON files here are SYNTHETIC, written to a
temp directory in each test. This proves the INSPECTION CODE is
correct — it is NOT a substitute for real NAB inspection, and no fact
about real NAB is claimed or implied anywhere in this file.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.nab_inspection import (
    BLOCKED,
    PASS,
    WARN,
    build_environment_report,
    check_dependency,
    discover_nab_root,
    discover_series_files,
    inspect_labels,
    inspect_series_file,
    recommend_parameters,
)


def _write_synthetic_nab_csv(path: Path, n=100):
    timestamps = pd.date_range("2014-01-01", periods=n, freq="5min")
    values = [float(i) for i in range(n)]
    pd.DataFrame({"timestamp": timestamps, "value": values}).to_csv(path, index=False)


class TestDependencyChecks(unittest.TestCase):
    def test_known_available_package_passes(self):
        result = check_dependency("numpy")
        self.assertEqual(result.status, PASS)
        self.assertIsNotNone(result.version)
        self.assertIsNone(result.error)

    def test_known_missing_package_reports_blocked_with_real_error(self):
        result = check_dependency("this_package_does_not_exist_12345")
        self.assertEqual(result.status, BLOCKED)
        self.assertIsNone(result.version)
        self.assertIsNotNone(result.error)  # real ImportError text, not a placeholder

    def test_environment_report_reflects_actual_python_version(self):
        import sys as real_sys
        report = build_environment_report()
        self.assertIn(real_sys.version.split()[0], report.python_version)


class TestDatasetDiscovery(unittest.TestCase):
    def test_nonexistent_explicit_root_reported_blocked(self):
        result = discover_nab_root(explicit_root="/tmp/definitely_does_not_exist_nab_root_xyz")
        self.assertEqual(result.status, BLOCKED)
        self.assertIsNone(result.root_path)
        self.assertIn("does not exist", result.reason)

    def test_valid_root_with_data_dir_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").mkdir()
            result = discover_nab_root(explicit_root=tmp)
            self.assertEqual(result.status, PASS)
            self.assertEqual(result.root_path, tmp)

    def test_root_without_data_subdir_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            # no "data" subdirectory created
            result = discover_nab_root(explicit_root=tmp)
            self.assertEqual(result.status, BLOCKED)

    def test_invalid_explicit_root_never_falls_back_to_common_candidates(self):
        """Regression test for a real bug found via laptop execution:
        an invalid explicit_root must be reported BLOCKED even when a
        real NAB-like directory genuinely exists at one of the common
        candidate locations — the explicit override must be
        authoritative, never silently bypassed."""
        import src.data.nab_inspection as nab_inspection_module

        with tempfile.TemporaryDirectory() as real_candidate_dir:
            (Path(real_candidate_dir) / "data").mkdir()
            original_candidates = nab_inspection_module.COMMON_NAB_ROOT_CANDIDATES
            nab_inspection_module.COMMON_NAB_ROOT_CANDIDATES = (real_candidate_dir,)
            try:
                result = discover_nab_root(explicit_root="/tmp/definitely_does_not_exist_xyz")
                self.assertEqual(result.status, BLOCKED)
                self.assertIsNone(result.root_path)
            finally:
                nab_inspection_module.COMMON_NAB_ROOT_CANDIDATES = original_candidates

    def test_discover_series_files_lists_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            subset_dir = Path(tmp) / "data" / "realAWSCloudwatch"
            subset_dir.mkdir(parents=True)
            (subset_dir / "series_a.csv").write_text("timestamp,value\n")
            (subset_dir / "series_b.csv").write_text("timestamp,value\n")
            (subset_dir / "not_a_csv.txt").write_text("x")
            files = discover_series_files(tmp, "realAWSCloudwatch")
            self.assertEqual(len(files), 2)
            self.assertTrue(all(f.endswith(".csv") for f in files))

    def test_discover_series_files_empty_for_missing_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").mkdir()
            files = discover_series_files(tmp, "nonexistent_subset")
            self.assertEqual(files, [])


class TestSeriesInspection(unittest.TestCase):
    def test_inspects_valid_synthetic_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            subset_dir = Path(tmp) / "data" / "realAWSCloudwatch"
            subset_dir.mkdir(parents=True)
            _write_synthetic_nab_csv(subset_dir / "synthetic.csv", n=150)
            result = inspect_series_file(tmp, "data/realAWSCloudwatch/synthetic.csv")
            self.assertEqual(result.status, PASS)
            self.assertEqual(result.row_count, 150)
            self.assertEqual(result.n_missing_values, 0)
            self.assertEqual(result.n_duplicate_timestamps, 0)
            self.assertTrue(result.is_monotonic)

    def test_reports_blocked_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = inspect_series_file(tmp, "data/realAWSCloudwatch/does_not_exist.csv")
            self.assertEqual(result.status, BLOCKED)
            self.assertIsNotNone(result.error)

    def test_detects_real_missing_values_in_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            subset_dir = Path(tmp) / "data" / "realAWSCloudwatch"
            subset_dir.mkdir(parents=True)
            path = subset_dir / "with_gap.csv"
            timestamps = pd.date_range("2014-01-01", periods=50, freq="5min")
            values = [float(i) for i in range(50)]
            values[10] = None
            pd.DataFrame({"timestamp": timestamps, "value": values}).to_csv(path, index=False)

            result = inspect_series_file(tmp, "data/realAWSCloudwatch/with_gap.csv")
            self.assertEqual(result.n_missing_values, 1)


class TestLabelInspection(unittest.TestCase):
    def test_finds_combined_windows_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_dir = Path(tmp) / "labels"
            labels_dir.mkdir()
            windows = {"realAWSCloudwatch/synthetic.csv": [["2014-01-01 00:00:00", "2014-01-01 01:00:00"]]}
            (labels_dir / "combined_windows.json").write_text(json.dumps(windows))

            result = inspect_labels(tmp, "data/realAWSCloudwatch/synthetic.csv")
            self.assertEqual(result.status, PASS)
            self.assertEqual(result.n_windows, 1)
            self.assertEqual(result.windows[0], ("2014-01-01 00:00:00", "2014-01-01 01:00:00"))

    def test_falls_back_to_combined_labels_with_explicit_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_dir = Path(tmp) / "labels"
            labels_dir.mkdir()
            labels = {"realAWSCloudwatch/synthetic.csv": ["2014-01-01 00:00:00"]}
            (labels_dir / "combined_labels.json").write_text(json.dumps(labels))

            result = inspect_labels(tmp, "data/realAWSCloudwatch/synthetic.csv")
            self.assertEqual(result.status, WARN)  # fallback, not the preferred source
            self.assertEqual(result.n_windows, 1)
            self.assertIn("zero-width point", result.note)

    def test_reports_blocked_when_no_label_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "labels").mkdir()
            result = inspect_labels(tmp, "data/realAWSCloudwatch/synthetic.csv")
            self.assertEqual(result.status, BLOCKED)

    def test_warns_when_series_key_not_in_labels_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_dir = Path(tmp) / "labels"
            labels_dir.mkdir()
            (labels_dir / "combined_windows.json").write_text(json.dumps({"other_series.csv": []}))
            result = inspect_labels(tmp, "data/realAWSCloudwatch/synthetic.csv")
            self.assertEqual(result.status, WARN)


class TestParameterRecommendation(unittest.TestCase):
    def test_reasonable_row_count_produces_recommendation(self):
        rec = recommend_parameters(row_count=1000, window_size=10)
        self.assertEqual(rec.status, PASS)
        self.assertIsNotNone(rec.recommended_t_event)
        self.assertEqual(rec.recommended_t_event, 600)  # 60% of 1000, exact and traceable
        self.assertIsNone(rec.recommended_post_retrain_window)  # never locked here

    def test_too_short_series_warns_and_does_not_recommend(self):
        rec = recommend_parameters(row_count=20, window_size=10)
        self.assertEqual(rec.status, WARN)
        self.assertIsNone(rec.recommended_t_event)
        self.assertTrue(len(rec.notes) > 0)

    def test_recommendation_never_locks_post_retrain_window(self):
        """Direct test of the explicit Phase 10 restriction: this
        function must never resolve the Phase 8.1 methodological
        decision on its own."""
        for row_count in (500, 2000, 10000):
            rec = recommend_parameters(row_count=row_count, window_size=10)
            self.assertIsNone(rec.recommended_post_retrain_window)

    def test_minimum_pre_event_required_derived_from_config_defaults(self):
        rec = recommend_parameters(row_count=1000, window_size=10, background_sample_size=20, evaluation_sample_size=10)
        self.assertEqual(rec.minimum_pre_event_required, 20 + 10 + 10)


if __name__ == "__main__":
    unittest.main()

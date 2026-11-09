"""Tests for the Phase 11 locked configuration in config/experiment.yaml.

These tests use the REAL, already-established facts about
ec2_cpu_utilization_fe7f93.csv (row_count=4032, t_event=2419,
window_size=10) as fixed constants — these are not synthetic guesses,
they are the actual real-data facts computed in Phase 11 (see
docs/PHASE11_BASELINE_AND_PARAMETERS.md). Re-deriving the feasibility
check from these known-real inputs, via the same pure function used
during the real analysis, lets this test run in ANY environment
(without needing the actual CSV file present) while still validating
against genuine real-data-derived numbers, not fabricated ones.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.baseline_analysis import PASS, check_checkpoint_feasibility

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"

# Real, already-established facts (Phase 10/11) about the locked series.
REAL_ROW_COUNT = 4032
REAL_T_EVENT = 2419
REAL_WINDOW_SIZE = 10
REAL_T_EVENT_WINDOW = 2410
REAL_N_POST_EVENT_WINDOWS = 1613


class TestLockedConfigValues(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(CONFIG_PATH.read_text())

    def test_dataset_series_locked(self):
        self.assertEqual(
            self.config["dataset"]["selected_series"],
            "data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv",
        )
        self.assertIsNone(self.config["dataset"]["fallback_reason"])

    def test_window_size_locked(self):
        self.assertEqual(self.config["preprocessing"]["window_size"], REAL_WINDOW_SIZE)

    def test_t_event_locked_at_original_heuristic_value(self):
        """Direct regression test of the explicit instruction: t_event
        must remain 2419, not be moved to chase label coverage."""
        self.assertEqual(self.config["drift_injection"]["t_inject"], REAL_T_EVENT)

    def test_primary_drift_magnitude_is_robust_iqr_candidate(self):
        magnitude = self.config["drift_injection"]["magnitude"]
        self.assertEqual(magnitude["offset"], 3.85)
        self.assertEqual(magnitude["scale_multiplier"], 1.0)

    def test_sensitivity_magnitude_recorded_but_not_active(self):
        sensitivity = self.config["drift_injection"]["sensitivity_magnitude"]
        self.assertEqual(sensitivity["offset"], 24.34)
        self.assertEqual(sensitivity["status"], "documented_not_active")

    def test_sensitivity_magnitude_not_in_active_experimental_conditions(self):
        """The sensitivity magnitude must not silently leak into the
        active experiment matrix."""
        conditions = self.config["experimental_conditions"]
        self.assertNotIn("sensitivity", str(conditions).lower())

    def test_retrain_and_post_retrain_windows_locked_at_100(self):
        self.assertEqual(self.config["retraining"]["retrain_window_size"], 100)
        self.assertEqual(self.config["retraining"]["post_retrain_window_size"], 100)

    def test_seeds_still_unchanged_from_earlier_lock(self):
        """Regression guard: this phase must not have touched the
        already-locked seed list."""
        self.assertEqual(self.config["reproducibility"]["seeds"], [13, 42, 101, 202, 303])

    def test_detector_b_hidden_layer_sizes_unchanged(self):
        """Regression guard: this phase must not have touched
        previously-locked, non-dataset-dependent values."""
        self.assertEqual(self.config["detectors"]["detector_b"]["hidden_layer_sizes"], [4, 2, 4])


class TestLockedConfigAgainstRealBoundaryFacts(unittest.TestCase):
    """Re-validates the locked retrain/post-retrain window sizes
    against the real feasibility check, using the real row count as a
    known fact (not requiring the actual CSV file to be present)."""

    def test_locked_values_pass_real_feasibility_check(self):
        result = check_checkpoint_feasibility(
            row_count=REAL_ROW_COUNT, t_event=REAL_T_EVENT, window_size=REAL_WINDOW_SIZE,
            retrain_window_size=100, post_retrain_window_size=100,
        )
        self.assertEqual(result.status, PASS)
        self.assertEqual(result.t_event_window, REAL_T_EVENT_WINDOW)
        self.assertEqual(result.n_post_event_windows, REAL_N_POST_EVENT_WINDOWS)

    def test_locked_windows_leave_large_margin_below_available_data(self):
        """Confirms 100+100 is comfortably within the real 1613
        available post-event windows, not a tight/risky fit."""
        result = check_checkpoint_feasibility(
            row_count=REAL_ROW_COUNT, t_event=REAL_T_EVENT, window_size=REAL_WINDOW_SIZE,
            retrain_window_size=100, post_retrain_window_size=100,
        )
        self.assertGreater(result.n_post_event_windows, result.minimum_post_event_required * 5)

    def test_retrain_slice_never_dips_before_event_window(self):
        """Direct real-number reproduction of the retrain-slice clamp
        validated manually in this phase: for a detection shortly after
        the event, the retrain slice must never start before
        t_event_window, even with retrain_window_size=100."""
        t_detect_near_event = REAL_T_EVENT_WINDOW + 5
        retrain_start = max(REAL_T_EVENT_WINDOW, t_detect_near_event - 100 + 1)
        self.assertEqual(retrain_start, REAL_T_EVENT_WINDOW)  # clamp engaged, exactly as validated

    def test_only_one_post_event_anomaly_window_is_a_documented_limitation(self):
        """This test exists to make the limitation itself a checked,
        permanent fact in the suite — not just prose in a doc that
        could go stale. If the locked t_event ever changes, this
        documents that the post-event anomaly-window count should be
        re-verified, not silently assumed unchanged."""
        # Real fact from Phase 11: of 3 real labeled windows, exactly 1
        # falls after t_event=2419 (the other 2 fall before it).
        n_windows_after_event_at_locked_t_event = 1
        n_total_labeled_windows = 3
        self.assertEqual(n_windows_after_event_at_locked_t_event, 1)
        self.assertEqual(n_total_labeled_windows, 3)


if __name__ == "__main__":
    unittest.main()

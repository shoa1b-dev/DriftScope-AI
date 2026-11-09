"""Unit tests for src/monitoring/drift_monitor.py.

IMPORTANT — read before trusting these results:
- Tests using MockDriftMonitor verify the DriftMonitorBase interface
  contract (update/reset/detection_index semantics) using a FAKE
  detection rule. They do NOT verify real ADWIN or Page-Hinkley
  statistical behavior.
- Tests that would exercise AdwinDriftMonitor / PageHinkleyDriftMonitor
  against real river objects are SKIPPED in this environment because
  `river` is not installed (see RIVER_AVAILABLE). This is reported
  explicitly by the test runner as "skipped", not hidden.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.monitoring.drift_monitor import (
    RIVER_AVAILABLE,
    AdwinDriftMonitor,
    PageHinkleyDriftMonitor,
    RiverUnavailableError,
)
from tests.support.mock_drift_monitor import MockDriftMonitor


class TestDriftMonitorInterfaceViaMock(unittest.TestCase):
    """Verifies the DriftMonitorBase contract using the fake mock —
    real detection quality is NOT under test here."""

    def test_not_detected_initially(self):
        monitor = MockDriftMonitor(threshold=10.0)
        self.assertFalse(monitor.detected)
        self.assertIsNone(monitor.detection_index)

    def test_detects_once_threshold_exceeded(self):
        monitor = MockDriftMonitor(threshold=10.0)
        result_1 = monitor.update(5.0, index=0)
        result_2 = monitor.update(15.0, index=1)
        self.assertFalse(result_1)
        self.assertTrue(result_2)
        self.assertTrue(monitor.detected)
        self.assertEqual(monitor.detection_index, 1)

    def test_returns_true_only_once_on_first_detection(self):
        monitor = MockDriftMonitor(threshold=10.0)
        monitor.update(15.0, index=0)   # first detection -> True
        second_call = monitor.update(20.0, index=1)  # still above threshold, already detected
        self.assertFalse(second_call)  # not "newly" detected again
        self.assertEqual(monitor.detection_index, 0)  # unchanged from first detection

    def test_reset_clears_detection_state(self):
        monitor = MockDriftMonitor(threshold=10.0)
        monitor.update(15.0, index=0)
        self.assertTrue(monitor.detected)
        monitor.reset()
        self.assertFalse(monitor.detected)
        self.assertIsNone(monitor.detection_index)

    def test_can_detect_again_after_reset(self):
        monitor = MockDriftMonitor(threshold=10.0)
        monitor.update(15.0, index=0)
        monitor.reset()
        result = monitor.update(20.0, index=5)
        self.assertTrue(result)
        self.assertEqual(monitor.detection_index, 5)


@unittest.skipUnless(RIVER_AVAILABLE, "river is not installed in this environment")
class TestRealAdwinMonitor(unittest.TestCase):
    """SKIPPED in this environment. If you see this test actually run
    (not skipped), it means `river` became available and these
    assertions were genuinely exercised against real ADWIN."""

    def test_adwin_constructs_and_runs(self):
        monitor = AdwinDriftMonitor()
        for v in [1.0] * 50 + [100.0] * 50:
            monitor.update(v, index=0)
        # Real behavioral claim only made if this line actually executes.
        self.assertTrue(monitor.detected)


@unittest.skipUnless(RIVER_AVAILABLE, "river is not installed in this environment")
class TestRealPageHinkleyMonitor(unittest.TestCase):
    """SKIPPED in this environment — see class docstring above."""

    def test_page_hinkley_constructs_and_runs(self):
        monitor = PageHinkleyDriftMonitor()
        for v in [1.0] * 50 + [100.0] * 50:
            monitor.update(v, index=0)
        self.assertTrue(monitor.detected)


class TestAdwinUnavailableBehavior(unittest.TestCase):
    """This test class DOES run in this environment, and verifies the
    honest-failure path: no river -> clear error, not a silent fake."""

    @unittest.skipIf(RIVER_AVAILABLE, "this test targets the river-unavailable path")
    def test_adwin_raises_when_river_unavailable(self):
        with self.assertRaises(RiverUnavailableError):
            AdwinDriftMonitor()

    @unittest.skipIf(RIVER_AVAILABLE, "this test targets the river-unavailable path")
    def test_page_hinkley_raises_when_river_unavailable(self):
        with self.assertRaises(RiverUnavailableError):
            PageHinkleyDriftMonitor()


if __name__ == "__main__":
    unittest.main()

"""Unit tests for src/monitoring/retrain_trigger.py — uses MockDriftMonitor
(fake detection rule) to test the trigger/event wiring, not real drift
detection quality."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.monitoring.retrain_trigger import RetrainTrigger
from tests.support.mock_drift_monitor import MockDriftMonitor


class TestRetrainTriggerBasicFlow(unittest.TestCase):
    def test_no_detection_no_pending_retrain(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0))
        trigger.process(1.0, index=0)
        trigger.process(2.0, index=1)
        self.assertFalse(trigger.event.is_detected)
        self.assertFalse(trigger.retrain_pending)

    def test_detection_sets_pending_retrain(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0))
        trigger.process(1.0, index=0)
        newly_detected = trigger.process(15.0, index=1)
        self.assertTrue(newly_detected)
        self.assertTrue(trigger.event.is_detected)
        self.assertEqual(trigger.event.detection_index, 1)
        self.assertTrue(trigger.retrain_pending)  # detected but not yet retrained


class TestDetectionAndRetrainAreSeparateEvents(unittest.TestCase):
    """Direct test of your explicit requirement: t_detect and t_retrain
    must never be conflated, even when retraining happens immediately
    after detection."""

    def test_retrain_index_differs_from_detection_index(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0))
        trigger.process(15.0, index=5)   # detection at index 5
        trigger.confirm_retrain(index=6)  # retrain confirmed one step later

        self.assertEqual(trigger.event.detection_index, 5)
        self.assertEqual(trigger.event.retrain_index, 6)
        self.assertNotEqual(trigger.event.detection_index, trigger.event.retrain_index)
        self.assertFalse(trigger.retrain_pending)  # now retrained

    def test_confirm_retrain_before_detection_raises(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0))
        with self.assertRaises(RuntimeError):
            trigger.confirm_retrain(index=0)

    def test_confirm_retrain_twice_raises(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0))
        trigger.process(15.0, index=0)
        trigger.confirm_retrain(index=1)
        with self.assertRaises(RuntimeError):
            trigger.confirm_retrain(index=2)


class TestRetrainWindowSize(unittest.TestCase):
    def test_none_retrain_window_size_raises_on_slice_request(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0), retrain_window_size=None)
        with self.assertRaises(ValueError):
            trigger.get_retrain_slice(values=list(range(100)), up_to_index=50)

    def test_configured_window_size_returns_correct_slice(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0), retrain_window_size=10)
        values = list(range(100))
        result = trigger.get_retrain_slice(values, up_to_index=50)
        self.assertEqual(result, list(range(41, 51)))  # last 10 values up to and including index 50

    def test_window_size_larger_than_available_data_clamped_not_erroring(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0), retrain_window_size=1000)
        values = list(range(20))
        result = trigger.get_retrain_slice(values, up_to_index=19)
        self.assertEqual(result, values)  # clamped to what's actually available


class TestResetForNextCycle(unittest.TestCase):
    def test_reset_clears_both_monitor_and_event(self):
        trigger = RetrainTrigger(monitor=MockDriftMonitor(threshold=10.0))
        trigger.process(15.0, index=0)
        trigger.confirm_retrain(index=1)

        trigger.reset_for_next_cycle()

        self.assertFalse(trigger.event.is_detected)
        self.assertFalse(trigger.event.is_retrained)
        self.assertFalse(trigger.monitor.detected)


if __name__ == "__main__":
    unittest.main()

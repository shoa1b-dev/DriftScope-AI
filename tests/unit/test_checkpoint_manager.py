"""Unit tests for src/checkpoints/checkpoint_manager.py — synthetic data,
but using the REAL Detector A and Detector B classes (not mocks) to
prove compatibility with both, as required."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkpoints.checkpoint_manager import (
    CheckpointManager,
    CheckpointOrderError,
    DuplicateCheckpointError,
    IncompleteCheckpointsError,
    UnlabeledEventIndexError,
    VALID_CHECKPOINTS,
)
from src.detectors.detector_a_isolation_forest import DetectorAIsolationForest
from src.detectors.detector_b_autoencoder import DetectorBAutoencoder
from src.explanation.fixed_sample import build_fixed_sample


def _make_sample():
    features = np.arange(50 * 5, dtype=float).reshape(50, 5)
    return build_fixed_sample(features, n_samples=10, random_state=42)


def _make_features(n=100, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_features))


class TestCheckpointOrdering(unittest.TestCase):
    def test_correct_order_succeeds(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())

        manager.capture("pre_drift", det)
        manager.capture("post_drift_pre_retrain", det, event_index=10, extra_metadata={"index_source": "adwin_detection"})
        manager.capture("post_retrain", det, event_index=11, extra_metadata={"index_source": "retrain_trigger"})

        self.assertEqual(manager.captured_checkpoints, list(VALID_CHECKPOINTS))
        self.assertTrue(manager.is_complete)

    def test_out_of_order_capture_rejected(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())

        with self.assertRaises(CheckpointOrderError):
            manager.capture("post_drift_pre_retrain", det, event_index=10, extra_metadata={"index_source": "adwin_detection"})

    def test_skipping_middle_checkpoint_rejected(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        manager.capture("pre_drift", det)

        with self.assertRaises(CheckpointOrderError):
            manager.capture("post_retrain", det, event_index=20, extra_metadata={"index_source": "retrain_trigger"})


class TestDuplicateRejection(unittest.TestCase):
    def test_duplicate_checkpoint_rejected(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        manager.capture("pre_drift", det)

        with self.assertRaises(DuplicateCheckpointError):
            manager.capture("pre_drift", det)


class TestMissingCheckpointDetection(unittest.TestCase):
    def test_missing_checkpoints_reported(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        manager.capture("pre_drift", det)

        self.assertEqual(manager.missing_checkpoints, ["post_drift_pre_retrain", "post_retrain"])
        self.assertFalse(manager.is_complete)
        with self.assertRaises(IncompleteCheckpointsError):
            manager.require_complete()


class TestUnlabeledEventIndexRejection(unittest.TestCase):
    def test_event_index_without_source_label_rejected(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        manager.capture("pre_drift", det)

        with self.assertRaises(UnlabeledEventIndexError):
            manager.capture("post_drift_pre_retrain", det, event_index=10)  # no index_source

    def test_pre_drift_never_needs_index_source(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        record = manager.capture("pre_drift", det)  # no event_index at all — fine
        self.assertIsNone(record.event_index)


class TestModelSnapshotIndependence(unittest.TestCase):
    """Direct test of the core safety property: retraining the LIVE
    detector after a checkpoint must not change that checkpoint's
    stored snapshot behavior."""

    def test_detector_a_snapshot_survives_retraining(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features(seed=1))

        eval_points = _make_features(n=5, seed=99)
        record = manager.capture("pre_drift", det)
        scores_at_capture = record.model_snapshot.anomaly_score(eval_points)

        # Retrain the LIVE detector on very different data
        det.fit(_make_features(n=100, seed=1234) * 100 + 500)
        scores_after_live_retrain = det.anomaly_score(eval_points)
        scores_from_snapshot_again = record.model_snapshot.anomaly_score(eval_points)

        # Snapshot must be unaffected by the live retrain
        np.testing.assert_array_equal(scores_at_capture, scores_from_snapshot_again)
        # And the live detector must actually have changed (sanity check
        # that retraining really happened, not a no-op)
        self.assertFalse(np.allclose(scores_after_live_retrain, scores_at_capture))

    def test_detector_b_snapshot_survives_retraining(self):
        manager = CheckpointManager("detector_b", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorBAutoencoder(random_state=42, max_iter=200).fit(_make_features(seed=1))

        eval_points = _make_features(n=5, seed=99)
        record = manager.capture("pre_drift", det)
        scores_at_capture = record.model_snapshot.anomaly_score(eval_points)

        det.fit(_make_features(n=100, seed=1234) * 100 + 500)
        scores_from_snapshot_again = record.model_snapshot.anomaly_score(eval_points)

        np.testing.assert_array_equal(scores_at_capture, scores_from_snapshot_again)


class TestConditionHandling(unittest.TestCase):
    def test_rejects_invalid_condition(self):
        with self.assertRaises(ValueError):
            CheckpointManager("detector_a", "not_a_real_condition", seed=42, explanation_sample=_make_sample())

    def test_no_drift_condition_requires_index_source_same_as_drift(self):
        """No-drift control must NOT get a free pass on the
        don't-fabricate-events rule — it still must label where its
        matched-arbitrary-point index came from."""
        manager = CheckpointManager("detector_a", "no_drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        manager.capture("pre_drift", det)

        with self.assertRaises(UnlabeledEventIndexError):
            manager.capture("post_drift_pre_retrain", det, event_index=30)

        # Correct usage: explicit, honest label for the control condition
        record = manager.capture(
            "post_drift_pre_retrain", det, event_index=30,
            extra_metadata={"index_source": "matched_arbitrary_point"},
        )
        self.assertEqual(record.metadata["index_source"], "matched_arbitrary_point")


class TestDeterministicNaming(unittest.TestCase):
    def test_checkpoint_id_format(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        record = manager.capture("pre_drift", det)
        self.assertEqual(record.checkpoint_id, "detector_a_drift_42_pre_drift")


class TestSerialization(unittest.TestCase):
    def test_metadata_dict_is_json_serializable(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        record = manager.capture("pre_drift", det)

        payload = record.to_metadata_dict()
        json_text = json.dumps(payload)  # must not raise
        reloaded = json.loads(json_text)
        self.assertEqual(reloaded["checkpoint_id"], "detector_a_drift_42_pre_drift")
        self.assertEqual(reloaded["explanation_sample_signature"], manager.explanation_sample.signature)

    def test_round_trip_metadata_without_model_snapshot(self):
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=_make_sample())
        det = DetectorAIsolationForest(random_state=42).fit(_make_features())
        record = manager.capture("pre_drift", det)

        payload = record.to_metadata_dict()
        from src.checkpoints.checkpoint_manager import CheckpointRecord
        reconstructed = CheckpointRecord.from_metadata_dict(payload, model_snapshot=None)

        self.assertEqual(reconstructed.checkpoint_id, record.checkpoint_id)
        self.assertIsNone(reconstructed.model_snapshot)


if __name__ == "__main__":
    unittest.main()

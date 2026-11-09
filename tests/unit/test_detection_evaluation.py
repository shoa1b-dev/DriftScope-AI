"""Unit tests for src/metrics/detection_evaluation.py — synthetic
labels/predictions only. No real NAB ground truth or detector output
is used or claimed anywhere in these tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metrics.detection_evaluation import (
    EmptyInputError,
    Episode,
    InvalidEpisodeError,
    InvalidLabelError,
    InvalidThresholdComparisonError,
    LengthMismatchError,
    confusion_matrix_counts,
    evaluate_detection,
    evaluate_episode,
    evaluate_episodes,
    extract_episodes_from_binary,
    precision_recall_f1,
    threshold_scores,
)


class TestConfusionMatrixCounts(unittest.TestCase):
    def test_correct_counts(self):
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        # idx: 0 TP, 1 FN, 2 TN, 3 FP, 4 TP, 5 TN
        counts = confusion_matrix_counts(y_true, y_pred)
        self.assertEqual((counts.tp, counts.tn, counts.fp, counts.fn), (2, 2, 1, 1))

    def test_counts_are_python_ints(self):
        counts = confusion_matrix_counts([1, 0], [1, 0])
        self.assertIsInstance(counts.tp, int)
        self.assertIsInstance(counts.tn, int)

    def test_counts_json_serializable(self):
        counts = confusion_matrix_counts([1, 0, 1], [1, 1, 0])
        json.dumps(counts.to_dict())  # must not raise

    def test_length_mismatch_rejected(self):
        with self.assertRaises(LengthMismatchError):
            confusion_matrix_counts([1, 0, 1], [1, 0])

    def test_empty_input_rejected(self):
        with self.assertRaises(EmptyInputError):
            confusion_matrix_counts([], [])

    def test_non_binary_label_rejected(self):
        with self.assertRaises(InvalidLabelError):
            confusion_matrix_counts([1, 2, 0], [1, 0, 0])

    def test_non_binary_prediction_rejected(self):
        with self.assertRaises(InvalidLabelError):
            confusion_matrix_counts([1, 0, 0], [1, 0, 5])

    def test_boolean_labels_accepted(self):
        counts = confusion_matrix_counts([True, False, True], [True, False, False])
        self.assertEqual((counts.tp, counts.tn, counts.fp, counts.fn), (1, 1, 0, 1))


class TestPrecisionRecallF1(unittest.TestCase):
    def test_normal_case(self):
        counts = confusion_matrix_counts([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
        # tp=2, fp=1, fn=1
        precision, recall, f1 = precision_recall_f1(counts)
        self.assertAlmostEqual(precision.value, 2 / 3)
        self.assertAlmostEqual(recall.value, 2 / 3)
        self.assertAlmostEqual(f1.value, 2 / 3)
        self.assertFalse(precision.undefined)
        self.assertFalse(recall.undefined)
        self.assertFalse(f1.undefined)

    def test_precision_undefined_when_no_positive_predictions(self):
        counts = confusion_matrix_counts([1, 1, 0], [0, 0, 0])  # no predicted positives
        precision, recall, f1 = precision_recall_f1(counts)
        self.assertEqual(precision.value, 0.0)
        self.assertTrue(precision.undefined)
        self.assertFalse(math_isnan_or_inf(precision.value))

    def test_recall_undefined_when_no_actual_positives(self):
        """This is exactly the no-drift-control case: zero true
        anomalies, so recall is undefined, not a real 'miss'."""
        counts = confusion_matrix_counts([0, 0, 0], [0, 1, 0])
        precision, recall, f1 = precision_recall_f1(counts)
        self.assertEqual(recall.value, 0.0)
        self.assertTrue(recall.undefined)

    def test_f1_undefined_when_precision_and_recall_both_zero(self):
        counts = confusion_matrix_counts([1, 0], [0, 1])  # tp=0, fp=1, fn=1
        precision, recall, f1 = precision_recall_f1(counts)
        self.assertEqual(f1.value, 0.0)
        self.assertTrue(f1.undefined)

    def test_never_nan_or_inf(self):
        counts = confusion_matrix_counts([0, 0, 0], [0, 0, 0])  # everything TN
        precision, recall, f1 = precision_recall_f1(counts)
        for m in (precision, recall, f1):
            self.assertFalse(math_isnan_or_inf(m.value))


def math_isnan_or_inf(x):
    import math
    return math.isnan(x) or math.isinf(x)


class TestThresholding(unittest.TestCase):
    def test_ge_threshold_includes_boundary(self):
        scores = np.array([0.1, 0.5, 0.9])
        preds = threshold_scores(scores, threshold=0.5, comparison=">=")
        np.testing.assert_array_equal(preds, [0, 1, 1])

    def test_gt_threshold_excludes_boundary(self):
        scores = np.array([0.1, 0.5, 0.9])
        preds = threshold_scores(scores, threshold=0.5, comparison=">")
        np.testing.assert_array_equal(preds, [0, 0, 1])

    def test_invalid_comparison_rejected(self):
        with self.assertRaises(InvalidThresholdComparisonError):
            threshold_scores(np.array([0.1, 0.2]), threshold=0.1, comparison="<=")


class TestEpisodeExtraction(unittest.TestCase):
    def test_single_episode(self):
        episodes = extract_episodes_from_binary([0, 0, 1, 1, 1, 0, 0])
        self.assertEqual(episodes, (Episode(start=2, end=4),))

    def test_multiple_separated_episodes(self):
        episodes = extract_episodes_from_binary([1, 0, 0, 1, 1, 0, 1])
        self.assertEqual(episodes, (Episode(0, 0), Episode(3, 4), Episode(6, 6)))

    def test_anomaly_at_beginning(self):
        episodes = extract_episodes_from_binary([1, 1, 0, 0])
        self.assertEqual(episodes, (Episode(0, 1),))

    def test_anomaly_at_end(self):
        episodes = extract_episodes_from_binary([0, 0, 1, 1])
        self.assertEqual(episodes, (Episode(2, 3),))

    def test_no_anomaly_no_drift_control(self):
        episodes = extract_episodes_from_binary([0, 0, 0, 0, 0])
        self.assertEqual(episodes, ())  # empty, never fabricated

    def test_all_anomalous(self):
        episodes = extract_episodes_from_binary([1, 1, 1])
        self.assertEqual(episodes, (Episode(0, 2),))

    def test_episode_end_before_start_rejected(self):
        with self.assertRaises(InvalidEpisodeError):
            Episode(start=5, end=2)


class TestEpisodeEvaluation(unittest.TestCase):
    def test_zero_delay_detection(self):
        episode = Episode(start=3, end=6)
        y_pred = [0, 0, 0, 1, 0, 0, 0]  # flags at index 3 = episode start
        result = evaluate_episode(episode, y_pred)
        self.assertTrue(result.detected)
        self.assertEqual(result.delay, 0)
        self.assertTrue(result.zero_delay)
        self.assertEqual(result.detected_index, 3)

    def test_positive_delay(self):
        episode = Episode(start=3, end=6)
        y_pred = [0, 0, 0, 0, 0, 1, 0]  # flags at index 5, episode starts at 3 -> delay 2
        result = evaluate_episode(episode, y_pred)
        self.assertTrue(result.detected)
        self.assertEqual(result.delay, 2)
        self.assertFalse(result.zero_delay)

    def test_missed_episode(self):
        episode = Episode(start=3, end=6)
        y_pred = [0, 0, 0, 0, 0, 0, 0]  # nothing flagged
        result = evaluate_episode(episode, y_pred)
        self.assertFalse(result.detected)
        self.assertIsNone(result.delay)
        self.assertIsNone(result.detected_index)
        self.assertFalse(result.zero_delay)

    def test_prediction_before_episode_not_counted(self):
        episode = Episode(start=3, end=6)
        y_pred = [1, 1, 1, 0, 0, 0, 0]  # all flags are BEFORE the episode
        result = evaluate_episode(episode, y_pred)
        self.assertFalse(result.detected)  # correctly missed — pre-episode flags don't count

    def test_prediction_after_episode_not_counted(self):
        episode = Episode(start=3, end=6)
        y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1]  # flags only AFTER episode ends (index 7,8)
        result = evaluate_episode(episode, y_pred)
        self.assertFalse(result.detected)  # per the documented definition: not searched beyond episode end

    def test_adjacent_anomalies_form_one_episode(self):
        episodes = extract_episodes_from_binary([0, 1, 1, 1, 1, 0])
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0], Episode(1, 4))


class TestMultipleEpisodes(unittest.TestCase):
    def test_aggregate_across_multiple_episodes(self):
        episodes = (Episode(0, 2), Episode(5, 7))
        y_pred = [0, 1, 0, 0, 0, 1, 0, 0]  # episode1: detected at idx1 (delay1), episode2: detected at idx5 (delay0)
        result = evaluate_episodes(episodes, y_pred)
        self.assertEqual(result.n_episodes, 2)
        self.assertEqual(result.n_missed, 0)
        self.assertEqual(result.n_zero_delay, 1)  # only episode2
        self.assertAlmostEqual(result.mean_delay_over_detected, (1 + 0) / 2)

    def test_some_missed_some_detected(self):
        episodes = (Episode(0, 2), Episode(5, 7))
        y_pred = [0, 0, 0, 0, 0, 0, 0, 0]  # nothing detected
        result = evaluate_episodes(episodes, y_pred)
        self.assertEqual(result.n_missed, 2)
        self.assertIsNone(result.mean_delay_over_detected)  # nothing to average — not fabricated as 0

    def test_no_episodes_no_drift_control(self):
        result = evaluate_episodes((), [0, 0, 0, 0])
        self.assertEqual(result.n_episodes, 0)
        self.assertEqual(result.n_missed, 0)
        self.assertEqual(result.n_zero_delay, 0)
        self.assertIsNone(result.mean_delay_over_detected)


class TestEvaluateDetectionTopLevel(unittest.TestCase):
    def test_full_evaluation_drift_condition(self):
        y_true = [0, 1, 1, 0, 0, 1, 1, 1, 0]
        y_pred = [0, 1, 0, 0, 0, 0, 1, 0, 0]
        result = evaluate_detection(
            y_true, y_pred, detector_name="detector_a", condition="drift", seed=42, checkpoint="post_retrain"
        )
        self.assertEqual(result.detector_name, "detector_a")
        self.assertEqual(result.condition, "drift")
        self.assertEqual(result.n_samples, 9)
        self.assertEqual(result.delay.n_episodes, 2)  # episode (1,2) and episode (5,7)

    def test_no_drift_control_does_not_fabricate_episodes(self):
        y_true = [0, 0, 0, 0, 0]
        y_pred = [0, 0, 0, 0, 0]
        result = evaluate_detection(
            y_true, y_pred, detector_name="detector_a", condition="no_drift", seed=42, checkpoint="pre_drift"
        )
        self.assertEqual(result.delay.n_episodes, 0)
        self.assertTrue(result.recall.undefined)  # no actual positives to recall

    def test_threshold_recorded_in_metadata(self):
        result = evaluate_detection(
            [1, 0], [1, 0], detector_name="detector_b", condition="drift", seed=1,
            checkpoint="pre_drift", threshold=0.75,
        )
        self.assertEqual(result.threshold, 0.75)

    def test_metadata_json_serializable(self):
        result = evaluate_detection(
            [1, 0, 1, 1], [1, 0, 0, 1], detector_name="detector_a", condition="drift",
            seed=42, checkpoint="post_drift_pre_retrain",
        )
        payload = result.to_metadata_dict()
        json_text = json.dumps(payload)
        reloaded = json.loads(json_text)
        self.assertEqual(reloaded["detector_name"], "detector_a")
        self.assertEqual(reloaded["n_episodes"], 2)


class TestNoFabricatedValues(unittest.TestCase):
    def test_delay_values_are_exact_not_approximated(self):
        episode = Episode(start=10, end=20)
        y_pred = [0] * 10 + [0, 0, 0, 1] + [0] * 7  # flag at absolute index 13
        result = evaluate_episode(episode, y_pred)
        self.assertEqual(result.delay, 3)  # exact: 13 - 10

    def test_missed_episode_never_gets_a_fake_delay(self):
        episode = Episode(start=0, end=3)
        result = evaluate_episode(episode, [0, 0, 0, 0])
        self.assertIsNone(result.delay)  # never 0, never -1, never any number


if __name__ == "__main__":
    unittest.main()

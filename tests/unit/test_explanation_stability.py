"""Unit tests for src/metrics/explanation_stability.py — synthetic
attribution vectors only. No real SHAP output is used or claimed
anywhere in these tests."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explanation.explanation_result import ExplanationResult
from src.metrics.explanation_stability import (
    APPROVED_COMPARISONS,
    InconsistentExplanationsError,
    InvalidComparisonError,
    InvalidKError,
    NonFiniteValueError,
    ShapeMismatchError,
    ZeroVectorError,
    compute_stability_metrics,
    cosine_similarity_matrix,
    cosine_similarity_pair,
    jaccard_overlap_matrix,
    jaccard_overlap_pair,
    select_top_k_features,
)

FEATURE_NAMES = ("mean", "std", "min", "max", "slope")


def _make_explanation(checkpoint, attributions, seed=42, detector="detector_a", condition="drift", sig="sig123"):
    return ExplanationResult(
        detector_name=detector,
        condition=condition,
        seed=seed,
        checkpoint=checkpoint,
        feature_names=FEATURE_NAMES,
        attributions=np.asarray(attributions, dtype=float),
        sample_shape=(np.asarray(attributions).shape[0], len(FEATURE_NAMES)),
        explanation_sample_signature=sig,
        background_sample_signature=None,
        shap_backend="TreeExplainer",
        model_type="IsolationForest",
    )


class TestCosineSimilarityPair(unittest.TestCase):
    def test_identical_vectors_give_1(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(cosine_similarity_pair(v, v), 1.0, places=10)

    def test_orthogonal_vectors_give_0(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity_pair(a, b), 0.0, places=10)

    def test_opposite_vectors_give_minus_1(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        self.assertAlmostEqual(cosine_similarity_pair(a, b), -1.0, places=10)

    def test_zero_vector_raises(self):
        a = np.zeros(5)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ZeroVectorError):
            cosine_similarity_pair(a, b)

    def test_both_zero_vectors_raises(self):
        a = np.zeros(5)
        b = np.zeros(5)
        with self.assertRaises(ZeroVectorError):
            cosine_similarity_pair(a, b)

    def test_nan_raises(self):
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(NonFiniteValueError):
            cosine_similarity_pair(a, b)

    def test_infinity_raises(self):
        a = np.array([1.0, np.inf, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(NonFiniteValueError):
            cosine_similarity_pair(a, b)

    def test_shape_mismatch_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ShapeMismatchError):
            cosine_similarity_pair(a, b)


class TestCosineSimilarityMatrix(unittest.TestCase):
    def test_matrix_shape_validation(self):
        A = np.ones((5, 3))
        B = np.ones((5, 4))
        with self.assertRaises(ShapeMismatchError):
            cosine_similarity_matrix(A, B)

    def test_mean_per_sample_aggregation(self):
        A = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])  # sims: 1.0, 0.0, -1.0
        result = cosine_similarity_matrix(A, B)
        np.testing.assert_allclose(result.values, [1.0, 0.0, -1.0], atol=1e-10)
        self.assertAlmostEqual(result.mean, 0.0, places=10)
        self.assertEqual(result.n_valid, 3)
        self.assertEqual(result.n_invalid, 0)

    def test_raise_mode_stops_on_first_invalid_row(self):
        A = np.array([[1.0, 0.0], [0.0, 0.0]])  # second row is a zero vector
        B = np.array([[1.0, 0.0], [1.0, 0.0]])
        with self.assertRaises(ZeroVectorError):
            cosine_similarity_matrix(A, B, on_invalid="raise")

    def test_mark_nan_mode_continues_past_invalid_rows(self):
        A = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        B = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        result = cosine_similarity_matrix(A, B, on_invalid="mark_nan")
        self.assertEqual(result.n_valid, 2)
        self.assertEqual(result.n_invalid, 1)
        self.assertEqual(result.invalid_indices, (1,))
        self.assertTrue(np.isnan(result.values[1]))
        self.assertAlmostEqual(result.mean, 1.0, places=10)  # mean over valid rows only

    def test_invalid_on_invalid_value_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity_matrix(np.ones((2, 2)), np.ones((2, 2)), on_invalid="not_a_real_mode")


class TestTopKSelection(unittest.TestCase):
    def test_selects_correct_top_k(self):
        vec = np.array([0.1, 0.9, 0.3, 0.05, 0.5])  # abs order: idx1(.9),idx4(.5),idx2(.3),idx0(.1),idx3(.05)
        top2 = select_top_k_features(vec, k=2, feature_names=FEATURE_NAMES)
        self.assertEqual(top2, (FEATURE_NAMES[1], FEATURE_NAMES[4]))

    def test_uses_absolute_magnitude_not_signed_value(self):
        vec = np.array([-0.9, 0.1, 0.2, 0.05, 0.3])
        top1 = select_top_k_features(vec, k=1, feature_names=FEATURE_NAMES)
        self.assertEqual(top1, (FEATURE_NAMES[0],))  # -0.9 has the largest absolute magnitude

    def test_k_must_be_at_least_1(self):
        vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        with self.assertRaises(InvalidKError):
            select_top_k_features(vec, k=0, feature_names=FEATURE_NAMES)

    def test_k_cannot_exceed_n_features(self):
        vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        with self.assertRaises(InvalidKError):
            select_top_k_features(vec, k=6, feature_names=FEATURE_NAMES)

    def test_deterministic_tie_breaking_lowest_index_wins(self):
        vec = np.array([0.5, 0.5, 0.5, 0.1, 0.1])  # first 3 tied
        top2 = select_top_k_features(vec, k=2, feature_names=FEATURE_NAMES)
        self.assertEqual(top2, (FEATURE_NAMES[0], FEATURE_NAMES[1]))  # lowest indices among ties

    def test_all_zero_vector_is_deterministic_not_an_error(self):
        vec = np.zeros(5)
        top2_run1 = select_top_k_features(vec, k=2, feature_names=FEATURE_NAMES)
        top2_run2 = select_top_k_features(vec, k=2, feature_names=FEATURE_NAMES)
        self.assertEqual(top2_run1, top2_run2)
        self.assertEqual(top2_run1, (FEATURE_NAMES[0], FEATURE_NAMES[1]))  # first-by-index

    def test_feature_name_count_mismatch_raises(self):
        vec = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(ShapeMismatchError):
            select_top_k_features(vec, k=1, feature_names=FEATURE_NAMES)  # 5 names, 3 values


class TestJaccardPair(unittest.TestCase):
    def test_identical_vectors_give_jaccard_1(self):
        vec = np.array([0.1, 0.9, 0.3, 0.05, 0.5])
        self.assertAlmostEqual(jaccard_overlap_pair(vec, vec, k=2, feature_names=FEATURE_NAMES), 1.0)

    def test_disjoint_top_k_gives_jaccard_0(self):
        a = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # top2 = idx0, idx1
        b = np.array([0.0, 0.0, 1.0, 1.0, 0.0])  # top2 = idx2, idx3
        self.assertAlmostEqual(jaccard_overlap_pair(a, b, k=2, feature_names=FEATURE_NAMES), 0.0)

    def test_partial_overlap(self):
        a = np.array([1.0, 1.0, 0.5, 0.0, 0.0])  # top2 = idx0, idx1
        b = np.array([1.0, 0.0, 0.5, 0.0, 0.0])  # top2 = idx0, idx2
        # intersection={idx0}, union={idx0,idx1,idx2} -> 1/3
        self.assertAlmostEqual(jaccard_overlap_pair(a, b, k=2, feature_names=FEATURE_NAMES), 1 / 3)


class TestJaccardMatrix(unittest.TestCase):
    def test_matrix_mean_aggregation(self):
        A = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        B = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        names = ("f0", "f1", "f2")
        result = jaccard_overlap_matrix(A, B, k=1, feature_names=names)
        # row0: top1 both = f0 -> jaccard 1.0 ; row1: top1 a=f0, b=f2 -> jaccard 0.0
        np.testing.assert_allclose(result.values, [1.0, 0.0])
        self.assertAlmostEqual(result.mean, 0.5)

    def test_nonfinite_value_raises_in_raise_mode(self):
        A = np.array([[1.0, np.nan, 0.0]])
        B = np.array([[1.0, 1.0, 0.0]])
        names = ("f0", "f1", "f2")
        with self.assertRaises(NonFiniteValueError):
            jaccard_overlap_matrix(A, B, k=1, feature_names=names, on_invalid="raise")

    def test_all_zero_rows_do_not_raise(self):
        A = np.zeros((3, 5))
        B = np.zeros((3, 5))
        result = jaccard_overlap_matrix(A, B, k=2, feature_names=FEATURE_NAMES)
        self.assertEqual(result.n_invalid, 0)
        np.testing.assert_allclose(result.values, [1.0, 1.0, 1.0])  # same deterministic top-k on both sides


class TestApprovedComparisons(unittest.TestCase):
    def test_pre_drift_to_post_drift_pre_retrain_is_approved(self):
        a = _make_explanation("pre_drift", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        b = _make_explanation("post_drift_pre_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        result = compute_stability_metrics(a, b, k=2)
        self.assertEqual(result.checkpoint_pair, ("pre_drift", "post_drift_pre_retrain"))

    def test_pre_drift_to_post_retrain_is_approved(self):
        a = _make_explanation("pre_drift", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        b = _make_explanation("post_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        result = compute_stability_metrics(a, b, k=2)
        self.assertEqual(result.checkpoint_pair, ("pre_drift", "post_retrain"))

    def test_unapproved_pair_rejected(self):
        a = _make_explanation("post_drift_pre_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        b = _make_explanation("post_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        with self.assertRaises(InvalidComparisonError):
            compute_stability_metrics(a, b, k=2)

    def test_reversed_approved_pair_rejected(self):
        """post_drift_pre_retrain -> pre_drift is NOT the same approved
        comparison as pre_drift -> post_drift_pre_retrain; direction matters."""
        a = _make_explanation("post_drift_pre_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        b = _make_explanation("pre_drift", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        with self.assertRaises(InvalidComparisonError):
            compute_stability_metrics(a, b, k=2)

    def test_exactly_two_approved_comparisons_defined(self):
        self.assertEqual(
            APPROVED_COMPARISONS,
            (("pre_drift", "post_drift_pre_retrain"), ("pre_drift", "post_retrain")),
        )


class TestConsistencyValidation(unittest.TestCase):
    def test_different_detector_rejected(self):
        a = _make_explanation("pre_drift", [[1.0, 0.0, 0.0, 0.0, 0.0]], detector="detector_a")
        b = _make_explanation("post_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]], detector="detector_b")
        with self.assertRaises(InconsistentExplanationsError):
            compute_stability_metrics(a, b, k=2)

    def test_different_seed_rejected(self):
        a = _make_explanation("pre_drift", [[1.0, 0.0, 0.0, 0.0, 0.0]], seed=1)
        b = _make_explanation("post_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]], seed=2)
        with self.assertRaises(InconsistentExplanationsError):
            compute_stability_metrics(a, b, k=2)

    def test_different_sample_signature_rejected(self):
        a = _make_explanation("pre_drift", [[1.0, 0.0, 0.0, 0.0, 0.0]], sig="sig_A")
        b = _make_explanation("post_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]], sig="sig_B")
        with self.assertRaises(InconsistentExplanationsError):
            compute_stability_metrics(a, b, k=2)


class TestMetadataAndSerialization(unittest.TestCase):
    def test_metadata_correctness(self):
        a = _make_explanation("pre_drift", [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
        b = _make_explanation("post_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
        result = compute_stability_metrics(a, b, k=3)

        self.assertEqual(result.detector_name, "detector_a")
        self.assertEqual(result.condition, "drift")
        self.assertEqual(result.seed, 42)
        self.assertEqual(result.n_samples, 2)
        self.assertEqual(result.n_features, 5)
        self.assertEqual(result.k, 3)
        self.assertEqual(result.aggregation_method, "mean")

    def test_json_serializable(self):
        a = _make_explanation("pre_drift", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        b = _make_explanation("post_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        result = compute_stability_metrics(a, b, k=2)
        payload = result.to_metadata_dict()
        json_text = json.dumps(payload)
        reloaded = json.loads(json_text)
        self.assertEqual(reloaded["checkpoint_pair"], ["pre_drift", "post_retrain"])

    def test_nan_serializes_as_null_not_invalid_json(self):
        a = _make_explanation("pre_drift", [[0.0, 0.0, 0.0, 0.0, 0.0]])  # zero vector -> cosine invalid
        b = _make_explanation("post_retrain", [[1.0, 0.0, 0.0, 0.0, 0.0]])
        result = compute_stability_metrics(a, b, k=2, on_invalid_cosine="mark_nan")
        payload = result.to_metadata_dict()
        json_text = json.dumps(payload)  # must not raise even with a NaN internally
        reloaded = json.loads(json_text)
        self.assertIsNone(reloaded["cosine"]["values"][0])


class TestNoFabricatedValues(unittest.TestCase):
    def test_cosine_values_are_exact_computed_result_not_approximated(self):
        a = np.array([[3.0, 4.0]])  # norm 5
        b = np.array([[3.0, 4.0]])
        result = cosine_similarity_matrix(a, b)
        self.assertEqual(result.values[0], 1.0)  # exact, not "close to 1"

    def test_known_hand_computed_cosine_value(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        # cos(45 degrees) = sqrt(2)/2
        result = cosine_similarity_matrix(a, b)
        self.assertAlmostEqual(result.values[0], math.sqrt(2) / 2, places=10)


if __name__ == "__main__":
    unittest.main()

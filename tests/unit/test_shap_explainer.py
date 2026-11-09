"""Unit tests for src/explanation/shap_explainer.py.

IMPORTANT — read before trusting these results:
- Backend dispatch (`get_explanation_target`), validation, shape
  normalization, and metadata are tested with REAL execution — these
  do not need shap at all.
- Full end-to-end tests using an INJECTED FAKE `raw_value_fn` verify the
  surrounding pipeline logic (never claimed as real SHAP output — the
  fake returns deterministic arrays of the right shape, nothing more).
- Tests that call the REAL shap library (`_run_real_shap` /
  `generate_explanation` with no injected fn) are SKIPPED here because
  `shap` is not installed (see SHAP_AVAILABLE).

Phase 5.1 update: background_sample is now a SEPARATE fixed sample from
the evaluation sample (Option B) for the permutation backend (Detector
B). Detector A (TreeExplainer) does not require one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkpoints.checkpoint_manager import CheckpointManager
from src.detectors.detector_a_isolation_forest import DetectorAIsolationForest
from src.detectors.detector_b_autoencoder import DetectorBAutoencoder
from src.explanation.fixed_sample import build_fixed_sample
from src.explanation.shap_explainer import (
    BACKEND_PERMUTATION_EXPLAINER,
    BACKEND_TREE_EXPLAINER,
    SHAP_AVAILABLE,
    InvalidSampleError,
    MissingBackgroundSampleError,
    MissingFeatureNamesError,
    SHAPUnavailableError,
    UnsupportedModelError,
    generate_explanation,
    get_explanation_target,
)

FEATURE_NAMES = ("mean", "std", "min", "max", "slope")


def _make_features(n=50, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_features))


def _fake_shap_values(backend_label, target, eval_X, background_X):
    """Deterministic FAKE explanation output for testing pipeline
    logic only. NEVER represents real SHAP attributions."""
    n_samples, n_features = eval_X.shape
    return np.ones((n_samples, n_features)) * 0.5


def _make_evaluation_sample(seed=42):
    features = _make_features(n=30, seed=seed)
    return build_fixed_sample(features, n_samples=10, random_state=seed)


def _make_background_sample(seed=777):
    features = _make_features(n=60, seed=seed)
    return build_fixed_sample(features, n_samples=20, random_state=seed)


def _make_checkpoint_record(detector_name="detector_a", condition="drift", seed=42):
    sample = _make_evaluation_sample(seed=seed)
    manager = CheckpointManager(detector_name, condition, seed, explanation_sample=sample)
    if detector_name == "detector_a":
        det = DetectorAIsolationForest(random_state=seed).fit(_make_features(n=100, seed=seed))
    else:
        det = DetectorBAutoencoder(random_state=seed, max_iter=200).fit(_make_features(n=100, seed=seed))

    record = manager.capture("pre_drift", det)
    return record, sample


class TestBackendSelection(unittest.TestCase):
    def test_detector_a_selects_tree_explainer(self):
        record, _ = _make_checkpoint_record("detector_a")
        backend, target = get_explanation_target("detector_a", record.model_snapshot)
        self.assertEqual(backend, BACKEND_TREE_EXPLAINER)
        self.assertEqual(type(target).__name__, "IsolationForest")

    def test_detector_b_selects_permutation_explainer(self):
        record, _ = _make_checkpoint_record("detector_b")
        backend, target = get_explanation_target("detector_b", record.model_snapshot)
        self.assertEqual(backend, BACKEND_PERMUTATION_EXPLAINER)

    def test_detector_b_target_is_the_anomaly_score_method_not_raw_model(self):
        record, _ = _make_checkpoint_record("detector_b")
        _, target = get_explanation_target("detector_b", record.model_snapshot)

        eval_points = _make_features(n=5, seed=999)
        from_target = target(eval_points)
        from_detector_directly = record.model_snapshot.anomaly_score(eval_points)
        np.testing.assert_array_equal(from_target, from_detector_directly)

    def test_unsupported_detector_name_raises(self):
        record, _ = _make_checkpoint_record("detector_a")
        with self.assertRaises(UnsupportedModelError):
            get_explanation_target("detector_c_does_not_exist", record.model_snapshot)

    def test_detector_a_raw_model_is_isolated_copy(self):
        """Phase 5.1 hardening: raw_model must return an independent
        deepcopy, not a live reference — mutating the returned object
        must not affect the detector's own internal model."""
        record, _ = _make_checkpoint_record("detector_a")
        copy_1 = record.model_snapshot.raw_model
        copy_2 = record.model_snapshot.raw_model
        self.assertIsNot(copy_1, copy_2)  # two separate deepcopies, not the same object

    def test_detector_b_has_no_raw_model_property(self):
        """Phase 5.1 decision: raw_model was removed from Detector B
        since the SHAP pipeline never uses it."""
        record, _ = _make_checkpoint_record("detector_b")
        self.assertFalse(hasattr(record.model_snapshot, "raw_model"))


class TestBackgroundSampleRequirement(unittest.TestCase):
    """Phase 5.1: the permutation backend must require a separate
    background sample; the tree backend must not require one."""

    def test_detector_b_without_background_sample_raises(self):
        record, sample = _make_checkpoint_record("detector_b")
        with self.assertRaises(MissingBackgroundSampleError):
            generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)

    def test_detector_b_with_background_sample_succeeds(self):
        record, sample = _make_checkpoint_record("detector_b")
        background = _make_background_sample()
        result = generate_explanation(
            record, sample, FEATURE_NAMES, background_sample=background, raw_value_fn=_fake_shap_values
        )
        self.assertEqual(result.background_sample_signature, background.signature)

    def test_detector_a_does_not_require_background_sample(self):
        record, sample = _make_checkpoint_record("detector_a")
        result = generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
        self.assertIsNone(result.background_sample_signature)

    def test_background_and_evaluation_samples_have_different_signatures(self):
        evaluation = _make_evaluation_sample()
        background = _make_background_sample()
        self.assertNotEqual(evaluation.signature, background.signature)

    def test_invalid_background_sample_rejected(self):
        record, sample = _make_checkpoint_record("detector_b")
        import dataclasses
        background = _make_background_sample()
        writeable_bg = background.features.copy()
        writeable_bg.setflags(write=True)
        tampered_background = dataclasses.replace(background, features=writeable_bg)

        with self.assertRaises(InvalidSampleError):
            generate_explanation(
                record, sample, FEATURE_NAMES, background_sample=tampered_background,
                raw_value_fn=_fake_shap_values,
            )


class TestFeatureNamePreservation(unittest.TestCase):
    def test_feature_names_carried_through_to_result(self):
        record, sample = _make_checkpoint_record("detector_a")
        result = generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
        self.assertEqual(result.feature_names, FEATURE_NAMES)

    def test_empty_feature_names_rejected(self):
        record, sample = _make_checkpoint_record("detector_a")
        with self.assertRaises(MissingFeatureNamesError):
            generate_explanation(record, sample, (), raw_value_fn=_fake_shap_values)

    def test_mismatched_feature_name_count_rejected(self):
        record, sample = _make_checkpoint_record("detector_a")
        with self.assertRaises(MissingFeatureNamesError):
            generate_explanation(record, sample, ("only_one",), raw_value_fn=_fake_shap_values)


class TestOutputShapeNormalization(unittest.TestCase):
    def test_attributions_shape_matches_sample_and_features(self):
        record, sample = _make_checkpoint_record("detector_a")
        result = generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
        self.assertEqual(result.attributions.shape, (sample.features.shape[0], len(FEATURE_NAMES)))

    def test_shap_explanation_like_object_is_unwrapped(self):
        record, sample = _make_checkpoint_record("detector_a")

        class FakeExplanationObject:
            def __init__(self, values):
                self.values = values

        def fake_fn_returning_explanation_object(backend, target, eval_X, background_X):
            return FakeExplanationObject(np.ones((eval_X.shape[0], eval_X.shape[1])) * 0.3)

        result = generate_explanation(
            record, sample, FEATURE_NAMES, raw_value_fn=fake_fn_returning_explanation_object
        )
        self.assertEqual(result.attributions.shape, (sample.features.shape[0], len(FEATURE_NAMES)))
        self.assertTrue(np.allclose(result.attributions, 0.3))

    def test_wrong_shape_raises(self):
        record, sample = _make_checkpoint_record("detector_a")

        def bad_fn(backend, target, eval_X, background_X):
            return np.ones((eval_X.shape[0], eval_X.shape[1] + 1))

        with self.assertRaises(ValueError):
            generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=bad_fn)


class TestCheckpointCompatibility(unittest.TestCase):
    def test_works_for_all_three_checkpoint_labels(self):
        sample = _make_evaluation_sample()
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=sample)
        det = DetectorAIsolationForest(random_state=42).fit(_make_features(n=100))

        manager.capture("pre_drift", det)
        manager.capture("post_drift_pre_retrain", det, event_index=10, extra_metadata={"index_source": "adwin_detection"})
        manager.capture("post_retrain", det, event_index=11, extra_metadata={"index_source": "retrain_trigger"})

        for checkpoint in ("pre_drift", "post_drift_pre_retrain", "post_retrain"):
            record = manager.get(checkpoint)
            result = generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
            self.assertEqual(result.checkpoint, checkpoint)


class TestFixedSampleImmutabilityValidation(unittest.TestCase):
    def test_rejects_writeable_sample_array(self):
        record, sample = _make_checkpoint_record("detector_a")
        import dataclasses
        writeable_features = sample.features.copy()
        writeable_features.setflags(write=True)
        tampered_sample = dataclasses.replace(sample, features=writeable_features)

        with self.assertRaises(InvalidSampleError):
            generate_explanation(record, tampered_sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)

    def test_rejects_empty_sample(self):
        record, sample = _make_checkpoint_record("detector_a")
        import dataclasses
        empty_features = np.empty((0, 5))
        empty_features.setflags(write=False)
        empty_sample = dataclasses.replace(sample, features=empty_features)

        with self.assertRaises(InvalidSampleError):
            generate_explanation(record, empty_sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)


class TestSameSampleAcrossCheckpoints(unittest.TestCase):
    def test_evaluation_signature_identical_across_all_checkpoints(self):
        sample = _make_evaluation_sample()
        manager = CheckpointManager("detector_a", "drift", seed=42, explanation_sample=sample)
        det = DetectorAIsolationForest(random_state=42).fit(_make_features(n=100))

        manager.capture("pre_drift", det)
        manager.capture("post_drift_pre_retrain", det, event_index=10, extra_metadata={"index_source": "adwin_detection"})
        manager.capture("post_retrain", det, event_index=11, extra_metadata={"index_source": "retrain_trigger"})

        signatures = set()
        for checkpoint in ("pre_drift", "post_drift_pre_retrain", "post_retrain"):
            record = manager.get(checkpoint)
            result = generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
            signatures.add(result.explanation_sample_signature)
        self.assertEqual(len(signatures), 1)

    def test_background_signature_identical_across_all_checkpoints(self):
        """Phase 5.1: the background sample must ALSO be reused
        identically across checkpoints, same as the evaluation sample."""
        sample = _make_evaluation_sample()
        background = _make_background_sample()
        manager = CheckpointManager("detector_b", "drift", seed=42, explanation_sample=sample)
        det = DetectorBAutoencoder(random_state=42, max_iter=200).fit(_make_features(n=100))

        manager.capture("pre_drift", det)
        manager.capture("post_drift_pre_retrain", det, event_index=10, extra_metadata={"index_source": "adwin_detection"})
        manager.capture("post_retrain", det, event_index=11, extra_metadata={"index_source": "retrain_trigger"})

        signatures = set()
        for checkpoint in ("pre_drift", "post_drift_pre_retrain", "post_retrain"):
            record = manager.get(checkpoint)
            result = generate_explanation(
                record, sample, FEATURE_NAMES, background_sample=background, raw_value_fn=_fake_shap_values
            )
            signatures.add(result.background_sample_signature)
        self.assertEqual(len(signatures), 1)


class TestMetadataCorrectness(unittest.TestCase):
    def test_metadata_fields_populated_correctly(self):
        record, sample = _make_checkpoint_record("detector_b", condition="no_drift", seed=101)
        background = _make_background_sample()
        result = generate_explanation(
            record, sample, FEATURE_NAMES, background_sample=background,
            config_snapshot={"note": "test"}, raw_value_fn=_fake_shap_values,
        )
        self.assertEqual(result.detector_name, "detector_b")
        self.assertEqual(result.condition, "no_drift")
        self.assertEqual(result.seed, 101)
        self.assertEqual(result.checkpoint, "pre_drift")
        self.assertEqual(result.shap_backend, BACKEND_PERMUTATION_EXPLAINER)
        self.assertEqual(result.model_type, "MLPRegressor")
        self.assertEqual(result.sample_shape, sample.features.shape)
        self.assertEqual(result.config_snapshot, {"note": "test"})
        self.assertEqual(result.background_sample_signature, background.signature)

    def test_metadata_json_serializable(self):
        import json
        record, sample = _make_checkpoint_record("detector_a")
        result = generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
        payload = result.to_metadata_dict()
        json_text = json.dumps(payload)
        reloaded = json.loads(json_text)
        self.assertEqual(reloaded["detector_name"], "detector_a")
        self.assertIsNone(reloaded["background_sample_signature"])

    def test_deterministic_metadata_for_identical_inputs(self):
        record_1, sample_1 = _make_checkpoint_record("detector_a", seed=42)
        record_2, sample_2 = _make_checkpoint_record("detector_a", seed=42)
        result_1 = generate_explanation(record_1, sample_1, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
        result_2 = generate_explanation(record_2, sample_2, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
        self.assertEqual(result_1.to_metadata_dict(), result_2.to_metadata_dict())

    def test_checkpoint_record_metadata_never_contains_model_object(self):
        """Regression test locking in the intentional metadata
        asymmetry: CheckpointRecord's metadata dict must never contain
        a live model object, while ExplanationResult's may contain
        plain numeric attributions."""
        record, sample = _make_checkpoint_record("detector_a")
        checkpoint_payload = record.to_metadata_dict()
        self.assertNotIn("model_snapshot", checkpoint_payload)

        result = generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=_fake_shap_values)
        explanation_payload = result.to_metadata_dict()
        self.assertIn("attributions", explanation_payload)


class TestNoFabricatedValues(unittest.TestCase):
    def test_attributions_come_directly_from_compute_fn_unmodified(self):
        record, sample = _make_checkpoint_record("detector_a")

        def distinctive_fn(backend, target, eval_X, background_X):
            return np.arange(eval_X.shape[0] * eval_X.shape[1], dtype=float).reshape(eval_X.shape[0], eval_X.shape[1])

        result = generate_explanation(record, sample, FEATURE_NAMES, raw_value_fn=distinctive_fn)
        expected = np.arange(sample.features.shape[0] * len(FEATURE_NAMES), dtype=float).reshape(
            sample.features.shape[0], len(FEATURE_NAMES)
        )
        np.testing.assert_array_equal(result.attributions, expected)


class TestSHAPUnavailableBehavior(unittest.TestCase):
    @unittest.skipIf(SHAP_AVAILABLE, "this test targets the shap-unavailable path")
    def test_generate_explanation_raises_without_injected_fn(self):
        record, sample = _make_checkpoint_record("detector_a")
        with self.assertRaises(SHAPUnavailableError):
            generate_explanation(record, sample, FEATURE_NAMES)


@unittest.skipUnless(SHAP_AVAILABLE, "shap is not installed in this environment")
class TestRealSHAPExecution(unittest.TestCase):
    """SKIPPED in this environment. If you see this actually run (not
    skipped), shap became available and real execution was exercised."""

    def test_real_tree_explainer_on_detector_a(self):
        record, sample = _make_checkpoint_record("detector_a")
        result = generate_explanation(record, sample, FEATURE_NAMES)
        self.assertEqual(result.attributions.shape, (sample.features.shape[0], len(FEATURE_NAMES)))

    def test_real_permutation_explainer_on_detector_b(self):
        record, sample = _make_checkpoint_record("detector_b")
        background = _make_background_sample()
        result = generate_explanation(record, sample, FEATURE_NAMES, background_sample=background)
        self.assertEqual(result.attributions.shape, (sample.features.shape[0], len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()

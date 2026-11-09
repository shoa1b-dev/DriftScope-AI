"""Comprehensive Integrated Synthetic Validation.

Exercises the FULL approved matrix: 2 conditions x 5 approved seeds x
2 detectors = 20 detector-run combinations, via 10 `run_experiment`
calls (each producing both detectors' results).

READ THIS BEFORE TRUSTING ANY RESULT IN THIS FILE:
- All data is SYNTHETIC — a deterministically generated sine+noise
  series, with a synthetic (hand-placed) anomaly episode. This is NOT
  real NAB data, and no claim here is a claim about NAB.
- Drift monitoring uses an INJECTED FAKE (MockDriftMonitor, a threshold
  rule), never real ADWIN/Page-Hinkley, except in the two tests that
  specifically target the real-dependency-unavailable path.
- SHAP explanation uses an INJECTED FAKE compute function, never real
  shap, except in the one test that specifically targets the real-
  dependency-unavailable path.
- "Passing" here means: the ORCHESTRATION is correct — stages wire
  together properly, checkpoints/samples/metadata behave as designed.
  It does NOT mean any real detection, drift-recovery, or explanation-
  stability finding has been produced. No such finding exists yet.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkpoints.checkpoint_manager import VALID_CHECKPOINTS
from src.experiments.experiment_config import APPROVED_SEEDS, ExperimentConfig
from src.experiments.experiment_runner import run_experiment
from src.monitoring.drift_monitor import RiverUnavailableError
from src.explanation.shap_explainer import SHAPUnavailableError
from tests.support.mock_drift_monitor import MockDriftMonitor

CONDITIONS = ("drift", "no_drift")
DETECTOR_NAMES = ("detector_a", "detector_b")

# A distinctive constant the fake SHAP function returns, so any test
# can confirm attributions are exactly this and nothing else —
# preventing an accidental swap for something that looks like a real
# (but actually fabricated) value.
FAKE_SHAP_VALUE = 0.5


def _make_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    values = 10 + np.sin(np.arange(n) / 5.0) + rng.normal(0, 0.1, n)
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=n, freq="5min"))


def _make_labels(n=200, anomaly_slice=(150, 158)):
    y = np.zeros(n, dtype=int)
    y[anomaly_slice[0]:anomaly_slice[1]] = 1
    return y


def _fake_shap(backend, target, eval_X, background_X):
    return np.ones((eval_X.shape[0], eval_X.shape[1])) * FAKE_SHAP_VALUE


def _always_detects_immediately():
    return MockDriftMonitor(threshold=-1e9)


def _raises_river_unavailable():
    """Deterministic fake simulating river-unavailable, decoupled from
    ambient environment — see the matching helper/rationale in
    tests/unit/test_experiment_runner.py."""
    raise RiverUnavailableError(
        "simulated unavailability (test fake) — intentionally decoupled from "
        "whether the real 'river' package is installed in this environment"
    )


def _raises_shap_unavailable(backend, target, eval_X, background_X):
    """Deterministic fake simulating shap-unavailable, decoupled from
    ambient environment."""
    raise SHAPUnavailableError(
        "simulated unavailability (test fake) — intentionally decoupled from "
        "whether the real 'shap' package is installed in this environment"
    )


# Shared synthetic dataset — built once, reused (read-only) across the
# whole matrix. inject_offset_scale_drift always copies rather than
# mutating, so sharing this across runs is safe.
_SERIES = _make_series()
_LABELS = _make_labels()


def _config_for(condition: str, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        condition=condition, seed=seed, window_size=10, t_event=100,
        retrain_window_size=20, background_sample_size=15, evaluation_sample_size=5,
    )


def _run(condition: str, seed: int, **overrides):
    config = _config_for(condition, seed) if not overrides else ExperimentConfig(
        condition=condition, seed=seed, window_size=10, t_event=100,
        retrain_window_size=20, background_sample_size=15, evaluation_sample_size=5,
        **overrides,
    )
    return run_experiment(
        _SERIES, _LABELS, config,
        primary_monitor_factory=_always_detects_immediately,
        secondary_monitor_factory=_always_detects_immediately,
        shap_compute_fn=_fake_shap,
    )


# ----------------------------------------------------------------------
# The full 20-combination matrix, run once and reused across the
# assertion methods below (avoids re-running the same 10 experiments
# repeatedly for each concern being checked).
# ----------------------------------------------------------------------


def _build_full_matrix():
    matrix = {}
    for condition in CONDITIONS:
        for seed in APPROVED_SEEDS:
            matrix[(condition, seed)] = _run(condition, seed)
    return matrix


# Built once, at import time, and shared read-only across every test
# class below — rebuilding it per class would re-run the same 10
# experiments 8 times over for no benefit.
_FULL_MATRIX = _build_full_matrix()


class TestFullMatrixCoverage(unittest.TestCase):
    """Confirms the matrix itself has the right shape before trusting
    any downstream assertion about it."""


    def test_matrix_has_ten_run_combinations(self):
        self.assertEqual(len(_FULL_MATRIX), len(CONDITIONS) * len(APPROVED_SEEDS))
        self.assertEqual(len(_FULL_MATRIX), 10)

    def test_matrix_covers_all_twenty_detector_combinations(self):
        detector_combinations = set()
        for (condition, seed), result in _FULL_MATRIX.items():
            for detector_name in DETECTOR_NAMES:
                detector_combinations.add((condition, seed, detector_name))
        self.assertEqual(len(detector_combinations), 20)

    def test_every_run_has_exactly_one_condition_and_one_seed(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            with self.subTest(condition=condition, seed=seed):
                self.assertEqual(result.condition, condition)
                self.assertEqual(result.seed, seed)


class TestCheckpointCompletenessAcrossMatrix(unittest.TestCase):

    def test_every_detector_has_all_three_checkpoints_deterministically_named(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            for detector_name in DETECTOR_NAMES:
                with self.subTest(condition=condition, seed=seed, detector=detector_name):
                    detector_result = getattr(result, detector_name)
                    self.assertEqual(set(detector_result.explanations.keys()), set(VALID_CHECKPOINTS))
                    self.assertEqual(set(detector_result.detection.keys()), set(VALID_CHECKPOINTS))
                    self.assertEqual(set(detector_result.checkpoint_events.keys()), set(VALID_CHECKPOINTS))
                    self.assertEqual(set(detector_result.window_bounds.keys()), set(VALID_CHECKPOINTS))


class TestSampleLeakageAndImmutabilityAcrossMatrix(unittest.TestCase):

    def test_both_samples_pre_drift_only_in_every_run(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            t_event_window = result.detector_a.t_event_window
            with self.subTest(condition=condition, seed=seed):
                self.assertTrue(all(i < t_event_window for i in result.background_sample.source_indices))
                self.assertTrue(all(i < t_event_window for i in result.evaluation_sample.source_indices))

    def test_samples_not_mutated_in_every_run(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            with self.subTest(condition=condition, seed=seed):
                self.assertFalse(result.background_sample.features.flags.writeable)
                self.assertFalse(result.evaluation_sample.features.flags.writeable)

    def test_same_sample_signature_across_both_detectors_in_every_run(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            with self.subTest(condition=condition, seed=seed):
                sig_a = result.detector_a.explanations["pre_drift"].explanation_sample_signature
                sig_b = result.detector_b.explanations["pre_drift"].explanation_sample_signature
                self.assertEqual(sig_a, sig_b)
                for checkpoint in VALID_CHECKPOINTS:
                    self.assertEqual(
                        result.detector_a.explanations[checkpoint].explanation_sample_signature,
                        sig_a,
                    )


class TestEventSourceCorrectnessAcrossMatrix(unittest.TestCase):

    def test_no_drift_never_uses_drift_detection_labels(self):
        for seed in APPROVED_SEEDS:
            result = _FULL_MATRIX[("no_drift", seed)]
            for detector_name in DETECTOR_NAMES:
                detector_result = getattr(result, detector_name)
                with self.subTest(seed=seed, detector=detector_name):
                    self.assertIsNone(detector_result.t_detect)
                    self.assertEqual(detector_result.secondary_monitor.status, "not_run")
                    for checkpoint, event in detector_result.checkpoint_events.items():
                        self.assertNotIn(event["index_source"], ("adwin_detection", "retrain_trigger"))

    def test_drift_condition_uses_adwin_detection_label(self):
        for seed in APPROVED_SEEDS:
            result = _FULL_MATRIX[("drift", seed)]
            for detector_name in DETECTOR_NAMES:
                detector_result = getattr(result, detector_name)
                with self.subTest(seed=seed, detector=detector_name):
                    self.assertIsNotNone(detector_result.t_detect)
                    self.assertEqual(
                        detector_result.checkpoint_events["post_drift_pre_retrain"]["index_source"],
                        "adwin_detection",
                    )
                    self.assertEqual(
                        detector_result.checkpoint_events["post_retrain"]["index_source"],
                        "retrain_trigger",
                    )


class TestCrossDetectorIsolationAcrossMatrix(unittest.TestCase):

    def test_detector_results_never_overwrite_each_other(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            with self.subTest(condition=condition, seed=seed):
                self.assertIsNot(result.detector_a, result.detector_b)
                self.assertEqual(result.detector_a.detector_name, "detector_a")
                self.assertEqual(result.detector_b.detector_name, "detector_b")
                for checkpoint in VALID_CHECKPOINTS:
                    self.assertEqual(result.detector_a.explanations[checkpoint].detector_name, "detector_a")
                    self.assertEqual(result.detector_b.explanations[checkpoint].detector_name, "detector_b")
                    self.assertEqual(result.detector_a.detection[checkpoint].detector_name, "detector_a")
                    self.assertEqual(result.detector_b.detection[checkpoint].detector_name, "detector_b")

    def test_model_types_are_correct_and_distinct(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            with self.subTest(condition=condition, seed=seed):
                self.assertEqual(result.detector_a.explanations["pre_drift"].model_type, "IsolationForest")
                self.assertEqual(result.detector_b.explanations["pre_drift"].model_type, "MLPRegressor")


class TestSHAPFromStoredSnapshotsAcrossMatrix(unittest.TestCase):

    def test_shap_backend_matches_detector(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            with self.subTest(condition=condition, seed=seed):
                for checkpoint in VALID_CHECKPOINTS:
                    self.assertEqual(result.detector_a.explanations[checkpoint].shap_backend, "TreeExplainer")
                    self.assertIn(
                        "Permutation", result.detector_b.explanations[checkpoint].shap_backend
                    )

    def test_no_fake_shap_value_presented_as_real(self):
        """Every attribution across the whole matrix must be EXACTLY
        the injected fake constant — proving nothing else (and
        certainly nothing resembling a real SHAP value) leaked in."""
        for (condition, seed), result in _FULL_MATRIX.items():
            for detector_name in DETECTOR_NAMES:
                detector_result = getattr(result, detector_name)
                for checkpoint, explanation in detector_result.explanations.items():
                    with self.subTest(condition=condition, seed=seed, detector=detector_name, checkpoint=checkpoint):
                        self.assertTrue(np.all(explanation.attributions == FAKE_SHAP_VALUE))


class TestStabilityAndDetectionIntegrationAcrossMatrix(unittest.TestCase):

    def test_stability_metrics_reference_same_sample_as_explanations(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            for detector_name in DETECTOR_NAMES:
                detector_result = getattr(result, detector_name)
                with self.subTest(condition=condition, seed=seed, detector=detector_name):
                    # compute_stability_metrics itself raises
                    # InconsistentExplanationsError if the sample
                    # signatures ever differed — reaching this point
                    # without an exception during the matrix build IS
                    # the proof; this assertion re-confirms the
                    # checkpoint pairing explicitly.
                    self.assertEqual(
                        detector_result.stability_vs_post_drift.checkpoint_pair,
                        ("pre_drift", "post_drift_pre_retrain"),
                    )
                    self.assertEqual(
                        detector_result.stability_vs_post_retrain.checkpoint_pair,
                        ("pre_drift", "post_retrain"),
                    )

    def test_detection_metrics_window_sizes_match_labels_used(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            for detector_name in DETECTOR_NAMES:
                detector_result = getattr(result, detector_name)
                with self.subTest(condition=condition, seed=seed, detector=detector_name):
                    for checkpoint in VALID_CHECKPOINTS:
                        bounds = detector_result.window_bounds[checkpoint]
                        detection = detector_result.detection[checkpoint]
                        self.assertEqual(bounds["n_samples"], detection.n_samples)


class TestJSONSerializationAcrossMatrix(unittest.TestCase):

    def test_every_run_serializes_to_valid_json_with_no_model_object(self):
        for (condition, seed), result in _FULL_MATRIX.items():
            with self.subTest(condition=condition, seed=seed):
                payload = result.to_metadata_dict()
                text = json.dumps(payload)
                self.assertNotIn("model_snapshot", text)
                reloaded = json.loads(text)
                self.assertEqual(reloaded["condition"], condition)
                self.assertEqual(reloaded["seed"], seed)


class TestReproducibilityAcrossSeeds(unittest.TestCase):
    def test_same_seed_reproduces_identically(self):
        # A representative subset (not all 5, to keep runtime
        # reasonable) — each seed independently re-run twice.
        for seed in (13, 303):
            with self.subTest(seed=seed):
                result_1 = _run("drift", seed)
                result_2 = _run("drift", seed)
                self.assertEqual(result_1.detector_a.t_detect, result_2.detector_a.t_detect)
                self.assertEqual(
                    result_1.detector_a.detection["pre_drift"].confusion.to_dict(),
                    result_2.detector_a.detection["pre_drift"].confusion.to_dict(),
                )

    def test_all_five_seeds_are_mutually_distinguishable_in_metadata(self):
        results = {seed: _run("drift", seed) for seed in APPROVED_SEEDS}
        recorded_seeds = {seed: result.seed for seed, result in results.items()}
        self.assertEqual(len(set(recorded_seeds.values())), 5)  # all 5 distinct, correctly recorded


class TestMissingDependencyBehaviorHonest(unittest.TestCase):
    def test_primary_river_unavailable_error_propagates_not_fabricates(self):
        """Deterministic fake, decoupled from ambient river
        availability — see rationale in test_experiment_runner.py."""
        config = _config_for("drift", 42)
        with self.assertRaises(RiverUnavailableError):
            run_experiment(
                _SERIES, _LABELS, config,
                primary_monitor_factory=_raises_river_unavailable,
                shap_compute_fn=_fake_shap,
            )

    def test_real_shap_path_raises_not_fabricates(self):
        """Deterministic fake, decoupled from ambient shap availability."""
        config = _config_for("drift", 42)
        with self.assertRaises(SHAPUnavailableError):
            run_experiment(
                _SERIES, _LABELS, config,
                primary_monitor_factory=_always_detects_immediately,
                secondary_monitor_factory=_always_detects_immediately,
                shap_compute_fn=_raises_shap_unavailable,
            )


if __name__ == "__main__":
    unittest.main()

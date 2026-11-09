"""Unit tests for scripts/run_phase12_experiment.py.

These tests monkeypatch `run_experiment` inside the script module to
simulate a MIX of successful and failing runs deterministically — this
is the only way to test the orchestration logic (duplicate detection,
correct row counts, correct failure recording) without requiring real
river/shap, which are unavailable in this sandbox. No claim is made
that these tests exercise real dependency behavior — they test the
script's own bookkeeping logic only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import the script as a module (it lives outside src/, so load it directly)
spec = importlib.util.spec_from_file_location(
    "run_phase12_experiment", REPO_ROOT / "scripts" / "run_phase12_experiment.py"
)
phase12_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase12_script)

from src.experiments.experiment_runner import DriftNeverDetectedError, ExperimentResult, DetectorRunResult
from src.monitoring.drift_monitor import RiverUnavailableError
from src.checkpoints.checkpoint_manager import CheckpointManager
from src.detectors.detector_a_isolation_forest import DetectorAIsolationForest
from src.explanation.fixed_sample import build_fixed_sample
from src.metrics.explanation_stability import compute_stability_metrics
from src.explanation.shap_explainer import generate_explanation
from src.metrics.detection_evaluation import evaluate_detection


def _make_fake_experiment_result(condition, seed):
    """Builds a minimal but STRUCTURALLY REAL ExperimentResult using
    synthetic data — reuses the actual dataclasses/checkpoint manager/
    metrics modules, just with tiny synthetic inputs, so the script's
    flattening logic is tested against the real result shape."""
    n = 30
    features = np.random.default_rng(seed).normal(size=(n, 5))
    sample = build_fixed_sample(features, n_samples=5, random_state=seed)

    manager = CheckpointManager("detector_a", condition, seed, explanation_sample=sample)
    det = DetectorAIsolationForest(random_state=seed).fit(features)
    manager.capture("pre_drift", det)
    manager.capture("post_drift_pre_retrain", det, event_index=10, extra_metadata={"index_source": "adwin_detection" if condition == "drift" else "matched_arbitrary_point"})
    manager.capture("post_retrain", det, event_index=11, extra_metadata={"index_source": "retrain_trigger" if condition == "drift" else "matched_arbitrary_point"})

    def fake_shap(backend, target, eval_X, background_X):
        return np.ones((eval_X.shape[0], eval_X.shape[1])) * 0.5

    explanations = {}
    for cp in ("pre_drift", "post_drift_pre_retrain", "post_retrain"):
        explanations[cp] = generate_explanation(
            manager.get(cp), sample, ("mean", "std", "min", "max", "slope"), raw_value_fn=fake_shap
        )

    stability_vs_post_drift = compute_stability_metrics(explanations["pre_drift"], explanations["post_drift_pre_retrain"], k=2)
    stability_vs_post_retrain = compute_stability_metrics(explanations["pre_drift"], explanations["post_retrain"], k=2)

    y_true = np.zeros(5, dtype=int)
    y_pred = np.zeros(5, dtype=int)
    detection = {
        cp: evaluate_detection(y_true, y_pred, "detector_a", condition, seed, cp)
        for cp in ("pre_drift", "post_drift_pre_retrain", "post_retrain")
    }

    from src.experiments.experiment_runner import SecondaryMonitorResult
    detector_a_result = DetectorRunResult(
        detector_name="detector_a", explanations=explanations,
        stability_vs_post_drift=stability_vs_post_drift, stability_vs_post_retrain=stability_vs_post_retrain,
        detection=detection, checkpoint_events={}, window_bounds={
            "pre_drift": {"start": 0, "end": 10, "n_samples": 10},
            "post_drift_pre_retrain": {"start": 10, "end": 11, "n_samples": 1},
            "post_retrain": {"start": 11, "end": 30, "n_samples": 19},
        },
        t_event_window=10, t_detect=10 if condition == "drift" else None, t_retrain=11,
        secondary_monitor=SecondaryMonitorResult(status="ran" if condition == "drift" else "not_run", detection_index=None),
    )
    detector_b_result = detector_a_result  # reuse structurally for detector_b in this fake

    return ExperimentResult(
        condition=condition, seed=seed, config=mock.Mock(to_metadata_dict=lambda: {}),
        detector_a=detector_a_result, detector_b=detector_b_result,
        background_sample=sample, evaluation_sample=sample,
    )


class TestRunAllOrchestration(unittest.TestCase):
    def _make_real_nab_dir(self, tmp):
        subset_dir = Path(tmp) / "data" / "realAWSCloudwatch"
        subset_dir.mkdir(parents=True)
        labels_dir = Path(tmp) / "labels"
        labels_dir.mkdir()
        timestamps = pd.date_range("2020-01-01", periods=50, freq="5min")
        pd.DataFrame({"timestamp": timestamps, "value": np.arange(50, dtype=float)}).to_csv(
            subset_dir / "synthetic.csv", index=False
        )
        (labels_dir / "combined_windows.json").write_text(json.dumps({}))
        return tmp

    def _make_cfg_file(self, tmp):
        cfg = {
            "dataset": {"selected_series": "data/realAWSCloudwatch/synthetic.csv"},
            "preprocessing": {"window_size": 5},
            "drift_injection": {"t_inject": 20, "magnitude": {"offset": 1.0, "scale_multiplier": 1.0}},
            "retraining": {"retrain_window_size": 5, "post_retrain_window_size": 5},
        }
        config_dir = Path(tmp) / "config"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "experiment.yaml").write_text(yaml.dump(cfg))
        return config_dir / "experiment.yaml"

    def test_all_success_gives_20_rows_all_completed(self):
        with mock.patch.object(phase12_script, "run_experiment", side_effect=lambda series, y_true, config: _make_fake_experiment_result(config.condition, config.seed)):
            with tempfile.TemporaryDirectory() as tmp:
                nab_dir = self._make_real_nab_dir(tmp)
                cfg_path = self._make_cfg_file(tmp)
                with mock.patch.object(phase12_script, "REPO_ROOT", Path(tmp)):
                    report = phase12_script.run_all(nab_dir)

        self.assertEqual(report["overall_status"], "COMPLETED")
        self.assertEqual(report["n_rows"], 20)
        self.assertEqual(report["n_completed"], 20)
        self.assertEqual(report["n_failed"], 0)

    def test_mixed_success_and_known_failure(self):
        call_count = {"n": 0}

        def fake_run_experiment(series, y_true, config):
            call_count["n"] += 1
            if config.condition == "drift" and config.seed == 13:
                raise RiverUnavailableError("simulated")
            return _make_fake_experiment_result(config.condition, config.seed)

        with mock.patch.object(phase12_script, "run_experiment", side_effect=fake_run_experiment):
            with tempfile.TemporaryDirectory() as tmp:
                nab_dir = self._make_real_nab_dir(tmp)
                self._make_cfg_file(tmp)
                with mock.patch.object(phase12_script, "REPO_ROOT", Path(tmp)):
                    report = phase12_script.run_all(nab_dir)

        self.assertEqual(report["n_rows"], 20)  # still exactly 20 — 2 rows for the failed run, not 0
        self.assertEqual(report["n_failed"], 2)  # detector_a + detector_b for the one failed (drift, seed=13) call
        self.assertEqual(report["n_completed"], 18)

        failed_rows = [r for r in report["rows"] if r["status"] == "failed"]
        self.assertEqual(len(failed_rows), 2)
        for row in failed_rows:
            self.assertEqual(row["condition"], "drift")
            self.assertEqual(row["seed"], 13)
            self.assertEqual(row["error_type"], "RiverUnavailableError")
            self.assertIsNone(row["pre_drift_precision"])  # never fabricated

    def test_no_duplicate_keys_across_20_rows(self):
        with mock.patch.object(phase12_script, "run_experiment", side_effect=lambda series, y_true, config: _make_fake_experiment_result(config.condition, config.seed)):
            with tempfile.TemporaryDirectory() as tmp:
                nab_dir = self._make_real_nab_dir(tmp)
                self._make_cfg_file(tmp)
                with mock.patch.object(phase12_script, "REPO_ROOT", Path(tmp)):
                    report = phase12_script.run_all(nab_dir)

        keys = [(r["condition"], r["seed"], r["detector"]) for r in report["rows"]]
        self.assertEqual(len(keys), len(set(keys)))  # no duplicates

    def test_blocked_when_nab_root_invalid(self):
        report = phase12_script.run_all("/tmp/definitely_does_not_exist_xyz_phase12")
        self.assertEqual(report["overall_status"], "BLOCKED")


class TestFlattenDetectorResult(unittest.TestCase):
    def test_failed_run_has_all_none_metrics(self):
        row = phase12_script.flatten_detector_result(
            "drift", 42, "detector_a", None, "failed",
            error_type="RiverUnavailableError", error_message="simulated",
        )
        self.assertIsNone(row["pre_drift_precision"])
        self.assertIsNone(row["t_detect"])
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error_type"], "RiverUnavailableError")

    def test_completed_run_has_real_values(self):
        result = _make_fake_experiment_result("drift", 42)
        row = phase12_script.flatten_detector_result("drift", 42, "detector_a", result.detector_a, "completed")
        self.assertEqual(row["status"], "completed")
        self.assertIsNotNone(row["pre_drift_precision"])
        self.assertIsNone(row["error_type"])


class TestOutputWriting(unittest.TestCase):
    def test_json_and_csv_written_with_no_model_object(self):
        with mock.patch.object(phase12_script, "run_experiment", side_effect=lambda series, y_true, config: _make_fake_experiment_result(config.condition, config.seed)):
            with tempfile.TemporaryDirectory() as tmp:
                nab_dir = self._make_real_nab_dir_static(tmp)
                self._make_cfg_file_static(tmp)
                with mock.patch.object(phase12_script, "REPO_ROOT", Path(tmp)):
                    report = phase12_script.run_all(nab_dir)
                    json_path, csv_path = phase12_script.write_outputs(report)

                self.assertTrue(json_path.exists())
                self.assertTrue(csv_path.exists())
                json_text = json_path.read_text()
                self.assertNotIn("model_snapshot", json_text)

                import csv as csv_module
                with open(csv_path) as f:
                    rows = list(csv_module.DictReader(f))
                self.assertEqual(len(rows), 20)

    def _make_real_nab_dir_static(self, tmp):
        return TestRunAllOrchestration()._make_real_nab_dir(tmp)

    def _make_cfg_file_static(self, tmp):
        return TestRunAllOrchestration()._make_cfg_file(tmp)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for src/experiments/result_writer.py — writes to a temp
directory only; nothing is written into the repository itself."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import run_experiment
from src.experiments.result_writer import (
    read_experiment_result_file,
    result_file_name,
    write_experiment_result,
)
from tests.support.mock_drift_monitor import MockDriftMonitor


def _make_result():
    n = 200
    rng = np.random.default_rng(0)
    values = 10 + np.sin(np.arange(n) / 5.0) + rng.normal(0, 0.1, n)
    series = pd.Series(values, index=pd.date_range("2020-01-01", periods=n, freq="5min"))
    y_true = np.zeros(n, dtype=int)
    y_true[120:125] = 1
    config = ExperimentConfig(
        condition="drift", seed=42, window_size=10, t_event=100,
        retrain_window_size=20, background_sample_size=15, evaluation_sample_size=5,
    )

    def fake_shap(backend, target, eval_X, background_X):
        return np.ones((eval_X.shape[0], eval_X.shape[1])) * 0.5

    return run_experiment(
        series, y_true, config,
        primary_monitor_factory=lambda: MockDriftMonitor(threshold=-1e9),
        secondary_monitor_factory=lambda: MockDriftMonitor(threshold=-1e9),
        shap_compute_fn=fake_shap,
    )


class TestResultFileName(unittest.TestCase):
    def test_deterministic_naming_format(self):
        self.assertEqual(result_file_name("detector_a", "drift", 42), "detector_a_drift_42.json")


class TestWriteAndReadRoundTrip(unittest.TestCase):
    def test_writes_one_file_per_detector(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            written = write_experiment_result(result, tmp)
            self.assertEqual(set(written.keys()), {"detector_a", "detector_b"})
            for path in written.values():
                self.assertTrue(Path(path).exists())

    def test_round_trip_preserves_key_fields(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            written = write_experiment_result(result, tmp)
            payload = read_experiment_result_file(written["detector_a"])
            self.assertEqual(payload["condition"], "drift")
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["detector_result"]["detector_name"], "detector_a")

    def test_no_model_object_in_written_file(self):
        """Regression check consistent with the Phase 4/5 metadata rule:
        a live model object must never appear in a JSON payload."""
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmp:
            written = write_experiment_result(result, tmp)
            text = Path(written["detector_a"]).read_text()
            self.assertNotIn("model_snapshot", text)


if __name__ == "__main__":
    unittest.main()

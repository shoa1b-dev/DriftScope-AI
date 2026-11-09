"""Unit tests for src/experiments/experiment_config.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.experiment_config import (
    APPROVED_SEEDS,
    ExperimentConfig,
    InvalidConfigurationError,
    InvalidSeedError,
)


def _base_kwargs(**overrides):
    kwargs = dict(condition="drift", seed=42, window_size=10, t_event=50)
    kwargs.update(overrides)
    return kwargs


class TestValidConfig(unittest.TestCase):
    def test_minimal_valid_config(self):
        config = ExperimentConfig(**_base_kwargs())
        self.assertEqual(config.condition, "drift")
        self.assertEqual(config.seed, 42)

    def test_no_drift_condition_accepted(self):
        config = ExperimentConfig(**_base_kwargs(condition="no_drift"))
        self.assertEqual(config.condition, "no_drift")

    def test_all_approved_seeds_accepted(self):
        for seed in APPROVED_SEEDS:
            config = ExperimentConfig(**_base_kwargs(seed=seed))
            self.assertEqual(config.seed, seed)


class TestInvalidConfig(unittest.TestCase):
    def test_invalid_condition_rejected(self):
        with self.assertRaises(InvalidConfigurationError):
            ExperimentConfig(**_base_kwargs(condition="sometimes_drift"))

    def test_unapproved_seed_rejected(self):
        with self.assertRaises(InvalidSeedError):
            ExperimentConfig(**_base_kwargs(seed=999))

    def test_window_size_too_small_rejected(self):
        with self.assertRaises(InvalidConfigurationError):
            ExperimentConfig(**_base_kwargs(window_size=1))

    def test_negative_t_event_rejected(self):
        with self.assertRaises(InvalidConfigurationError):
            ExperimentConfig(**_base_kwargs(t_event=0))

    def test_zero_retrain_window_rejected(self):
        with self.assertRaises(InvalidConfigurationError):
            ExperimentConfig(**_base_kwargs(retrain_window_size=0))

    def test_zero_k_jaccard_rejected(self):
        with self.assertRaises(InvalidConfigurationError):
            ExperimentConfig(**_base_kwargs(k_jaccard=0))

    def test_nonpositive_scale_multiplier_rejected(self):
        with self.assertRaises(InvalidConfigurationError):
            ExperimentConfig(**_base_kwargs(drift_scale_multiplier=0.0))


class TestMetadata(unittest.TestCase):
    def test_metadata_json_serializable(self):
        import json
        config = ExperimentConfig(**_base_kwargs())
        json.dumps(config.to_metadata_dict())  # must not raise

    def test_metadata_reflects_actual_values(self):
        config = ExperimentConfig(**_base_kwargs(seed=101, window_size=15))
        payload = config.to_metadata_dict()
        self.assertEqual(payload["seed"], 101)
        self.assertEqual(payload["window_size"], 15)


if __name__ == "__main__":
    unittest.main()

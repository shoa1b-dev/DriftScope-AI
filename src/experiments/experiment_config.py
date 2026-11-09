"""Experiment configuration.

STATUS: IMPLEMENTED, unit-tested.

Every field that is genuinely dataset-dependent (what a real NAB series
would require) is a plain constructor argument here — there is no
built-in "sensible default" for anything like drift magnitude or window
size, because your instruction is explicit: those must never be chosen
without real NAB inspection. This class validates STRUCTURE (is the
seed one of the five approved values? is window_size >= 2?), never
CONTENT (it has no opinion on what a "good" drift magnitude is).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.checkpoints.checkpoint_manager import VALID_CONDITIONS
from src.detectors.detector_b_autoencoder import DEFAULT_HIDDEN_LAYER_SIZES

APPROVED_SEEDS = (13, 42, 101, 202, 303)


class InvalidConfigurationError(ValueError):
    """Raised for any structurally invalid configuration value."""


class InvalidSeedError(InvalidConfigurationError):
    """Raised when seed is not one of the five approved, locked seeds."""


@dataclass(frozen=True)
class ExperimentConfig:
    condition: str                 # "drift" or "no_drift"
    seed: int                      # must be one of APPROVED_SEEDS
    window_size: int
    t_event: int                   # raw-series index: drift injection point ("drift") or matched arbitrary point ("no_drift") — see experiment_runner module docstring for why one field serves both roles
    stride: int = 1
    drift_offset: float = 0.0
    drift_scale_multiplier: float = 1.0
    retrain_window_size: int = 50
    background_sample_size: int = 20
    evaluation_sample_size: int = 10
    background_random_state: int = 777
    evaluation_random_state: int = 555
    k_jaccard: int = 2
    detection_threshold: float = 0.0
    post_retrain_window_size: int | None = None   # TBD — Phase 8.1 decision: if None, use all remaining data (documented fallback); if set, matches the proposal's "matched to W where series length allows" design (Phase 13 §13)
    detector_b_hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES
    detector_b_max_iter: int = 500
    extra_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.condition not in VALID_CONDITIONS:
            raise InvalidConfigurationError(
                f"condition must be one of {VALID_CONDITIONS}, got '{self.condition}'"
            )
        if self.seed not in APPROVED_SEEDS:
            raise InvalidSeedError(
                f"seed must be one of the five approved seeds {APPROVED_SEEDS}, got {self.seed}"
            )
        if self.window_size < 2:
            raise InvalidConfigurationError(f"window_size must be >= 2, got {self.window_size}")
        if self.stride < 1:
            raise InvalidConfigurationError(f"stride must be >= 1, got {self.stride}")
        if self.t_event < 1:
            raise InvalidConfigurationError(f"t_event must be >= 1, got {self.t_event}")
        if self.retrain_window_size < 1:
            raise InvalidConfigurationError(
                f"retrain_window_size must be >= 1, got {self.retrain_window_size}"
            )
        if self.background_sample_size < 1:
            raise InvalidConfigurationError("background_sample_size must be >= 1")
        if self.evaluation_sample_size < 1:
            raise InvalidConfigurationError("evaluation_sample_size must be >= 1")
        if self.k_jaccard < 1:
            raise InvalidConfigurationError(f"k_jaccard must be >= 1, got {self.k_jaccard}")
        if self.post_retrain_window_size is not None and self.post_retrain_window_size < 1:
            raise InvalidConfigurationError(
                f"post_retrain_window_size must be >= 1 or None, got {self.post_retrain_window_size}"
            )
        if self.drift_scale_multiplier <= 0:
            raise InvalidConfigurationError(
                f"drift_scale_multiplier must be > 0, got {self.drift_scale_multiplier}"
            )

    def to_metadata_dict(self) -> dict:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "window_size": self.window_size,
            "t_event": self.t_event,
            "stride": self.stride,
            "drift_offset": self.drift_offset,
            "drift_scale_multiplier": self.drift_scale_multiplier,
            "retrain_window_size": self.retrain_window_size,
            "background_sample_size": self.background_sample_size,
            "evaluation_sample_size": self.evaluation_sample_size,
            "background_random_state": self.background_random_state,
            "evaluation_random_state": self.evaluation_random_state,
            "k_jaccard": self.k_jaccard,
            "detection_threshold": self.detection_threshold,
            "post_retrain_window_size": self.post_retrain_window_size,
            "detector_b_hidden_layer_sizes": list(self.detector_b_hidden_layer_sizes),
            "detector_b_max_iter": self.detector_b_max_iter,
            "extra_metadata": self.extra_metadata,
        }

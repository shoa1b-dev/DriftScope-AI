"""Checkpoint framework.

STATUS: IMPLEMENTED, unit-tested against synthetic data + both detector
classes.

Manages the three checkpoints required by the approved proposal
(Phase 11 §13): pre_drift -> post_drift_pre_retrain -> post_retrain.
Order, duplication, and completeness are enforced structurally (raise,
not silently ignore) — not left as a convention for callers to remember.

Model state capture uses `copy.deepcopy` on the detector instance, so a
later `detector.fit(...)` call (retraining) cannot mutate an earlier
checkpoint's stored snapshot — verified directly in
tests/unit/test_checkpoint_manager.py for both Detector A and Detector B.

No-drift control handling: this module does NOT hardcode what
"post_drift_pre_retrain" means for the no_drift condition — it never
invents a drift event. Instead, any checkpoint captured with a non-None
`event_index` MUST also supply `extra_metadata["index_source"]`,
explicitly stating where that index came from (e.g.
"adwin_detection" for the drift condition, or "matched_arbitrary_point"
for the no_drift control, per the approved proposal's Phase 11 §14
control design). This makes "don't fabricate an event" a checked
contract, not just a documentation note.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

VALID_CHECKPOINTS = ("pre_drift", "post_drift_pre_retrain", "post_retrain")
VALID_CONDITIONS = ("drift", "no_drift")


class CheckpointOrderError(Exception):
    """Raised when a checkpoint is captured before its required
    predecessor(s)."""


class DuplicateCheckpointError(Exception):
    """Raised when a checkpoint is captured twice for the same run."""


class IncompleteCheckpointsError(Exception):
    """Raised when completeness is required but checkpoints are missing."""


class UnlabeledEventIndexError(Exception):
    """Raised when a non-pre_drift checkpoint supplies an event_index
    without stating index_source — prevents silently fabricated or
    unexplained events, per your explicit requirement."""


@dataclass
class CheckpointRecord:
    detector_name: str
    condition: str
    seed: int
    checkpoint: str
    event_index: int | None
    model_snapshot: object              # deepcopy of the detector — not JSON-safe, handled separately
    explanation_sample_signature: str
    config_snapshot: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def checkpoint_id(self) -> str:
        """Deterministic naming: {detector}_{condition}_{seed}_{checkpoint}"""
        return f"{self.detector_name}_{self.condition}_{self.seed}_{self.checkpoint}"

    def to_metadata_dict(self) -> dict:
        """JSON-serializable metadata only — excludes model_snapshot,
        which must be persisted separately (e.g. via pickle) since it
        is a live model object, not JSON-safe.

        This is the deliberate counterpart to
        ExplanationResult.to_metadata_dict(), which DOES include its
        numeric attributions: a live model object must never appear in
        a metadata dict anywhere in this project, but a plain numeric
        result is safe and expected to. The two functions differ
        because what they're serializing differs in kind, not because
        one of them is incomplete."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "detector_name": self.detector_name,
            "condition": self.condition,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "event_index": self.event_index,
            "explanation_sample_signature": self.explanation_sample_signature,
            "config_snapshot": self.config_snapshot,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_metadata_dict(d: dict, model_snapshot=None) -> "CheckpointRecord":
        """Reconstruct from a metadata dict (e.g. loaded from JSON).
        `model_snapshot` must be supplied separately since it was never
        part of the JSON-safe payload."""
        return CheckpointRecord(
            detector_name=d["detector_name"],
            condition=d["condition"],
            seed=d["seed"],
            checkpoint=d["checkpoint"],
            event_index=d["event_index"],
            model_snapshot=model_snapshot,
            explanation_sample_signature=d["explanation_sample_signature"],
            config_snapshot=d.get("config_snapshot", {}),
            metadata=d.get("metadata", {}),
        )


class CheckpointManager:
    def __init__(self, detector_name: str, condition: str, seed: int, explanation_sample):
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"condition must be one of {VALID_CONDITIONS}, got '{condition}'")
        self.detector_name = detector_name
        self.condition = condition
        self.seed = seed
        self.explanation_sample = explanation_sample
        self._records: dict[str, CheckpointRecord] = {}

    def capture(
        self,
        checkpoint: str,
        detector,
        event_index: int | None = None,
        config_snapshot: dict | None = None,
        extra_metadata: dict | None = None,
    ) -> CheckpointRecord:
        if checkpoint not in VALID_CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint '{checkpoint}', must be one of {VALID_CHECKPOINTS}")
        if checkpoint in self._records:
            raise DuplicateCheckpointError(
                f"Checkpoint '{checkpoint}' already captured for "
                f"{self.detector_name}/{self.condition}/seed={self.seed}"
            )

        required_prior = VALID_CHECKPOINTS[:VALID_CHECKPOINTS.index(checkpoint)]
        missing_prior = [c for c in required_prior if c not in self._records]
        if missing_prior:
            raise CheckpointOrderError(
                f"Cannot capture '{checkpoint}' before prior checkpoint(s) {missing_prior} "
                f"have been captured"
            )

        extra_metadata = dict(extra_metadata or {})
        if event_index is not None and "index_source" not in extra_metadata:
            raise UnlabeledEventIndexError(
                f"Checkpoint '{checkpoint}' supplies event_index={event_index} but no "
                "extra_metadata['index_source'] — every non-pre_drift event index must "
                "state where it came from (e.g. 'adwin_detection' or "
                "'matched_arbitrary_point'), so no event is ever silently fabricated."
            )

        snapshot = copy.deepcopy(detector)
        record = CheckpointRecord(
            detector_name=self.detector_name,
            condition=self.condition,
            seed=self.seed,
            checkpoint=checkpoint,
            event_index=event_index,
            model_snapshot=snapshot,
            explanation_sample_signature=self.explanation_sample.signature,
            config_snapshot=dict(config_snapshot or {}),
            metadata=extra_metadata,
        )
        self._records[checkpoint] = record
        return record

    def get(self, checkpoint: str) -> CheckpointRecord:
        if checkpoint not in self._records:
            raise KeyError(f"Checkpoint '{checkpoint}' has not been captured yet")
        return self._records[checkpoint]

    @property
    def captured_checkpoints(self) -> list[str]:
        return [c for c in VALID_CHECKPOINTS if c in self._records]

    @property
    def missing_checkpoints(self) -> list[str]:
        return [c for c in VALID_CHECKPOINTS if c not in self._records]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_checkpoints) == 0

    def require_complete(self) -> None:
        if not self.is_complete:
            raise IncompleteCheckpointsError(
                f"Missing checkpoint(s) {self.missing_checkpoints} for "
                f"{self.detector_name}/{self.condition}/seed={self.seed}"
            )

    def all_records(self) -> list[CheckpointRecord]:
        """All captured records, in checkpoint order (not insertion order)."""
        return [self._records[c] for c in VALID_CHECKPOINTS if c in self._records]

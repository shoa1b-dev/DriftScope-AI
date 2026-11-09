"""Retraining trigger foundation.

STATUS: IMPLEMENTED, unit-tested against synthetic data + MockDriftMonitor.

Design decision: detection and retraining are modeled as two DISTINCT
events with two distinct timestamps/indices (t_detect, t_retrain), per
your explicit requirement and the approved proposal's Phase 11 §18
three-timestamp discipline — even though in the eventual pipeline
retraining may be triggered immediately after detection, this class
never conflates the two into one field. A caller must explicitly call
`confirm_retrain(...)` to record t_retrain; it is never auto-set inside
`process(...)`.

retrain_window_size is accepted as a constructor parameter and may be
None (TBD). `get_retrain_slice(...)` raises ValueError if called while
it's still None — this is a deliberate hard stop, not a guessed default,
per your instruction not to guess this value before real NAB inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.monitoring.drift_monitor import DriftMonitorBase


@dataclass
class RetrainEvent:
    """A real record of what happened — every field is either None
    (not yet occurred) or an actual observed index, never a placeholder
    number."""

    detection_index: int | None = None
    retrain_index: int | None = None

    @property
    def is_detected(self) -> bool:
        return self.detection_index is not None

    @property
    def is_retrained(self) -> bool:
        return self.retrain_index is not None


class RetrainTrigger:
    def __init__(self, monitor: DriftMonitorBase, retrain_window_size: int | None = None):
        self.monitor = monitor
        self.retrain_window_size = retrain_window_size
        self.event = RetrainEvent()

    def process(self, score: float, index: int) -> bool:
        """Feed one anomaly-score value to the underlying monitor.
        Returns True exactly once — on the update where drift is newly
        detected. Does NOT trigger retraining itself; that is a
        separate, explicit step (see class docstring)."""
        newly_detected = self.monitor.update(score, index)
        if newly_detected:
            self.event.detection_index = index
        return newly_detected

    @property
    def retrain_pending(self) -> bool:
        """True once drift has been detected but retraining has not
        yet been confirmed — the window during which the approved
        proposal's 'post-drift/pre-retrain' checkpoint is defined."""
        return self.event.is_detected and not self.event.is_retrained

    def confirm_retrain(self, index: int) -> None:
        """Explicitly record that retraining occurred at `index`. Raises
        if called before detection (retraining without a detected drift
        is not a supported flow in this design) or if already retrained
        for this detection (prevents silent double-counting)."""
        if not self.event.is_detected:
            raise RuntimeError("confirm_retrain called before drift was detected")
        if self.event.is_retrained:
            raise RuntimeError("confirm_retrain called twice for the same detection event")
        self.event.retrain_index = index

    def get_retrain_slice(self, values: list, up_to_index: int) -> list:
        """Return the most recent `retrain_window_size` values up to
        (and including) `up_to_index`. Raises ValueError if
        retrain_window_size is still None (TBD, pending real NAB
        inspection) — never silently assumes a default window size."""
        if self.retrain_window_size is None:
            raise ValueError(
                "retrain_window_size is not set (TBD pending real NAB "
                "inspection) — cannot compute a retrain slice yet."
            )
        start = max(0, up_to_index + 1 - self.retrain_window_size)
        return values[start:up_to_index + 1]

    def reset_for_next_cycle(self) -> None:
        """Reset both the monitor and the event record, e.g. if this
        trigger will be reused to watch a subsequent segment."""
        self.monitor.reset()
        self.event = RetrainEvent()

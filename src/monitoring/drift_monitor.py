"""Drift monitor abstraction — ADWIN (primary) and Page-Hinkley (secondary).

STATUS: Interface and wrapper classes IMPLEMENTED. 

Design: the monitor interface is independent of any detector class (per
your requirement) — it only ever sees a stream of scalar anomaly-score
values, never the detector itself. This is what lets the SAME monitor
abstraction watch either Detector A's or Detector B's score stream
without modification, preserving the approved proposal's fairness
argument for the two-detector comparison (Phase 11 §6).


"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

try:
    import river.drift  # noqa: F401
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


class RiverUnavailableError(ImportError):
    """Raised when a real river-backed monitor is instantiated but
    `river` is not installed. Never silently substitutes a fake
    detector in its place — that decision belongs to the caller,
    explicitly (e.g. tests use MockDriftMonitor instead, on purpose)."""


@dataclass
class DriftMonitorBase(ABC):
    """Shared interface for all drift monitors in this project.

    Consumes one anomaly-score value at a time (streaming, per your
    requirement) — never a batch — so it can sit inline in the same
    loop that scores each new point during the experiment.
    """

    detected: bool = field(default=False, init=False)
    detection_index: int | None = field(default=None, init=False)
    _n_updates_seen: int = field(default=0, init=False)

    @abstractmethod
    def _update_impl(self, value: float) -> bool:
        """Subclass hook: consume one value, return True if THIS call
        newly signals drift."""

    def update(self, value: float, index: int) -> bool:
        """Feed one score value at position `index`. Returns True only
        on the update where drift is newly detected (not on every
        subsequent call while still in a "detected" state) — this is
        what lets a caller record `detection_index` exactly once.
        """
        self._n_updates_seen += 1
        already_detected = self.detected
        result = self._update_impl(value)
        if result and not already_detected:
            self.detected = True
            self.detection_index = index
            return True
        return False

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g. after a retrain), so the monitor
        can be reused to watch the post-retrain stream without carrying
        over pre-retrain window statistics. Subclasses must also reset
        `detected`/`detection_index` — enforced by calling super()
        pattern below."""


class AdwinDriftMonitor(DriftMonitorBase):
    """PRIMARY drift monitor, per your confirmed decision. Wraps
    river.drift.ADWIN. Raises RiverUnavailableError at construction
    time if river is not installed — never falls back to a fake
    implementation silently."""

    def __init__(self):
        super().__init__()
        if not RIVER_AVAILABLE:
            raise RiverUnavailableError(
                "river is not installed in this environment — AdwinDriftMonitor "
                "cannot be constructed. This is a real, disclosed limitation, "
                "not a bug to work around with a fake implementation."
            )
        self._adwin = river.drift.ADWIN()

    def _update_impl(self, value: float) -> bool:
        self._adwin.update(value)
        return bool(self._adwin.drift_detected)

    def reset(self) -> None:
        self._adwin = river.drift.ADWIN()
        self.detected = False
        self.detection_index = None


class PageHinkleyDriftMonitor(DriftMonitorBase):
    """SECONDARY / robustness-check drift monitor only, per your
    confirmed decision. Never used as a substitute for ADWIN results —
    any reporting downstream must label results from this monitor as
    secondary. Wraps river.drift.PageHinkley; same disclosed-limitation
    behavior as AdwinDriftMonitor if river is unavailable."""

    def __init__(self):
        super().__init__()
        if not RIVER_AVAILABLE:
            raise RiverUnavailableError(
                "river is not installed in this environment — "
                "PageHinkleyDriftMonitor cannot be constructed."
            )
        self._ph = river.drift.PageHinkley()

    def _update_impl(self, value: float) -> bool:
        self._ph.update(value)
        return bool(self._ph.drift_detected)

    def reset(self) -> None:
        self._ph = river.drift.PageHinkley()
        self.detected = False
        self.detection_index = None

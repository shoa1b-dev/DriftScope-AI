"""Mock drift monitor — TEST FIXTURE ONLY.

STATUS: This is a fake, deliberately simplistic monitor used ONLY to
test RetrainTrigger and other pipeline wiring without requiring real
`river`-backed ADWIN/Page-Hinkley. It must never be imported from
`src/` production code and never appears in any results reporting.

Detection rule (fake, for testing only): drift is "detected" the first
time an incoming value exceeds a fixed, caller-supplied threshold. This
has no relationship to real ADWIN/Page-Hinkley statistical behavior —
it exists purely so tests can deterministically trigger "detected"
state on a known input, to verify what happens AFTER detection (e.g.
retrain trigger wiring), not to test detection quality itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.monitoring.drift_monitor import DriftMonitorBase


class MockDriftMonitor(DriftMonitorBase):
    """FAKE — test fixture only. See module docstring."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def _update_impl(self, value: float) -> bool:
        return value > self.threshold

    def reset(self) -> None:
        self.detected = False
        self.detection_index = None

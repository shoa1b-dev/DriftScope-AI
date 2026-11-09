"""Detector A — Isolation Forest.

STATUS: IMPLEMENTED, unit-tested against synthetic data. Not yet run
against real NAB features (depends on real window_size, still TBD).

Design decisions (documented, per the approved proposal Phase 11 §4):
- sklearn's IsolationForest.decision_function returns HIGHER values for
  MORE NORMAL points (opposite convention to score_samples' raw path-
  length). We invert here so this detector's anomaly_score() follows
  the BaseDetector convention (higher = more anomalous), consistently
  with Detector B — this matters because the drift monitor and
  downstream evaluation treat both detectors identically (Phase 11 §6).
- random_state is per-seed, per the reproducibility.seeds config list —
  never a fixed constant, so the 5-seed grid is genuinely 5 different
  initializations, not 5 identical runs.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import IsolationForest

from src.detectors.base_detector import BaseDetector


class DetectorAIsolationForest(BaseDetector):
    def __init__(self, random_state: int, n_estimators: int = 100):
        super().__init__(random_state)
        self._model = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> "DetectorAIsolationForest":
        self._model.fit(X)
        self._fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DetectorAIsolationForest.anomaly_score called before fit()")
        # decision_function: high = normal, low/negative = anomalous.
        # Negate so higher = more anomalous (BaseDetector convention).
        return -self._model.decision_function(X)

    @property
    def raw_model(self):
        """The underlying sklearn IsolationForest — exposed for
        shap.TreeExplainer, which needs the raw tree-ensemble object
        directly, not our anomaly_score() wrapper. This IS necessary:
        TreeExplainer has no equivalent way to explain a wrapped
        callable the way the permutation backend does for Detector B.

        Returns a DEEPCOPY, not the live internal model, on every
        access — so nothing downstream (TreeExplainer or otherwise)
        can mutate this detector's actual internal state through the
        returned object. Since this is typically called on an already-
        deep-copied checkpoint snapshot (see CheckpointManager.capture),
        this is a second, redundant layer of isolation — cheap at this
        model size, and it means the safety property holds even if
        raw_model is ever called on a live (non-snapshotted) detector."""
        return copy.deepcopy(self._model)

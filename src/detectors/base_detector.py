"""Shared detector interface.

STATUS: IMPLEMENTED. Both Detector A (Isolation Forest) and Detector B
(MLP autoencoder) implement this interface, so the rest of the pipeline
(drift monitoring, retraining, SHAP explanation, evaluation) can treat
them uniformly — this is what keeps the two-detector comparison (RO4)
methodologically fair, per the approved proposal's Phase 11 §6
justification (both detectors must expose the same kind of signal).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Common interface: fit on windowed features, produce a continuous
    anomaly score per row. Higher score = more anomalous, by convention
    enforced in each subclass (raw library conventions differ and are
    normalized here, not left inconsistent).
    """

    def __init__(self, random_state: int):
        self.random_state = random_state
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        """Fit (or re-fit, for retraining) the detector on windowed
        feature rows X, shape (n_samples, n_features)."""

    @abstractmethod
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D array of continuous anomaly scores, one per row
        of X. Higher = more anomalous. This is the signal the drift
        monitor watches (per the locked design decision: each
        detector's OWN anomaly-score stream)."""

    @property
    def fitted(self) -> bool:
        return self._fitted

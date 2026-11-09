"""Detector B — MLP-based bottleneck autoencoder.

STATUS: IMPLEMENTED, unit-tested against synthetic data. Not yet run
against real NAB features.

Design decisions (documented, since these must be defensible):

hidden_layer_sizes default = (4, 2, 4). This IS locked now, unlike the
dataset-dependent values elsewhere in this project, because it depends
only on the fixed feature-vector dimensionality our own preprocessing
produces (5 features: mean/std/min/max/slope — see
src/data/preprocessing.py), not on anything NAB-specific. A 5 -> 4 -> 2
-> 4 -> 5 bottleneck is a small, defensible compression given only 5
input dimensions; a deeper/wider network would be unjustifiable
over-parameterization for this input size and would risk violating the
CPU-only / lightweight-detector constraint.

Autoencoder-via-MLPRegressor pattern: MLPRegressor is trained with
y = X (reconstruct its own input), which sklearn supports natively as
multi-output regression. This is the standard "autoencoder without a
deep-learning framework" technique referenced in the approved proposal
(Phase 11 §5) — not a novel technique invented here.

anomaly_score() returns per-row reconstruction error (mean squared
error between input and reconstruction) — higher = worse reconstruction
= more anomalous, matching Detector A's convention after its inversion.

Phase 5.1 note: a `raw_model` property (returning the live MLPRegressor)
was removed from this class — it was added in Phase 5 "for symmetry"
with Detector A but was never actually used anywhere in the SHAP
pipeline (Detector B's explanation target is `anomaly_score`, not the
raw model — see src/explanation/shap_explainer.py). Exposing a mutable
internal object with no real consumer was an unnecessary risk, so it
was removed rather than kept "just in case". `model_class_name` below
replaces the one legitimate need (metadata reporting) with a safe,
string-only accessor instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

from src.detectors.base_detector import BaseDetector

DEFAULT_HIDDEN_LAYER_SIZES = (4, 2, 4)


class DetectorBAutoencoder(BaseDetector):
    def __init__(
        self,
        random_state: int,
        hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES,
        max_iter: int = 500,
    ):
        super().__init__(random_state)
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
        )

    def fit(self, X: np.ndarray) -> "DetectorBAutoencoder":
        X = np.asarray(X, dtype=float)
        self._model.fit(X, X)  # reconstruct its own input
        self._fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DetectorBAutoencoder.anomaly_score called before fit()")
        X = np.asarray(X, dtype=float)
        reconstruction = self._model.predict(X)
        # Per-row mean squared reconstruction error.
        return np.mean((X - reconstruction) ** 2, axis=1)

    @property
    def model_class_name(self) -> str:
        """Plain string identifying the underlying sklearn model class
        (e.g. "MLPRegressor") — used for metadata/reporting. Unlike a
        `raw_model`-style property, this returns only a string, never a
        live mutable object, so there is no unsafe-exposure concern for
        this accessor."""
        return type(self._model).__name__

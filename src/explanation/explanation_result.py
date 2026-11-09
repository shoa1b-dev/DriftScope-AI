"""ExplanationResult — the normalized output of SHAP explanation generation.

STATUS: IMPLEMENTED, unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExplanationResult:
    detector_name: str
    condition: str
    seed: int
    checkpoint: str
    feature_names: tuple[str, ...]
    attributions: np.ndarray            # shape (n_samples, n_features), normalized
    sample_shape: tuple[int, ...]
    explanation_sample_signature: str
    shap_backend: str
    model_type: str
    background_sample_signature: str | None = None   # None for TreeExplainer (no background needed); required for the permutation backend
    config_snapshot: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        """JSON-serializable metadata. Deliberately asymmetric with
        CheckpointRecord.to_metadata_dict(): attributions are plain
        numeric results, safe and intended to be serialized here.
        A live model object (like CheckpointRecord.model_snapshot) must
        NEVER appear in a metadata dict anywhere in this project — this
        function contains no such object, by design, not by omission."""
        return {
            "detector_name": self.detector_name,
            "condition": self.condition,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "feature_names": list(self.feature_names),
            "attributions": self.attributions.tolist(),
            "sample_shape": list(self.sample_shape),
            "explanation_sample_signature": self.explanation_sample_signature,
            "background_sample_signature": self.background_sample_signature,
            "shap_backend": self.shap_backend,
            "model_type": self.model_type,
            "config_snapshot": self.config_snapshot,
        }

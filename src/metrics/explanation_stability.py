"""Explanation-stability metrics — cosine similarity (primary) and
top-k feature-overlap Jaccard (secondary).

STATUS: IMPLEMENTED, unit-tested against synthetic attribution vectors.

Matrix-level aggregation (documented per your requirement): for a
matrix of attribution vectors (n_samples, n_features), this module
computes ONE metric value PER SAMPLE ROW (comparing sample i of A
against sample i of B), then reports the arithmetic mean across
samples. This is the ONLY aggregation method implemented — it was
chosen over a single "flattened" cosine similarity across the whole
matrix because per-sample comparison preserves the pairing between a
specific explained point's attribution vector before and after
drift/retraining; flattening would mix different points' attributions
together and answer a different (less meaningful) question.

Approved checkpoint comparisons (enforced at the top-level API,
`compute_stability_metrics`): pre_drift -> post_drift_pre_retrain, and
pre_drift -> post_retrain. A generic pairwise function
(`cosine_similarity_matrix`, `jaccard_overlap_matrix`) is available
internally/for testing, but does not itself know about checkpoints —
the approved-pair restriction is enforced only by
`compute_stability_metrics`, per your instruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.explanation.explanation_result import ExplanationResult

# ----------------------------------------------------------------------
# Errors — every invalid case raises a specific, documented exception.
# Nothing in this module silently returns a misleading value.
# ----------------------------------------------------------------------


class ZeroVectorError(ValueError):
    """Raised when cosine similarity is requested for a zero-norm
    vector. Cosine similarity is mathematically undefined for a zero
    vector (division by zero) — there is no single "correct" default
    (0.0? 1.0? NaN?), so this is raised explicitly rather than picking
    one silently. See `on_invalid` in the matrix functions for how to
    handle this at scale without stopping the whole computation."""


class NonFiniteValueError(ValueError):
    """Raised when an attribution vector contains NaN or +/-Inf."""


class ShapeMismatchError(ValueError):
    """Raised when two attribution matrices/vectors have incompatible
    shapes for comparison."""


class InvalidKError(ValueError):
    """Raised when k < 1 or k > number of features."""


class InvalidComparisonError(ValueError):
    """Raised when a checkpoint pair is not one of the two approved
    comparisons."""


class InconsistentExplanationsError(ValueError):
    """Raised when two ExplanationResult objects being compared don't
    actually belong to the same run (different detector/condition/seed/
    sample/feature set) — comparing them would be meaningless."""


def _validate_finite(vec: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(vec)):
        raise NonFiniteValueError(f"{label} contains NaN or infinite values")


# ----------------------------------------------------------------------
# Cosine similarity (primary metric)
# ----------------------------------------------------------------------


def cosine_similarity_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D attribution vectors.

    Raises NonFiniteValueError if either vector contains NaN/Inf.
    Raises ZeroVectorError if either vector has zero norm.
    Result is clipped to [-1.0, 1.0] to guard against float rounding
    pushing a value slightly outside the mathematically valid range.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ShapeMismatchError(f"Vector shapes differ: {a.shape} vs {b.shape}")

    _validate_finite(a, "vector a")
    _validate_finite(b, "vector b")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ZeroVectorError(
            f"Cannot compute cosine similarity: norm_a={norm_a}, norm_b={norm_b} "
            "(at least one vector is all-zero)"
        )

    value = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, value))


@dataclass
class PerSampleMetricResult:
    """Shared result shape for both metrics' matrix/batch functions."""

    values: np.ndarray          # shape (n_samples,); may contain NaN in "mark_nan" mode
    n_valid: int
    n_invalid: int
    invalid_indices: tuple[int, ...] = field(default_factory=tuple)

    @property
    def mean(self) -> float:
        """nanmean across samples — invalid (NaN) rows are excluded
        from the mean, not treated as 0."""
        if self.n_valid == 0:
            return float("nan")
        return float(np.nanmean(self.values))

    def to_dict(self) -> dict:
        """JSON-serializable — NaN values become `null`, since strict
        JSON has no NaN literal."""
        return {
            "values": [None if math.isnan(v) else float(v) for v in self.values],
            "mean": None if math.isnan(self.mean) else self.mean,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "invalid_indices": list(self.invalid_indices),
        }


def cosine_similarity_matrix(
    A: np.ndarray, B: np.ndarray, on_invalid: str = "raise"
) -> PerSampleMetricResult:
    """Per-sample cosine similarity between matching rows of A and B,
    aggregated by mean (see module docstring for why per-sample, not
    flattened).

    `on_invalid`:
      - "raise" (default): the first invalid row (zero-vector or
        non-finite) raises immediately — strict, safest default.
      - "mark_nan": invalid rows are recorded as NaN in `.values` and
        excluded from the mean; valid rows are still computed. Useful
        when a caller wants a partial result rather than an all-or-
        nothing failure, but must be opted into explicitly.
    """
    if on_invalid not in ("raise", "mark_nan"):
        raise ValueError(f"on_invalid must be 'raise' or 'mark_nan', got '{on_invalid}'")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ShapeMismatchError(f"Expected 2D matrices, got shapes {A.shape} and {B.shape}")
    if A.shape != B.shape:
        raise ShapeMismatchError(f"Matrix shapes differ: {A.shape} vs {B.shape}")

    n_samples = A.shape[0]
    values = np.full(n_samples, np.nan)
    invalid_indices = []

    for i in range(n_samples):
        try:
            values[i] = cosine_similarity_pair(A[i], B[i])
        except (ZeroVectorError, NonFiniteValueError):
            if on_invalid == "raise":
                raise
            invalid_indices.append(i)

    n_invalid = len(invalid_indices)
    return PerSampleMetricResult(
        values=values, n_valid=n_samples - n_invalid, n_invalid=n_invalid,
        invalid_indices=tuple(invalid_indices),
    )


# ----------------------------------------------------------------------
# Top-k Jaccard overlap (secondary metric)
# ----------------------------------------------------------------------


def select_top_k_features(
    attribution_vector: np.ndarray, k: int, feature_names: Sequence[str]
) -> tuple[str, ...]:
    """Select the top-k feature NAMES by absolute attribution
    magnitude. Deterministic tie-breaking: ties (including an all-zero
    vector, where every feature ties at magnitude 0) are broken by
    original feature index — the lowest-index feature among tied
    values wins — so results never depend on sort-algorithm stability
    quirks or random tie resolution.
    """
    n_features = len(attribution_vector)
    if len(feature_names) != n_features:
        raise ShapeMismatchError(
            f"feature_names has {len(feature_names)} entries but attribution_vector "
            f"has {n_features}"
        )
    if k < 1:
        raise InvalidKError(f"k must be >= 1, got {k}")
    if k > n_features:
        raise InvalidKError(f"k must be <= number_of_features ({n_features}), got {k}")

    _validate_finite(np.asarray(attribution_vector, dtype=float), "attribution_vector")

    order = sorted(range(n_features), key=lambda i: (-abs(attribution_vector[i]), i))
    top_k_indices = order[:k]
    return tuple(feature_names[i] for i in top_k_indices)


def jaccard_overlap_pair(
    a: np.ndarray, b: np.ndarray, k: int, feature_names: Sequence[str]
) -> float:
    """Top-k Jaccard overlap between two attribution vectors:
    |top_k(a) ∩ top_k(b)| / |top_k(a) ∪ top_k(b)|.
    """
    set_a = set(select_top_k_features(a, k, feature_names))
    set_b = set(select_top_k_features(b, k, feature_names))
    union = set_a | set_b
    if not union:  # unreachable in practice since k >= 1, but no silent 0/0
        raise InvalidKError("Union of top-k feature sets is empty — invalid k or feature set")
    return len(set_a & set_b) / len(union)


def jaccard_overlap_matrix(
    A: np.ndarray, B: np.ndarray, k: int, feature_names: Sequence[str], on_invalid: str = "raise"
) -> PerSampleMetricResult:
    """Per-sample top-k Jaccard overlap between matching rows of A and
    B, aggregated by mean. Same `on_invalid` semantics as
    `cosine_similarity_matrix`, but note Jaccard has no zero-vector
    failure mode (an all-zero row is a valid, deterministic case — see
    `select_top_k_features`) — only non-finite values can make a row
    invalid here.
    """
    if on_invalid not in ("raise", "mark_nan"):
        raise ValueError(f"on_invalid must be 'raise' or 'mark_nan', got '{on_invalid}'")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ShapeMismatchError(f"Expected 2D matrices, got shapes {A.shape} and {B.shape}")
    if A.shape != B.shape:
        raise ShapeMismatchError(f"Matrix shapes differ: {A.shape} vs {B.shape}")

    n_samples = A.shape[0]
    values = np.full(n_samples, np.nan)
    invalid_indices = []

    for i in range(n_samples):
        try:
            values[i] = jaccard_overlap_pair(A[i], B[i], k, feature_names)
        except NonFiniteValueError:
            if on_invalid == "raise":
                raise
            invalid_indices.append(i)

    n_invalid = len(invalid_indices)
    return PerSampleMetricResult(
        values=values, n_valid=n_samples - n_invalid, n_invalid=n_invalid,
        invalid_indices=tuple(invalid_indices),
    )


# ----------------------------------------------------------------------
# Top-level, checkpoint-aware API — the only entry point that enforces
# the approved comparisons.
# ----------------------------------------------------------------------

APPROVED_COMPARISONS = (
    ("pre_drift", "post_drift_pre_retrain"),
    ("pre_drift", "post_retrain"),
)


@dataclass
class StabilityMetricsResult:
    cosine: PerSampleMetricResult
    jaccard: PerSampleMetricResult
    detector_name: str
    condition: str
    seed: int
    checkpoint_pair: tuple[str, str]
    n_samples: int
    n_features: int
    k: int
    aggregation_method: str = "mean"

    def to_metadata_dict(self) -> dict:
        return {
            "metric_names": ["cosine_similarity", "top_k_jaccard_overlap"],
            "checkpoint_pair": list(self.checkpoint_pair),
            "detector_name": self.detector_name,
            "condition": self.condition,
            "seed": self.seed,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "k": self.k,
            "aggregation_method": self.aggregation_method,
            "cosine": self.cosine.to_dict(),
            "jaccard": self.jaccard.to_dict(),
        }


def compute_stability_metrics(
    explanation_a: ExplanationResult,
    explanation_b: ExplanationResult,
    k: int,
    on_invalid_cosine: str = "raise",
    on_invalid_jaccard: str = "raise",
) -> StabilityMetricsResult:
    """Compute BOTH approved stability metrics for one approved
    checkpoint pair. `explanation_a` must be the 'pre_drift' checkpoint
    (the fixed baseline); `explanation_b` must be either
    'post_drift_pre_retrain' or 'post_retrain' — any other pair raises
    InvalidComparisonError, even if the caller has a generic pairwise
    function available internally.
    """
    pair = (explanation_a.checkpoint, explanation_b.checkpoint)
    if pair not in APPROVED_COMPARISONS:
        raise InvalidComparisonError(
            f"Checkpoint pair {pair} is not an approved comparison. "
            f"Approved comparisons: {APPROVED_COMPARISONS}"
        )

    if explanation_a.detector_name != explanation_b.detector_name:
        raise InconsistentExplanationsError("detector_name differs between explanations")
    if explanation_a.condition != explanation_b.condition:
        raise InconsistentExplanationsError("condition differs between explanations")
    if explanation_a.seed != explanation_b.seed:
        raise InconsistentExplanationsError("seed differs between explanations")
    if explanation_a.explanation_sample_signature != explanation_b.explanation_sample_signature:
        raise InconsistentExplanationsError(
            "explanation_sample_signature differs — these explanations were not "
            "computed on the same fixed evaluation sample"
        )
    if explanation_a.feature_names != explanation_b.feature_names:
        raise InconsistentExplanationsError("feature_names differ between explanations")

    cosine_result = cosine_similarity_matrix(
        explanation_a.attributions, explanation_b.attributions, on_invalid=on_invalid_cosine
    )
    jaccard_result = jaccard_overlap_matrix(
        explanation_a.attributions, explanation_b.attributions, k=k,
        feature_names=explanation_a.feature_names, on_invalid=on_invalid_jaccard,
    )

    return StabilityMetricsResult(
        cosine=cosine_result,
        jaccard=jaccard_result,
        detector_name=explanation_a.detector_name,
        condition=explanation_a.condition,
        seed=explanation_a.seed,
        checkpoint_pair=pair,
        n_samples=explanation_a.attributions.shape[0],
        n_features=explanation_a.attributions.shape[1],
        k=k,
    )

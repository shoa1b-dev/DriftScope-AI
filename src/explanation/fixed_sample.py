"""Fixed explanation sample — selected once, from pre-drift data only.

STATUS: IMPLEMENTED, unit-tested against synthetic data.

Implements the approved proposal's leakage-prevention design (Phase 11
§13): a fixed sample of evaluation points, drawn only from the pre-drift
region, reused UNCHANGED at every checkpoint. The sample is never
re-drawn after drift, and this module structurally prevents post-drift
data from entering it — `build_fixed_sample_from_full_series` slices to
`t_inject` internally before sampling, so a caller cannot leak post-drift
rows into the sample even by passing the full feature matrix.

The returned array is marked read-only (`writeable = False`) as an
enforced immutability guarantee, not just a naming convention — any
attempted mutation raises, which is verified directly in
tests/unit/test_fixed_sample.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


class FixedSampleError(Exception):
    """Raised for invalid fixed-sample construction — never silently
    clamped or worked around."""


@dataclass(frozen=True)
class FixedExplanationSample:
    """Immutable record of the fixed explanation sample. `features` is
    a read-only numpy array (enforced, see build functions below)."""

    source_indices: tuple[int, ...]
    features: np.ndarray
    random_state: int
    n_samples: int

    @property
    def signature(self) -> str:
        """Short, deterministic identity string for this sample — used
        to prove the SAME sample was reused across checkpoints without
        having to compare full arrays every time (e.g. in checkpoint
        metadata dicts, which must stay JSON-serializable)."""
        payload = f"{self.source_indices}:{self.random_state}:{self.n_samples}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_fixed_sample(pre_drift_features: np.ndarray, n_samples: int, random_state: int) -> FixedExplanationSample:
    """Build a fixed sample from a features array that is ALREADY
    restricted to pre-drift rows by the caller. Prefer
    `build_fixed_sample_from_full_series` below when you have the full
    series available — it enforces the pre-drift restriction itself
    rather than trusting the caller.
    """
    n_available = len(pre_drift_features)
    if n_samples > n_available:
        raise FixedSampleError(
            f"Requested n_samples={n_samples} exceeds available pre-drift rows ({n_available})"
        )
    if n_samples < 1:
        raise FixedSampleError(f"n_samples must be >= 1, got {n_samples}")

    rng = np.random.default_rng(random_state)
    chosen = rng.choice(n_available, size=n_samples, replace=False)
    source_indices = tuple(sorted(int(i) for i in chosen))

    features = np.asarray(pre_drift_features)[list(source_indices)].copy()
    features.setflags(write=False)  # enforced immutability, not just convention

    return FixedExplanationSample(
        source_indices=source_indices,
        features=features,
        random_state=random_state,
        n_samples=n_samples,
    )


def build_fixed_sample_from_full_series(
    all_features: np.ndarray, t_inject: int, n_samples: int, random_state: int
) -> FixedExplanationSample:
    """Leakage-safe wrapper: slices `all_features` to [:t_inject]
    BEFORE sampling, so post-drift rows structurally cannot be selected
    — this is the preferred entry point when the caller has the full
    (pre+post drift) feature matrix, which is the normal situation once
    drift injection has produced one combined array.
    """
    if not (0 < t_inject <= len(all_features)):
        raise FixedSampleError(
            f"t_inject must satisfy 0 < t_inject <= {len(all_features)}; got {t_inject}"
        )
    pre_drift_only = all_features[:t_inject]
    return build_fixed_sample(pre_drift_only, n_samples=n_samples, random_state=random_state)


def build_background_and_evaluation_samples(
    all_features: np.ndarray,
    t_inject: int,
    n_background: int,
    n_evaluation: int,
    background_random_state: int,
    evaluation_random_state: int,
) -> tuple[FixedExplanationSample, FixedExplanationSample]:
    """Build TWO independent fixed samples from the pre-drift region —
    one for the permutation background distribution, one for the
    points actually explained at each checkpoint (Phase 5.1 decision:
    Option B). Both are drawn via the same leakage-safe, pre-drift-only
    mechanism as `build_fixed_sample_from_full_series`; using different
    random_state values is what makes them independent draws.

    Overlap between the two samples' indices is permitted and NOT a
    leakage concern — both are still strictly confined to [:t_inject].
    Overlap is simply not disallowed; it is a standard, accepted
    practice in SHAP usage (background sets commonly include points
    also being explained), unlike using post-drift data, which would be
    a genuine leakage error.
    """
    background_sample = build_fixed_sample_from_full_series(
        all_features, t_inject, n_samples=n_background, random_state=background_random_state
    )
    evaluation_sample = build_fixed_sample_from_full_series(
        all_features, t_inject, n_samples=n_evaluation, random_state=evaluation_random_state
    )
    return background_sample, evaluation_sample

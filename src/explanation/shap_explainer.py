"""SHAP explanation generation — detector-specific backend selection.

STATUS: Backend-dispatch logic, validation, shape normalization, and
metadata construction are IMPLEMENTED and unit-tested (using an
injected fake in place of the real shap library — see `raw_value_fn`
parameter below). REAL shap execution is UNVERIFIED in this
environment: `shap` is not installed here and this sandbox has no
network egress to install it (same disclosed limitation as `river`).

Approved design (per your instructions):
- Detector A (Isolation Forest): shap.TreeExplainer on the raw sklearn
  estimator. No background sample is needed — TreeExplainer's default
  (tree_path_dependent) mode computes exact attributions from the tree
  structure itself.
- Detector B (MLP autoencoder): shap.Explainer with a permutation
  backend, wrapping the detector's OWN anomaly_score() method — this is
  what makes the explanation target "the model's anomaly score", not an
  arbitrary classifier output.

Phase 5.1 revision (background sample): the permutation backend now
REQUIRES a separate `background_sample`, distinct from the evaluation
sample being explained (Option B from the Phase 5.1 review). Using the
same small fixed sample for both roles — the original Phase 5
implementation — conflated two statistically different purposes: the
background is meant to represent the general pre-drift distribution
(and benefits from more points, for a stable expectation estimate),
while the evaluation sample is deliberately small and fixed so the same
exact points can be compared across checkpoints. Both samples are still
drawn exclusively from the pre-drift region (see
src/explanation/fixed_sample.py), so this change adds no leakage risk —
it only separates two roles that were previously collapsed into one.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from src.checkpoints.checkpoint_manager import CheckpointRecord
from src.explanation.explanation_result import ExplanationResult
from src.explanation.fixed_sample import FixedExplanationSample

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPUnavailableError(ImportError):
    """Raised when real SHAP execution is requested but the `shap`
    package is not installed. Never silently substitutes fabricated
    values in its place."""


class UnsupportedModelError(ValueError):
    """Raised when a detector_name has no known SHAP backend mapping."""


class MissingFeatureNamesError(ValueError):
    """Raised when feature_names is empty/None or doesn't match the
    sample's feature dimensionality."""


class InvalidSampleError(ValueError):
    """Raised for a structurally invalid or non-immutable sample."""


class MissingBackgroundSampleError(ValueError):
    """Raised when the permutation backend (Detector B) is used without
    a background_sample — per the Phase 5.1 decision, this is required,
    not optional, and never silently defaulted to the evaluation sample."""


BACKEND_TREE_EXPLAINER = "TreeExplainer"
BACKEND_PERMUTATION_EXPLAINER = "PermutationExplainer (shap.Explainer)"


def get_explanation_target(detector_name: str, model_snapshot) -> tuple[str, object]:
    """Pure dispatch logic — decides WHAT would be explained and with
    WHICH backend, without requiring shap to be installed.

    Returns (backend_label, target):
    - Detector A: target is the raw sklearn IsolationForest estimator
      (a fresh deepcopy each access — see DetectorAIsolationForest.raw_model).
    - Detector B: target is the detector's bound anomaly_score method
      (NOT the raw MLPRegressor), so SHAP explains the reconstruction-
      error anomaly score directly.
    """
    if detector_name == "detector_a":
        return BACKEND_TREE_EXPLAINER, model_snapshot.raw_model
    elif detector_name == "detector_b":
        return BACKEND_PERMUTATION_EXPLAINER, model_snapshot.anomaly_score
    else:
        raise UnsupportedModelError(
            f"No SHAP backend mapping for detector_name='{detector_name}' "
            "(expected 'detector_a' or 'detector_b')"
        )


def _validate_sample(sample: FixedExplanationSample, role: str) -> None:
    """Shared structural + immutability validation, used for both the
    evaluation sample and (when applicable) the background sample.
    `role` is only used to make error messages specific."""
    if sample.features.ndim != 2 or sample.features.shape[0] < 1:
        raise InvalidSampleError(f"Invalid {role} sample shape {sample.features.shape}")
    if sample.features.flags.writeable:
        raise InvalidSampleError(
            f"The {role} sample must be immutable (read-only) — got a writeable array."
        )


def _run_real_shap(backend_label: str, target, eval_X: np.ndarray, background_X: np.ndarray | None) -> np.ndarray:
    """Executes the ACTUAL shap library. Raises SHAPUnavailableError if
    shap is not installed — never returns a fabricated array."""
    if not SHAP_AVAILABLE:
        raise SHAPUnavailableError(
            "shap is not installed in this environment — cannot compute "
            "real SHAP values. This is a disclosed limitation, not a bug "
            "to work around with fabricated output."
        )
    import shap as _shap  # local import: only reached when SHAP_AVAILABLE

    if backend_label == BACKEND_TREE_EXPLAINER:
        # TreeExplainer's default (tree_path_dependent) mode does not
        # require a background sample; background_X is accepted here
        # only for interface symmetry and is intentionally unused.
        explainer = _shap.TreeExplainer(target)
        values = explainer.shap_values(eval_X)
    elif backend_label == BACKEND_PERMUTATION_EXPLAINER:
        if background_X is None:
            raise MissingBackgroundSampleError(
                "Permutation backend requires background_X, none was provided."
            )
        explainer = _shap.Explainer(target, background_X, algorithm="permutation")
        values = explainer(eval_X).values
    else:
        raise UnsupportedModelError(f"Unknown backend_label '{backend_label}'")

    return np.asarray(values)


def _normalize_shape(raw_values, expected_shape: tuple[int, int]) -> np.ndarray:
    """Normalizes whatever shape a given SHAP backend happens to
    return into a plain (n_samples, n_features) array. Raises
    ValueError (not a silent reshape-and-hope) if the data genuinely
    doesn't match after normalization."""
    values = raw_values
    if hasattr(values, "values"):  # shap.Explanation-like object
        values = values.values
    values = np.asarray(values)

    if values.shape != expected_shape:
        raise ValueError(
            f"SHAP output shape {values.shape} does not match expected "
            f"{expected_shape} after normalization"
        )
    return values


def generate_explanation(
    checkpoint_record: CheckpointRecord,
    sample: FixedExplanationSample,
    feature_names: Sequence[str],
    background_sample: FixedExplanationSample | None = None,
    config_snapshot: dict | None = None,
    raw_value_fn: Callable[[str, object, np.ndarray, np.ndarray | None], np.ndarray] | None = None,
) -> ExplanationResult:
    """Generate a normalized SHAP explanation for one checkpoint's
    stored model snapshot, using the fixed evaluation sample.

    `background_sample` (Phase 5.1): REQUIRED for Detector B (the
    permutation backend needs a background distribution, deliberately
    separate from the evaluation sample — see module docstring).
    Not required for Detector A (TreeExplainer needs none) and is
    ignored if supplied for it.

    `raw_value_fn`, if supplied, replaces the real shap call — this is
    a test-injection seam ONLY. Production code should never pass this;
    leaving it None routes to `_run_real_shap`, which raises
    SHAPUnavailableError honestly if shap isn't installed.
    """
    # --- feature name validation ---
    if not feature_names:
        raise MissingFeatureNamesError("feature_names must be a non-empty sequence")
    feature_names = tuple(feature_names)
    if len(feature_names) != sample.features.shape[1]:
        raise MissingFeatureNamesError(
            f"feature_names has {len(feature_names)} entries but sample has "
            f"{sample.features.shape[1]} feature columns"
        )

    # --- sample validation: structural + immutability ---
    _validate_sample(sample, role="evaluation")

    # --- backend dispatch (never requires shap itself) ---
    backend_label, target = get_explanation_target(
        checkpoint_record.detector_name, checkpoint_record.model_snapshot
    )

    # --- background sample: required for permutation, N/A for tree ---
    background_X = None
    background_signature = None
    if backend_label == BACKEND_PERMUTATION_EXPLAINER:
        if background_sample is None:
            raise MissingBackgroundSampleError(
                f"Detector '{checkpoint_record.detector_name}' uses the permutation "
                "backend, which requires an explicit background_sample "
                "(Phase 5.1 decision) — the evaluation sample is no longer "
                "reused as a background by default."
            )
        _validate_sample(background_sample, role="background")
        background_X = background_sample.features
        background_signature = background_sample.signature

    # --- actual computation (real shap, or an injected fake for tests) ---
    compute_fn = raw_value_fn or _run_real_shap
    raw_values = compute_fn(backend_label, target, sample.features, background_X)

    expected_shape = (sample.features.shape[0], len(feature_names))
    attributions = _normalize_shape(raw_values, expected_shape)

    if checkpoint_record.detector_name == "detector_a":
        model_type = type(checkpoint_record.model_snapshot.raw_model).__name__
    else:
        model_type = checkpoint_record.model_snapshot.model_class_name

    return ExplanationResult(
        detector_name=checkpoint_record.detector_name,
        condition=checkpoint_record.condition,
        seed=checkpoint_record.seed,
        checkpoint=checkpoint_record.checkpoint,
        feature_names=feature_names,
        attributions=attributions,
        sample_shape=tuple(sample.features.shape),
        explanation_sample_signature=sample.signature,
        background_sample_signature=background_signature,
        shap_backend=backend_label,
        model_type=model_type,
        config_snapshot=dict(config_snapshot or {}),
    )

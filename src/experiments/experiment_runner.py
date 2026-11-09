"""Experiment runner — orchestrates existing modules into one
reproducible, leakage-safe experiment run.

STATUS: Orchestration logic is IMPLEMENTED and unit-tested using
SYNTHETIC deterministic data and INJECTED FAKES for the drift monitor
and SHAP computation. This module does not implement
any new statistical/ML logic itself — every stage delegates to an
already-approved module (Phase 2-7); see the stage-by-stage comments
below for exactly which function is called at each step.

ONE call to `run_experiment` = ONE condition + ONE seed, but BOTH
detectors (matching the approved proposal's stage list, which lists
"Train Detector A" then "Train Detector B" as two stages within one
run). Internally, the two detectors are processed through fully
separate, independent sub-pipelines (`_run_single_detector`) — neither
shares state, a monitor, a checkpoint manager, or a live model object
with the other — preserving the Phase 12 "independent parallel
pipelines" architecture at the processing level, even though both
happen within one function call for convenience (they consume the same
input series and ground truth, which is why running them together is
useful — computing/injecting drift twice for two separate calls would
be wasteful and would also relax the guarantee that both detectors saw
the EXACT same drift event). See TestCrossDetectorIsolation in
tests/unit/test_experiment_runner.py for the regression proof that
detector_a and detector_b results never cross-contaminate.

Design decision — one config field serves two roles: `config.t_event`
is the raw-series drift-injection point for the "drift" condition, and
the matched-arbitrary-point for the "no_drift" control. This is
deliberate: reusing the same index (rather than drawing a second,
separate random point for the control) is what keeps the two
conditions' checkpoint timelines numerically aligned, exactly as the
approved proposal's control design requires. No second random draw is
introduced, and nothing about this fabricates a drift event for the
control — no_drift never touches drift_injection.py at all, and every
event_index recorded for the no_drift condition is labeled
"matched_arbitrary_point", never "adwin_detection" — enforced
structurally by CheckpointManager's UnlabeledEventIndexError (Phase 4)
and re-verified at this module's own level in
TestNoDriftNeverFabricatesDriftEvent.

Phase review — retraining slice (Decision 3): the base sliding
window is computed by REUSING RetrainTrigger.get_retrain_slice (not a
re-implemented formula), by passing the window's own index positions
as its "values" and reading back which index range it selected. An
ADDITIONAL clamp — `max(t_event_window, ...)` — is then applied on top,
strictly for the "drift" condition, to guarantee the retrain slice
never includes genuinely pre-event data; RetrainTrigger's own generic
formula only clamps to 0, since it has no concept of "which points are
pre-event" (that is an experiment-level concept, not something a
generic retrain-trigger utility could know). This is a deliberate,
reported EXTENSION of the approved sliding-window concept, not a
silent change to it. The slice is INCLUSIVE of both `retrain_slice_start`
and `t_detect` (Python slice `[start:t_detect+1]`). The "no_drift"
condition mirrors the same base formula directly (documented at the
point of use) rather than instantiating a RetrainTrigger, because
RetrainTrigger.confirm_retrain requires a prior detection event, which
the control condition correctly never has.

Phase review — t_retrain = t_detect + 1 (Decision 4): this is a
SYNTHETIC ORCHESTRATION CONVENTION for this runner, not a scientific
claim about real detection-to-retrain latency. It exists only to keep
t_detect and t_retrain numerically distinct (per the Phase 4 three-
timestamp discipline) while modeling "retrain immediately after
detection" from the approved architecture. If detection happens at the
last available index, t_retrain would fall outside the data entirely —
this is NOT silently handled by moving the event; it raises
InsufficientPostRetrainDataError explicitly (see
TestDetectionAtFinalValidIndex).

Phase review — post-retrain evaluation window (Decision 5): the
approved proposal (Phase 13 §13) describes post-retrain windows
"matched to W where series length allows" — i.e., Option B (a
comparable fixed size), not Option A (all remaining data) as the
target design. `config.post_retrain_window_size` now exists for this
(TBD/None by default). When None, the runner falls back to using all
remaining data (Option A), clearly labeled as a fallback, not the
preferred design — this keeps existing behavior working without
forcing a numeric choice before real NAB inspection. Whichever policy
is used, the EXACT window bounds and sample count actually used for
every checkpoint (not just post-retrain) are now recorded in
`DetectorRunResult.window_bounds`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.checkpoints.checkpoint_manager import CheckpointManager, VALID_CHECKPOINTS
from src.data.drift_injection import inject_offset_scale_drift
from src.data.preprocessing import N_FEATURES, FEATURE_NAMES, PrefixFittedScaler, extract_window_features
from src.data.validation import validate_series
from src.detectors.detector_a_isolation_forest import DetectorAIsolationForest
from src.detectors.detector_b_autoencoder import DetectorBAutoencoder
from src.experiments.experiment_config import ExperimentConfig
from src.explanation.fixed_sample import build_background_and_evaluation_samples
from src.explanation.shap_explainer import generate_explanation
from src.metrics.detection_evaluation import DetectionEvaluationResult, evaluate_detection, threshold_scores
from src.metrics.explanation_stability import StabilityMetricsResult, compute_stability_metrics
from src.monitoring.drift_monitor import AdwinDriftMonitor, PageHinkleyDriftMonitor, RiverUnavailableError
from src.monitoring.retrain_trigger import RetrainTrigger

DETECTOR_NAMES = ("detector_a", "detector_b")


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class MissingDataError(ValueError):
    """Raised for missing/mismatched series or ground-truth labels."""


class TooShortPreDriftRegionError(ValueError):
    """Raised when the series is too short for the configured
    t_event/window_size to leave a usable pre- and post-event region."""


class InvalidFeatureDimensionError(ValueError):
    """Raised if windowing ever produces a feature width other than
    the fixed N_FEATURES."""


class InsufficientPostRetrainDataError(ValueError):
    """Raised when there is no data left after the retrain point to
    form a (non-empty) post_retrain evaluation window."""


class DriftNeverDetectedError(RuntimeError):
    """Raised when the primary (ADWIN) monitor never detects drift
    before the data runs out — a real, reportable experimental outcome,
    surfaced explicitly rather than fabricating a checkpoint that never
    happened."""


class UnsupportedDetectorError(ValueError):
    """Raised for a detector_name outside DETECTOR_NAMES."""


def _first_window_touched_by_event(t_event: int, window_size: int, stride: int) -> int:
    """Index of the first windowed-feature row whose raw span includes
    `t_event`. Rows BEFORE this index are guaranteed composed entirely
    of raw points with index < t_event."""
    return (t_event - window_size) // stride + 1


def _align_labels_to_windows(y_true_raw: np.ndarray, window_size: int, stride: int, n_windows: int) -> np.ndarray:
    """Each window's label = the raw ground-truth label at the window's
    LAST raw index (the same index convention used to timestamp a
    window throughout this project)."""
    labels = np.empty(n_windows, dtype=int)
    for i in range(n_windows):
        last_raw_index = i * stride + window_size - 1
        labels[i] = int(y_true_raw[last_raw_index])
    return labels


def _make_detector(detector_name: str, seed: int, config: ExperimentConfig):
    if detector_name == "detector_a":
        return DetectorAIsolationForest(random_state=seed)
    elif detector_name == "detector_b":
        return DetectorBAutoencoder(
            random_state=seed,
            hidden_layer_sizes=config.detector_b_hidden_layer_sizes,
            max_iter=config.detector_b_max_iter,
        )
    else:
        raise UnsupportedDetectorError(f"Unknown detector_name '{detector_name}'")


@dataclass(frozen=True)
class SecondaryMonitorResult:
    """Phase 8.1 Decision 1: pairs status and detection_index together
    so the two can never be read independently and misinterpreted.

    `status` is exactly one of:
      - "not_run"    — no_drift condition; secondary monitoring never
                        applies (there is no drift-detection concept
                        for the control at all).
      - "ran"         — river WAS available and Page-Hinkley genuinely
                        executed; `detection_index` is either a real
                        detected index, or None meaning "ran but did
                        not detect within the scanned range" — itself a
                        real, valid finding, not a missing one.
      - "unavailable" — river was NOT available; Page-Hinkley never
                         executed at all; `detection_index` is always
                         None here, but for a DIFFERENT reason than the
                         "ran, no detection" case above — never
                         conflate the two.
    """

    status: str
    detection_index: int | None

    def to_dict(self) -> dict:
        return {"status": self.status, "detection_index": self.detection_index}


@dataclass
class DetectorRunResult:
    detector_name: str
    explanations: dict                     # checkpoint name -> ExplanationResult
    stability_vs_post_drift: StabilityMetricsResult
    stability_vs_post_retrain: StabilityMetricsResult
    detection: dict                        # checkpoint name -> DetectionEvaluationResult
    checkpoint_events: dict                # checkpoint name -> {"event_index": int|None, "index_source": str|None}
    window_bounds: dict                    # checkpoint name -> {"start": int, "end": int, "n_samples": int}
    t_event_window: int
    t_detect: int | None                   # None for no_drift
    t_retrain: int
    secondary_monitor: SecondaryMonitorResult

    def to_metadata_dict(self) -> dict:
        return {
            "detector_name": self.detector_name,
            "t_event_window": self.t_event_window,
            "t_detect": self.t_detect,
            "t_retrain": self.t_retrain,
            "checkpoint_events": self.checkpoint_events,
            "window_bounds": self.window_bounds,
            "secondary_monitor": self.secondary_monitor.to_dict(),
            "explanations": {k: v.to_metadata_dict() for k, v in self.explanations.items()},
            "stability_vs_post_drift": self.stability_vs_post_drift.to_metadata_dict(),
            "stability_vs_post_retrain": self.stability_vs_post_retrain.to_metadata_dict(),
            "detection": {k: v.to_metadata_dict() for k, v in self.detection.items()},
        }


@dataclass
class ExperimentResult:
    condition: str
    seed: int
    config: ExperimentConfig
    detector_a: DetectorRunResult
    detector_b: DetectorRunResult
    background_sample: object
    evaluation_sample: object

    def to_metadata_dict(self) -> dict:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "config": self.config.to_metadata_dict(),
            "background_sample_signature": self.background_sample.signature,
            "evaluation_sample_signature": self.evaluation_sample.signature,
            "detector_a": self.detector_a.to_metadata_dict(),
            "detector_b": self.detector_b.to_metadata_dict(),
        }


def _scan_secondary_monitor(monitor, scores: np.ndarray, start_index: int) -> int | None:
    """Best-effort Page-Hinkley scan over the SAME score range the
    primary monitor searched. Returns the first detection index, or
    None if never detected within that range (a real, valid "ran but
    did not detect" finding — see SecondaryMonitorResult docstring)."""
    for offset, score in enumerate(scores):
        idx = start_index + offset
        if monitor.update(score, idx):
            return idx
    return None


def _retrain_slice_bounds_drift(trigger: RetrainTrigger, t_detect: int, t_event_window: int) -> tuple[int, int]:
    """Phase 8.1 Decision 3: base window REUSED from
    RetrainTrigger.get_retrain_slice (by feeding it index positions as
    its 'values', so its own formula determines the base start), then
    an additional experiment-level clamp is applied so the slice never
    dips before t_event_window. Returns (start, end_inclusive)."""
    index_positions = list(range(t_detect + 1))
    base_window = trigger.get_retrain_slice(index_positions, up_to_index=t_detect)
    base_start = base_window[0]
    clamped_start = max(t_event_window, base_start)
    return clamped_start, t_detect  # inclusive end


def _run_single_detector(
    detector_name: str,
    scaled_features: np.ndarray,
    y_true_windowed: np.ndarray,
    config: ExperimentConfig,
    t_event_window: int,
    background_sample,
    evaluation_sample,
    primary_monitor_factory: Callable[[], object] | None,
    secondary_monitor_factory: Callable[[], object] | None,
    shap_compute_fn,
) -> DetectorRunResult:
    detector = _make_detector(detector_name, config.seed, config)
    detector.fit(scaled_features[:t_event_window])

    manager = CheckpointManager(detector_name, config.condition, config.seed, explanation_sample=evaluation_sample)
    manager.capture("pre_drift", detector)

    secondary_result = SecondaryMonitorResult(status="not_run", detection_index=None)

    if config.condition == "drift":
        monitor_factory = primary_monitor_factory or AdwinDriftMonitor
        primary_monitor = monitor_factory()
        trigger = RetrainTrigger(monitor=primary_monitor, retrain_window_size=config.retrain_window_size)

        scores_seen = []
        t_detect = None
        for idx in range(t_event_window, len(scaled_features)):
            score = float(detector.anomaly_score(scaled_features[idx:idx + 1])[0])
            scores_seen.append(score)
            if trigger.process(score, idx):
                t_detect = idx
                break

        if t_detect is None:
            raise DriftNeverDetectedError(
                f"{detector_name}: primary monitor never detected drift within the "
                f"available post-event data (scanned indices {t_event_window}..{len(scaled_features) - 1})"
            )

        # Secondary (Page-Hinkley) — best-effort, never blocks the run.
        try:
            ph_factory = secondary_monitor_factory or PageHinkleyDriftMonitor
            ph_monitor = ph_factory()
            ph_index = _scan_secondary_monitor(ph_monitor, np.array(scores_seen), t_event_window)
            secondary_result = SecondaryMonitorResult(status="ran", detection_index=ph_index)
        except RiverUnavailableError:
            secondary_result = SecondaryMonitorResult(status="unavailable", detection_index=None)

        t_retrain = t_detect + 1
        if t_retrain >= len(scaled_features):
            raise InsufficientPostRetrainDataError(
                f"{detector_name}: detection at index {t_detect} leaves no data for a "
                f"post_retrain window (t_retrain={t_retrain} >= available length "
                f"{len(scaled_features)}) — the event is NOT silently moved; this is raised "
                f"explicitly."
            )

        retrain_start, retrain_end_incl = _retrain_slice_bounds_drift(trigger, t_detect, t_event_window)

        manager.capture(
            "post_drift_pre_retrain", detector, event_index=t_detect,
            extra_metadata={"index_source": "adwin_detection"},
        )
        detector.fit(scaled_features[retrain_start:retrain_end_incl + 1])
        manager.capture(
            "post_retrain", detector, event_index=t_retrain,
            extra_metadata={"index_source": "retrain_trigger"},
        )
        mid_start, mid_end_incl = t_event_window, t_detect
        checkpoint_events = {
            "pre_drift": {"event_index": None, "index_source": None},
            "post_drift_pre_retrain": {"event_index": t_detect, "index_source": "adwin_detection"},
            "post_retrain": {"event_index": t_retrain, "index_source": "retrain_trigger"},
        }

    else:  # no_drift control — Phase 8.1 Decision 2: never touches a
        # monitor, never uses "adwin_detection" as a label, always
        # "matched_arbitrary_point".
        t_detect = None
        mid_index = t_event_window
        t_retrain = t_event_window + 1
        if t_retrain >= len(scaled_features):
            raise InsufficientPostRetrainDataError(
                f"{detector_name}: no data remains after the matched point for a "
                f"post_retrain window"
            )
        # Mirrors RetrainTrigger.get_retrain_slice's own base formula
        # directly (max(0, up_to_index+1-window)) rather than
        # instantiating RetrainTrigger, since there is no detection
        # event to satisfy its confirm_retrain contract — documented
        # here, not silently duplicated.
        retrain_start = max(0, t_retrain - config.retrain_window_size)
        retrain_end_incl = t_retrain - 1

        manager.capture(
            "post_drift_pre_retrain", detector, event_index=mid_index,
            extra_metadata={"index_source": "matched_arbitrary_point"},
        )
        detector.fit(scaled_features[retrain_start:retrain_end_incl + 1])
        manager.capture(
            "post_retrain", detector, event_index=t_retrain,
            extra_metadata={"index_source": "matched_arbitrary_point"},
        )
        mid_start, mid_end_incl = t_event_window, mid_index
        checkpoint_events = {
            "pre_drift": {"event_index": None, "index_source": None},
            "post_drift_pre_retrain": {"event_index": mid_index, "index_source": "matched_arbitrary_point"},
            "post_retrain": {"event_index": t_retrain, "index_source": "matched_arbitrary_point"},
        }

    manager.require_complete()

    # --- SHAP explanations from stored (isolated) checkpoint snapshots ---
    background = background_sample if detector_name == "detector_b" else None
    explanations = {}
    for checkpoint in VALID_CHECKPOINTS:
        record = manager.get(checkpoint)
        explanations[checkpoint] = generate_explanation(
            record, evaluation_sample, FEATURE_NAMES,
            background_sample=background, raw_value_fn=shap_compute_fn,
        )

    # --- stability metrics (approved comparisons only) ---
    # Real-execution finding (Phase 10.2, via laptop run with real
    # shap installed): a real SHAP attribution vector for one sample
    # CAN legitimately be all-zero (zero marginal contribution from
    # every feature for that specific point) — cosine similarity is
    # mathematically undefined for a zero vector, and the strict
    # default policy (on_invalid="raise") correctly raises
    # ZeroVectorError for that single comparison per Phase 6's design.
    # But raising there would crash the ENTIRE experiment over one
    # degenerate sample, which is worse than the alternative: this
    # runner explicitly opts into "mark_nan" mode, so that one invalid
    # per-sample comparison is recorded as NaN (excluded from the mean,
    # counted in n_invalid — never silently averaged in as if valid)
    # while the rest of the experiment completes. This is a deliberate,
    # reported choice at the runner/orchestration level — the strict
    # "raise" default remains correct and unchanged for direct callers
    # of src.metrics.explanation_stability itself (see its own tests).
    stability_vs_post_drift = compute_stability_metrics(
        explanations["pre_drift"], explanations["post_drift_pre_retrain"], k=config.k_jaccard,
        on_invalid_cosine="mark_nan", on_invalid_jaccard="mark_nan",
    )
    stability_vs_post_retrain = compute_stability_metrics(
        explanations["pre_drift"], explanations["post_retrain"], k=config.k_jaccard,
        on_invalid_cosine="mark_nan", on_invalid_jaccard="mark_nan",
    )

    # --- detection metrics per checkpoint, using ISOLATED snapshots only ---
    # Phase 8.1 Decision 5: post_retrain window uses post_retrain_window_size
    # when configured (Option B, matching the approved proposal's design
    # intent), falling back to "all remaining data" (Option A) when not
    # yet configured — documented as a fallback, not the preferred design.
    post_retrain_end = len(scaled_features)
    if config.post_retrain_window_size is not None:
        post_retrain_end = min(t_retrain + config.post_retrain_window_size, len(scaled_features))
        if post_retrain_end <= t_retrain:
            raise InsufficientPostRetrainDataError(
                f"{detector_name}: post_retrain_window_size={config.post_retrain_window_size} "
                f"leaves an empty post-retrain window"
            )

    windows = {
        "pre_drift": (0, t_event_window),
        "post_drift_pre_retrain": (mid_start, mid_end_incl + 1),
        "post_retrain": (t_retrain, post_retrain_end),
    }
    detection = {}
    window_bounds = {}
    for checkpoint, (start, end) in windows.items():
        record = manager.get(checkpoint)
        snapshot = record.model_snapshot
        window_features = scaled_features[start:end]
        window_labels = y_true_windowed[start:end]
        scores = snapshot.anomaly_score(window_features)
        preds = threshold_scores(scores, threshold=config.detection_threshold)
        detection[checkpoint] = evaluate_detection(
            window_labels, preds, detector_name=detector_name, condition=config.condition,
            seed=config.seed, checkpoint=checkpoint, threshold=config.detection_threshold,
        )
        window_bounds[checkpoint] = {"start": start, "end": end, "n_samples": end - start}

    return DetectorRunResult(
        detector_name=detector_name,
        explanations=explanations,
        stability_vs_post_drift=stability_vs_post_drift,
        stability_vs_post_retrain=stability_vs_post_retrain,
        detection=detection,
        checkpoint_events=checkpoint_events,
        window_bounds=window_bounds,
        t_event_window=t_event_window,
        t_detect=t_detect,
        t_retrain=t_retrain,
        secondary_monitor=secondary_result,
    )


def run_experiment(
    series: pd.Series,
    y_true_raw: np.ndarray,
    config: ExperimentConfig,
    primary_monitor_factory: Callable[[], object] | None = None,
    secondary_monitor_factory: Callable[[], object] | None = None,
    shap_compute_fn=None,
) -> ExperimentResult:
    """Run one complete experiment (one condition, one seed, both
    detectors). `primary_monitor_factory`/`secondary_monitor_factory`/
    `shap_compute_fn` are test-injection seams ONLY — leaving them None
    routes to the real ADWIN/Page-Hinkley/shap paths, which raise their
    own documented errors honestly in this environment.
    """
    if series is None or y_true_raw is None:
        raise MissingDataError("series and y_true_raw are both required")
    if len(series) != len(y_true_raw):
        raise MissingDataError(
            f"series (len {len(series)}) and y_true_raw (len {len(y_true_raw)}) must match"
        )
    y_true_raw = np.asarray(y_true_raw)
    if not np.all(np.isin(y_true_raw.astype(int), [0, 1])):
        raise MissingDataError("y_true_raw must be binary (0/1)")

    minimum_length_required = config.t_event + config.window_size
    report = validate_series(series, minimum_length_required=minimum_length_required)
    if not report.passes_minimum_length:
        raise TooShortPreDriftRegionError(
            f"series length {report.length} insufficient for t_event={config.t_event}, "
            f"window_size={config.window_size} (need >= {minimum_length_required})"
        )
    if report.n_missing_values > 0:
        raise MissingDataError(f"series contains {report.n_missing_values} missing values")

    if config.condition == "drift":
        injected_series, _record = inject_offset_scale_drift(
            series, t_inject=config.t_event, offset=config.drift_offset,
            scale_multiplier=config.drift_scale_multiplier, random_state=config.seed,
        )
    else:
        injected_series = series

    values = injected_series.to_numpy(dtype=float)
    all_features = extract_window_features(values, window_size=config.window_size, stride=config.stride)
    if all_features.shape[1] != N_FEATURES:
        raise InvalidFeatureDimensionError(
            f"Expected {N_FEATURES} features, got {all_features.shape[1]}"
        )

    t_event_window = _first_window_touched_by_event(config.t_event, config.window_size, config.stride)
    if t_event_window < 1 or t_event_window >= len(all_features) - 1:
        raise TooShortPreDriftRegionError(
            f"t_event_window={t_event_window} leaves no usable pre/post-event windows "
            f"(total windows={len(all_features)})"
        )

    y_true_windowed = _align_labels_to_windows(
        y_true_raw, config.window_size, config.stride, n_windows=len(all_features)
    )

    scaler = PrefixFittedScaler().fit(all_features[:t_event_window])
    scaled_features = scaler.transform(all_features)

    background_sample, evaluation_sample = build_background_and_evaluation_samples(
        scaled_features, t_inject=t_event_window,
        n_background=config.background_sample_size, n_evaluation=config.evaluation_sample_size,
        background_random_state=config.background_random_state,
        evaluation_random_state=config.evaluation_random_state,
    )

    results = {}
    for detector_name in DETECTOR_NAMES:
        results[detector_name] = _run_single_detector(
            detector_name, scaled_features, y_true_windowed, config, t_event_window,
            background_sample, evaluation_sample,
            primary_monitor_factory, secondary_monitor_factory, shap_compute_fn,
        )

    return ExperimentResult(
        condition=config.condition, seed=config.seed, config=config,
        detector_a=results["detector_a"], detector_b=results["detector_b"],
        background_sample=background_sample, evaluation_sample=evaluation_sample,
    )

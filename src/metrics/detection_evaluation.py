"""Detection-performance evaluation — confusion matrix, precision/
recall/F1, and episode-based zero-delay detection.

STATUS: IMPLEMENTED, unit-tested against synthetic labels/predictions.
No real NAB ground truth or detector predictions have been evaluated
anywhere in this project yet — everything here is validated against
hand-constructed binary sequences with known, hand-computable answers.

This module is deliberately independent of the experiment runner (not
yet implemented) and of the checkpoint/SHAP/stability modules — it only
consumes plain ground-truth/prediction arrays and returns plain,
JSON-serializable results.

Zero-delay definition (per the approved proposal, matching the CAIN
2024 anchor study's strict delay=0 evaluation): for each ground-truth
anomaly EPISODE (a contiguous run of anomalous points), delay is
measured from the episode's own start index to the first predicted-
anomaly index found WITHIN that episode's span [start, end] inclusive.
If no prediction falls inside that span, the episode is MISSED — a
prediction occurring only after the episode has ended is not searched
for and does not retroactively "detect" the episode. Zero delay means
delay == 0, i.e. the detector flagged the episode's very first point.
Predictions before an episode start are never counted toward that
episode (they would either be false positives or belong to an earlier
episode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class LengthMismatchError(ValueError):
    """Raised when y_true and y_pred have different lengths."""


class EmptyInputError(ValueError):
    """Raised when y_true or y_pred is empty."""


class InvalidLabelError(ValueError):
    """Raised when a label/prediction value is not strictly binary
    (0/1 or False/True) after casting."""


class InvalidThresholdComparisonError(ValueError):
    """Raised for an unrecognized threshold comparison operator."""


class InvalidEpisodeError(ValueError):
    """Raised for a structurally invalid episode (e.g. end < start, or
    indices outside the sequence length)."""


def _validate_binary_array(arr: np.ndarray, label: str) -> np.ndarray:
    if len(arr) == 0:
        raise EmptyInputError(f"{label} must not be empty")
    arr = np.asarray(arr)
    try:
        as_int = arr.astype(int)
    except (TypeError, ValueError) as exc:
        raise InvalidLabelError(f"{label} contains non-numeric values") from exc
    if not np.array_equal(as_int, arr.astype(float)) or not np.all(np.isin(as_int, [0, 1])):
        raise InvalidLabelError(
            f"{label} must contain only binary values (0/1 or False/True); "
            f"got values outside {{0, 1}}"
        )
    return as_int


# ----------------------------------------------------------------------
# Score-to-prediction thresholding
# ----------------------------------------------------------------------


def threshold_scores(scores: np.ndarray, threshold: float, comparison: str = ">=") -> np.ndarray:
    """Convert continuous anomaly scores into binary predictions.

    `comparison`: ">=" (default) means a score exactly equal to the
    threshold is classified as anomalous (1); ">" excludes the boundary
    value. This is a deliberate, documented, deterministic choice — no
    automatic threshold tuning is implemented in this phase.
    """
    scores = np.asarray(scores, dtype=float)
    if comparison == ">=":
        return (scores >= threshold).astype(int)
    elif comparison == ">":
        return (scores > threshold).astype(int)
    else:
        raise InvalidThresholdComparisonError(
            f"comparison must be '>=' or '>', got '{comparison}'"
        )


# ----------------------------------------------------------------------
# Confusion matrix + precision/recall/F1
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionMatrixCounts:
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return {"tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn}


def confusion_matrix_counts(y_true, y_pred) -> ConfusionMatrixCounts:
    """Confusion-matrix counts for binary y_true/y_pred.

    Raises LengthMismatchError, EmptyInputError, or InvalidLabelError
    for invalid input — never silently coerces or truncates.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise LengthMismatchError(
            f"y_true (len {len(y_true)}) and y_pred (len {len(y_pred)}) must have equal length"
        )

    y_true = _validate_binary_array(y_true, "y_true")
    y_pred = _validate_binary_array(y_pred, "y_pred")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return ConfusionMatrixCounts(tp=tp, tn=tn, fp=fp, fn=fn)


@dataclass(frozen=True)
class MetricValue:
    """A metric value paired with an explicit `undefined` flag.

    Policy (consistent, documented, chosen over raising or returning
    NaN/Inf): when a metric's denominator is zero, `value` is 0.0 and
    `undefined` is True. This means a caller can always safely use
    `.value` numerically (never NaN/Inf to accidentally propagate), but
    can also check `.undefined` to avoid MISINTERPRETING a 0.0 that
    actually means "no positive class present" (e.g. recall on a
    no-drift control with zero true anomalies) as "the detector missed
    everything"."""

    value: float
    undefined: bool

    def to_dict(self) -> dict:
        return {"value": self.value, "undefined": self.undefined}


def precision_recall_f1(counts: ConfusionMatrixCounts) -> tuple[MetricValue, MetricValue, MetricValue]:
    """Precision, recall, F1 from confusion-matrix counts, using the
    zero-denominator policy documented on MetricValue."""
    pred_positive = counts.tp + counts.fp
    actual_positive = counts.tp + counts.fn

    if pred_positive == 0:
        precision = MetricValue(0.0, undefined=True)
    else:
        precision = MetricValue(counts.tp / pred_positive, undefined=False)

    if actual_positive == 0:
        recall = MetricValue(0.0, undefined=True)
    else:
        recall = MetricValue(counts.tp / actual_positive, undefined=False)

    denom = precision.value + recall.value
    if denom == 0:
        f1 = MetricValue(0.0, undefined=True)
    else:
        f1 = MetricValue(2 * precision.value * recall.value / denom, undefined=False)

    return precision, recall, f1


# ----------------------------------------------------------------------
# Episodes and zero-delay detection
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """A contiguous ground-truth anomaly episode. `end` is INCLUSIVE."""

    start: int
    end: int

    def __post_init__(self):
        if self.end < self.start:
            raise InvalidEpisodeError(f"Episode end ({self.end}) must be >= start ({self.start})")


def extract_episodes_from_binary(y_true) -> tuple[Episode, ...]:
    """Extract contiguous anomaly episodes from a binary ground-truth
    sequence. An all-zero sequence (e.g. a genuine no-drift control)
    returns an EMPTY tuple — no episode is ever fabricated."""
    y_true = _validate_binary_array(np.asarray(y_true), "y_true")

    episodes = []
    in_episode = False
    start = None
    for i, v in enumerate(y_true):
        if v == 1 and not in_episode:
            in_episode = True
            start = i
        elif v == 0 and in_episode:
            episodes.append(Episode(start=start, end=i - 1))
            in_episode = False
    if in_episode:  # sequence ends while still inside an episode
        episodes.append(Episode(start=start, end=len(y_true) - 1))

    return tuple(episodes)


@dataclass(frozen=True)
class EpisodeDetectionResult:
    episode: Episode
    detected: bool
    detected_index: int | None   # None iff not detected (missed)
    delay: int | None            # None iff missed; >= 0 otherwise
    zero_delay: bool             # True iff delay == 0

    def to_dict(self) -> dict:
        return {
            "episode": {"start": self.episode.start, "end": self.episode.end},
            "detected": self.detected,
            "detected_index": self.detected_index,
            "delay": self.delay,
            "zero_delay": self.zero_delay,
        }


def evaluate_episode(episode: Episode, y_pred: np.ndarray) -> EpisodeDetectionResult:
    """Evaluate ONE episode against a binary prediction sequence.
    Searches for the first predicted-positive index WITHIN
    [episode.start, episode.end] only — per the module's documented
    zero-delay definition, a prediction outside this window is never
    attributed to this episode (never fabricates a detection)."""
    y_pred = np.asarray(y_pred)
    window = y_pred[episode.start:episode.end + 1]
    positive_offsets = np.flatnonzero(window == 1)

    if len(positive_offsets) == 0:
        return EpisodeDetectionResult(
            episode=episode, detected=False, detected_index=None, delay=None, zero_delay=False
        )

    first_offset = int(positive_offsets[0])
    detected_index = episode.start + first_offset
    delay = detected_index - episode.start
    return EpisodeDetectionResult(
        episode=episode, detected=True, detected_index=detected_index,
        delay=delay, zero_delay=(delay == 0),
    )


@dataclass(frozen=True)
class DelayAggregateResult:
    episode_results: tuple[EpisodeDetectionResult, ...]
    n_episodes: int
    n_missed: int
    n_zero_delay: int
    mean_delay_over_detected: float | None   # None iff there are zero detected episodes to average

    def to_dict(self) -> dict:
        return {
            "episode_results": [e.to_dict() for e in self.episode_results],
            "n_episodes": self.n_episodes,
            "n_missed": self.n_missed,
            "n_zero_delay": self.n_zero_delay,
            "mean_delay_over_detected": self.mean_delay_over_detected,
        }


def evaluate_episodes(episodes: Sequence[Episode], y_pred: np.ndarray) -> DelayAggregateResult:
    """Evaluate MULTIPLE episodes. An empty `episodes` sequence (e.g.
    genuine no-drift control) returns a result with n_episodes=0 and
    mean_delay_over_detected=None — never a fabricated value."""
    results = tuple(evaluate_episode(ep, y_pred) for ep in episodes)
    n_missed = sum(1 for r in results if not r.detected)
    n_zero_delay = sum(1 for r in results if r.zero_delay)
    detected_delays = [r.delay for r in results if r.detected]
    mean_delay = float(np.mean(detected_delays)) if detected_delays else None

    return DelayAggregateResult(
        episode_results=results, n_episodes=len(results), n_missed=n_missed,
        n_zero_delay=n_zero_delay, mean_delay_over_detected=mean_delay,
    )


# ----------------------------------------------------------------------
# Top-level evaluation combining everything, with metadata
# ----------------------------------------------------------------------


@dataclass
class DetectionEvaluationResult:
    confusion: ConfusionMatrixCounts
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    delay: DelayAggregateResult
    detector_name: str
    condition: str
    seed: int
    checkpoint: str
    n_samples: int
    threshold: float | None = None
    aggregation_method: str = "mean_over_detected_episodes"
    config_snapshot: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        return {
            "detector_name": self.detector_name,
            "condition": self.condition,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "n_samples": self.n_samples,
            "n_episodes": self.delay.n_episodes,
            "threshold": self.threshold,
            "aggregation_method": self.aggregation_method,
            "confusion_matrix": self.confusion.to_dict(),
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "f1": self.f1.to_dict(),
            "delay": self.delay.to_dict(),
            "config_snapshot": self.config_snapshot,
        }


def evaluate_detection(
    y_true,
    y_pred,
    detector_name: str,
    condition: str,
    seed: int,
    checkpoint: str,
    threshold: float | None = None,
    config_snapshot: dict | None = None,
) -> DetectionEvaluationResult:
    """Full detection-performance evaluation: confusion matrix,
    precision/recall/F1, and episode-based zero-delay/delay analysis —
    combined into one result with metadata. The no-drift control is
    handled naturally: if y_true is all-zero, episodes will be empty
    and recall will be flagged `undefined=True` rather than silently
    reported as a plain 0.0 detection failure.
    """
    counts = confusion_matrix_counts(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(counts)

    episodes = extract_episodes_from_binary(y_true)
    delay_result = evaluate_episodes(episodes, np.asarray(y_pred))

    return DetectionEvaluationResult(
        confusion=counts, precision=precision, recall=recall, f1=f1, delay=delay_result,
        detector_name=detector_name, condition=condition, seed=seed, checkpoint=checkpoint,
        n_samples=len(np.asarray(y_true)), threshold=threshold,
        config_snapshot=dict(config_snapshot or {}),
    )

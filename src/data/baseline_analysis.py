"""Real NAB baseline analysis and parameter-locking logic.

STATUS: IMPLEMENTED. All logic here is unit-tested against SYNTHETIC
fixture data with hand-computable answers (see
tests/unit/test_baseline_analysis.py). 

Reuses existing modules rather than duplicating their logic:
src.data.loader.load_nab_series for loading,
src.experiments.experiment_runner._first_window_touched_by_event for
the exact windowed-index boundary math (imported directly rather than
re-derived, since re-deriving it would risk a silent divergence from
the runner's own, already-tested arithmetic).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.experiments.experiment_runner import _first_window_touched_by_event

PASS = "PASS"
WARN = "WARN"
BLOCKED = "BLOCKED"
TBD = "TBD"
LOCKED = "LOCKED"

DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


# ----------------------------------------------------------------------
# Baseline statistics — real pre-event region only
# ----------------------------------------------------------------------


@dataclass
class BaselineStatistics:
    baseline_start_index: int
    baseline_end_index_exclusive: int      # Python-slice convention: [start, end)
    count: int
    mean: float
    std: float
    variance: float
    minimum: float
    maximum: float
    median: float
    quantiles: dict                        # {"q10": ..., "q25": ..., ...}
    coefficient_of_variation: float | None  # None if mean == 0 (mathematically undefined)
    n_labeled_anomalies_in_baseline: int
    anomaly_treatment_note: str

    def to_dict(self) -> dict:
        return {
            "baseline_start_index": self.baseline_start_index,
            "baseline_end_index_exclusive": self.baseline_end_index_exclusive,
            "count": self.count, "mean": self.mean, "std": self.std, "variance": self.variance,
            "minimum": self.minimum, "maximum": self.maximum, "median": self.median,
            "quantiles": self.quantiles, "coefficient_of_variation": self.coefficient_of_variation,
            "n_labeled_anomalies_in_baseline": self.n_labeled_anomalies_in_baseline,
            "anomaly_treatment_note": self.anomaly_treatment_note,
        }


def compute_baseline_statistics(
    series: pd.Series,
    t_event: int,
    labeled_anomaly_windows: list | None = None,
    quantiles: tuple = DEFAULT_QUANTILES,
) -> BaselineStatistics:
    """Computes real descriptive statistics on series[:t_event] ONLY —
    never touches series[t_event:]. If any labeled anomaly window
    overlaps the baseline region (by timestamp), this is REPORTED, not
    silently excluded or silently included — the anomaly points remain
    IN the statistics (matching the approved proposal: baseline
    statistics characterize the actual pre-event data as it is; an
    intern-scale project does not get to quietly clean the data to make
    statistics look nicer), with the overlap explicitly flagged so a
    human can review whether that's acceptable for the final choice.
    """
    if not (0 < t_event <= len(series)):
        raise ValueError(f"t_event must satisfy 0 < t_event <= {len(series)}; got {t_event}")

    baseline = series.iloc[:t_event]
    values = baseline.to_numpy(dtype=float)

    n_overlapping = 0
    if labeled_anomaly_windows:
        baseline_start_ts = series.index[0]
        baseline_end_ts = series.index[t_event - 1]
        for w_start, w_end in labeled_anomaly_windows:
            w_start_ts, w_end_ts = pd.Timestamp(w_start), pd.Timestamp(w_end)
            if w_start_ts <= baseline_end_ts and w_end_ts >= baseline_start_ts:
                n_overlapping += 1

    if n_overlapping > 0:
        note = (
            f"{n_overlapping} labeled anomaly window(s) overlap the baseline region "
            f"(index [0, {t_event})). Anomaly points are NOT excluded from these "
            f"statistics — they remain part of the real pre-event data as recorded. "
            f"This should be reviewed before finalizing t_event."
        )
    else:
        note = "No labeled anomaly windows overlap the baseline region."

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = (std / mean) if mean != 0 else None

    quantile_dict = {
        f"q{int(q * 100)}": float(np.quantile(values, q)) for q in quantiles
    }

    return BaselineStatistics(
        baseline_start_index=0, baseline_end_index_exclusive=t_event,
        count=len(values), mean=mean, std=std, variance=float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
        minimum=float(np.min(values)), maximum=float(np.max(values)), median=float(np.median(values)),
        quantiles=quantile_dict, coefficient_of_variation=cv,
        n_labeled_anomalies_in_baseline=n_overlapping, anomaly_treatment_note=note,
    )


# ----------------------------------------------------------------------
# Event-index validity and anomaly-window overlap
# ----------------------------------------------------------------------


@dataclass
class EventIndexValidityResult:
    status: str
    t_event: int
    row_count: int
    reason: str

    def to_dict(self) -> dict:
        return {"status": self.status, "t_event": self.t_event, "row_count": self.row_count, "reason": self.reason}


def check_event_index_validity(t_event: int, row_count: int) -> EventIndexValidityResult:
    if t_event <= 0:
        return EventIndexValidityResult(BLOCKED, t_event, row_count, "t_event must be > 0")
    if t_event >= row_count:
        return EventIndexValidityResult(BLOCKED, t_event, row_count, "t_event must be < row_count")
    return EventIndexValidityResult(PASS, t_event, row_count, "t_event is within valid bounds")


@dataclass
class WindowOverlapEntry:
    window_start: str
    window_end: str
    position: str       # "before_event", "after_event", "overlaps_event"

    def to_dict(self) -> dict:
        return {"window_start": self.window_start, "window_end": self.window_end, "position": self.position}


def check_anomaly_window_overlap(
    series: pd.Series, t_event: int, labeled_anomaly_windows: list
) -> list[WindowOverlapEntry]:
    """Classifies each REAL labeled window relative to t_event, using
    actual series timestamps — never guessed."""
    event_ts = series.index[t_event]
    results = []
    for w_start, w_end in labeled_anomaly_windows:
        w_start_ts, w_end_ts = pd.Timestamp(w_start), pd.Timestamp(w_end)
        if w_end_ts < event_ts:
            position = "before_event"
        elif w_start_ts > event_ts:
            position = "after_event"
        else:
            position = "overlaps_event"
        results.append(WindowOverlapEntry(str(w_start), str(w_end), position))
    return results


# ----------------------------------------------------------------------
# Checkpoint feasibility (reuses the runner's own boundary arithmetic)
# ----------------------------------------------------------------------


@dataclass
class CheckpointFeasibilityResult:
    status: str
    t_event_window: int
    n_total_windows: int
    n_pre_event_windows: int
    n_post_event_windows: int
    minimum_post_event_required: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "status": self.status, "t_event_window": self.t_event_window,
            "n_total_windows": self.n_total_windows, "n_pre_event_windows": self.n_pre_event_windows,
            "n_post_event_windows": self.n_post_event_windows,
            "minimum_post_event_required": self.minimum_post_event_required, "reason": self.reason,
        }


def check_checkpoint_feasibility(
    row_count: int, t_event: int, window_size: int, stride: int = 1,
    retrain_window_size: int = 20, post_retrain_window_size: int = 20,
) -> CheckpointFeasibilityResult:
    """Checks whether pre_drift / post_drift_pre_retrain / post_retrain
    checkpoints can ALL be constructed without overlap, using the EXACT
    same windowed-index boundary function the real runner uses
    (imported directly, not re-derived)."""
    n_total_windows = max(0, row_count - window_size + 1)
    t_event_window = _first_window_touched_by_event(t_event, window_size, stride)

    if t_event_window < 1 or t_event_window >= n_total_windows - 1:
        return CheckpointFeasibilityResult(
            BLOCKED, t_event_window, n_total_windows, max(0, t_event_window), 0, 0,
            "t_event_window leaves no usable pre/post-event windows",
        )

    n_pre_event = t_event_window
    n_post_event = n_total_windows - t_event_window
    minimum_required = retrain_window_size + post_retrain_window_size + 2  # +2: at least one window each for detection and retrain-trigger indices

    if n_post_event < minimum_required:
        return CheckpointFeasibilityResult(
            WARN, t_event_window, n_total_windows, n_pre_event, n_post_event, minimum_required,
            f"Only {n_post_event} post-event windows available; recommend >= {minimum_required} "
            f"for a comfortable retrain_window_size={retrain_window_size} + "
            f"post_retrain_window_size={post_retrain_window_size}",
        )

    return CheckpointFeasibilityResult(
        PASS, t_event_window, n_total_windows, n_pre_event, n_post_event, minimum_required,
        "Sufficient post-event windows for all three checkpoints without overlap",
    )


# ----------------------------------------------------------------------
# Drift-magnitude candidate analysis (pure function of baseline stats)
# ----------------------------------------------------------------------


@dataclass
class DriftMagnitudeCandidate:
    label: str
    offset: float
    scale_multiplier: float
    expected_shifted_mean: float
    distinguishable_from_noise: bool       # |offset| large relative to baseline std
    plausible_range: bool                  # does not push values negative when baseline is non-negative (e.g. CPU %)
    notes: list

    def to_dict(self) -> dict:
        return {
            "label": self.label, "offset": self.offset, "scale_multiplier": self.scale_multiplier,
            "expected_shifted_mean": self.expected_shifted_mean,
            "distinguishable_from_noise": self.distinguishable_from_noise,
            "plausible_range": self.plausible_range, "notes": self.notes,
        }


def recommend_drift_magnitudes(baseline: BaselineStatistics, baseline_is_nonnegative: bool = True) -> list[DriftMagnitudeCandidate]:
    """Generates candidate drift magnitudes purely as a function of
    ACTUAL baseline statistics — never a fixed constant. Two families
    are generated:

    (1) std-based candidates (multiples of the baseline standard
        deviation) — the natural default, but NOTE: if
        `n_labeled_anomalies_in_baseline > 0`, mean/std are computed
        from a baseline region that itself contains real anomalous
        spikes, which can substantially inflate them relative to the
        series' typical (median) behavior. This is reported via
        `contaminated_by_baseline_anomalies` on each std-based
        candidate — it is not fixed automatically, since the correct
        response (accept it, or choose a cleaner baseline region, or
        prefer the robust candidates below) is a decision for review,
        not something this function silently resolves.

    (2) IQR-based ("robust") candidates, using the median and
        interquartile range instead of mean/std — much less sensitive
        to a small number of extreme baseline points, useful
        specifically when contamination is flagged.
    """
    candidates = []
    contaminated = baseline.n_labeled_anomalies_in_baseline > 0

    for k, label in ((1.0, "1_std_shift"), (2.0, "2_std_shift"), (3.0, "3_std_shift")):
        offset = k * baseline.std
        shifted_mean = baseline.mean + offset
        distinguishable = k >= 2.0
        plausible = (not baseline_is_nonnegative) or (baseline.minimum + offset >= 0)
        notes = []
        if not distinguishable:
            notes.append("A 1-std shift may be hard to distinguish from ordinary baseline variation.")
        if not plausible:
            notes.append("This offset would push the minimum baseline value negative — implausible for a non-negative signal (e.g. CPU %).")
        if contaminated:
            notes.append(
                f"CAUTION: baseline std is computed from a region containing "
                f"{baseline.n_labeled_anomalies_in_baseline} real labeled anomaly window(s) — "
                f"std may be substantially inflated above the series' typical variability. "
                f"Compare against the IQR-based candidates below."
            )
        candidates.append(DriftMagnitudeCandidate(
            label=label, offset=offset, scale_multiplier=1.0, expected_shifted_mean=shifted_mean,
            distinguishable_from_noise=distinguishable, plausible_range=plausible, notes=notes,
        ))

    # One combined offset+scale candidate, moderate on both axes
    combined_offset = 1.5 * baseline.std
    combined_scale = 1.3
    combined_mean = baseline.mean * combined_scale + combined_offset
    combined_plausible = (not baseline_is_nonnegative) or (baseline.minimum * combined_scale + combined_offset >= 0)
    combined_notes = ["Combined offset+scale — moderate shift on both axes, per the approved primary drift mechanism."]
    if contaminated:
        combined_notes.append("CAUTION: uses baseline std, which may be anomaly-inflated — see note above.")
    candidates.append(DriftMagnitudeCandidate(
        label="combined_offset_and_scale", offset=combined_offset, scale_multiplier=combined_scale,
        expected_shifted_mean=combined_mean, distinguishable_from_noise=True, plausible_range=combined_plausible,
        notes=combined_notes,
    ))

    # Robust (IQR-based) candidates — much less sensitive to a small
    # number of extreme baseline points than mean/std.
    iqr = baseline.quantiles.get("q75", baseline.mean) - baseline.quantiles.get("q25", baseline.mean)
    for k, label in ((1.5, "1.5_iqr_shift_robust"), (3.0, "3_iqr_shift_robust")):
        offset = k * iqr
        shifted_median = baseline.median + offset
        plausible = (not baseline_is_nonnegative) or (baseline.minimum + offset >= 0)
        notes = [
            "Robust candidate: uses median/IQR instead of mean/std, so it is not "
            "distorted by a small number of extreme baseline points."
        ]
        if not plausible:
            notes.append("This offset would push the minimum baseline value negative.")
        candidates.append(DriftMagnitudeCandidate(
            label=label, offset=offset, scale_multiplier=1.0, expected_shifted_mean=shifted_median,
            distinguishable_from_noise=(k >= 1.5), plausible_range=plausible, notes=notes,
        ))

    return candidates

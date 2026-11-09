"""NAB dataset inspection — environment checks, dataset discovery,
series/label inspection, and evidence-based parameter recommendation.

STATUS: IMPLEMENTED. Parsing/inspection logic is unit-tested against
SYNTHETIC fixture CSV/JSON files (see tests/unit/test_nab_inspection.py)
— this proves the CODE is correct. 

This module reuses existing project modules rather than re-implementing
their logic: src.data.loader.load_nab_series for CSV loading,
src.data.validation.validate_series/baseline_statistics for quality
checks. It adds NEW logic only for what didn't exist yet: dependency
checks, dataset/label discovery, and parameter recommendation.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.loader import DatasetLoadError, load_nab_series
from src.data.validation import baseline_statistics, validate_series

# Statuses used consistently throughout this module's output.
PASS = "PASS"
WARN = "WARN"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"

REQUIRED_PACKAGES = ("numpy", "pandas", "sklearn", "joblib", "yaml", "river", "shap")

# Common NAB subset names, checked in this priority order.
NAB_SUBSET_PRIORITY = ("realAWSCloudwatch", "artificialWithAnomaly")

# Common relative locations to look for a local NAB checkout — the
# script also accepts an explicit --nab-root override (see
# scripts/inspect_nab_full.py).
COMMON_NAB_ROOT_CANDIDATES = (
    "./NAB", "./data/NAB", "../NAB", "~/NAB", "~/Downloads/NAB",
)


# ----------------------------------------------------------------------
# Environment / dependency checks
# ----------------------------------------------------------------------


@dataclass
class DependencyCheckResult:
    package: str
    status: str            # PASS or BLOCKED
    version: str | None
    error: str | None

    def to_dict(self) -> dict:
        return {"package": self.package, "status": self.status, "version": self.version, "error": self.error}


def check_dependency(package_name: str) -> DependencyCheckResult:
    """Actually attempts the import — never assumes availability."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        return DependencyCheckResult(package=package_name, status=PASS, version=version, error=None)
    except ImportError as exc:
        return DependencyCheckResult(package=package_name, status=BLOCKED, version=None, error=str(exc))


def check_all_dependencies() -> list[DependencyCheckResult]:
    return [check_dependency(pkg) for pkg in REQUIRED_PACKAGES]


@dataclass
class EnvironmentReport:
    python_version: str
    operating_system: str
    cpu_only: bool                 # always True by project policy — reported, not measured
    dependencies: list

    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "operating_system": self.operating_system,
            "cpu_only": self.cpu_only,
            "dependencies": [d.to_dict() for d in self.dependencies],
        }


def build_environment_report() -> EnvironmentReport:
    return EnvironmentReport(
        python_version=sys.version.split()[0],
        operating_system=f"{platform.system()} {platform.release()}",
        cpu_only=True,
        dependencies=check_all_dependencies(),
    )


# ----------------------------------------------------------------------
# Dataset discovery
# ----------------------------------------------------------------------


@dataclass
class DatasetDiscoveryResult:
    status: str                    # PASS or BLOCKED
    root_path: str | None
    reason: str | None             # populated when BLOCKED

    def to_dict(self) -> dict:
        return {"status": self.status, "root_path": self.root_path, "reason": self.reason}


def discover_nab_root(explicit_root: str | None = None) -> DatasetDiscoveryResult:
    """If `explicit_root` is given, it is AUTHORITATIVE — checked alone,
    never silently falling back to common candidate locations if it's
    invalid. (Real bug found via laptop execution: the original version
    always checked common candidates too, so an explicit-but-wrong root
    could silently succeed by finding an unrelated real NAB checkout
    elsewhere on the machine — exactly what happened when a real NAB
    checkout existed at ~/Downloads/NAB. An explicit override must mean
    exactly that path, not 'prefer this, but fall back quietly'.)

    Only when NO explicit_root is given does this fall back to checking
    the common candidate locations."""
    if explicit_root:
        path = Path(explicit_root).expanduser()
        if path.is_dir() and (path / "data").is_dir():
            return DatasetDiscoveryResult(status=PASS, root_path=str(path), reason=None)
        return DatasetDiscoveryResult(
            status=BLOCKED, root_path=None,
            reason=f"Explicit --nab-root '{explicit_root}' does not exist or has no data/ subdirectory.",
        )

    checked = []
    for candidate in COMMON_NAB_ROOT_CANDIDATES:
        path = Path(candidate).expanduser()
        checked.append(str(path))
        if path.is_dir() and (path / "data").is_dir():
            return DatasetDiscoveryResult(status=PASS, root_path=str(path), reason=None)

    return DatasetDiscoveryResult(
        status=BLOCKED, root_path=None,
        reason=f"No local NAB checkout found. Checked: {checked}",
    )


def discover_series_files(nab_root: str, subset: str) -> list[str]:
    """Lists CSV files under {nab_root}/data/{subset}/. Returns an
    empty list (not an error) if the subset directory doesn't exist —
    callers decide whether that's a problem."""
    subset_dir = Path(nab_root) / "data" / subset
    if not subset_dir.is_dir():
        return []
    return sorted(str(p.relative_to(nab_root)) for p in subset_dir.glob("*.csv"))


# ----------------------------------------------------------------------
# Per-series inspection (reuses loader.py / validation.py)
# ----------------------------------------------------------------------


@dataclass
class SeriesInspectionResult:
    relative_path: str
    status: str                    # PASS or BLOCKED
    row_count: int | None = None
    feature_count: int = 1         # NAB series are univariate (timestamp, value)
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    inferred_frequency: str | None = None
    is_monotonic: bool | None = None
    n_missing_values: int | None = None
    n_duplicate_timestamps: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "relative_path": self.relative_path, "status": self.status,
            "row_count": self.row_count, "feature_count": self.feature_count,
            "timestamp_start": self.timestamp_start, "timestamp_end": self.timestamp_end,
            "inferred_frequency": self.inferred_frequency, "is_monotonic": self.is_monotonic,
            "n_missing_values": self.n_missing_values,
            "n_duplicate_timestamps": self.n_duplicate_timestamps, "error": self.error,
        }


def inspect_series_file(nab_root: str, relative_path: str) -> SeriesInspectionResult:
    """Loads and validates ONE series file using the EXISTING loader/
    validation modules — no new parsing logic duplicated here."""
    full_path = Path(nab_root) / relative_path
    try:
        series = load_nab_series(full_path)
    except DatasetLoadError as exc:
        return SeriesInspectionResult(relative_path=relative_path, status=BLOCKED, error=str(exc))

    report = validate_series(series, minimum_length_required=0)  # length gate applied separately, by the caller
    return SeriesInspectionResult(
        relative_path=relative_path, status=PASS, row_count=report.length,
        timestamp_start=str(report.start), timestamp_end=str(report.end),
        inferred_frequency=report.inferred_frequency, is_monotonic=report.is_monotonic,
        n_missing_values=report.n_missing_values, n_duplicate_timestamps=report.n_duplicate_timestamps,
    )


# ----------------------------------------------------------------------
# Label / anomaly-window inspection
# ----------------------------------------------------------------------


@dataclass
class LabelInspectionResult:
    status: str                     # PASS, WARN, or BLOCKED
    source_file: str | None
    n_windows: int
    windows: list                   # list of (start, end) timestamp-string pairs
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status, "source_file": self.source_file,
            "n_windows": self.n_windows, "windows": self.windows, "note": self.note,
        }


def inspect_labels(nab_root: str, series_relative_path: str) -> LabelInspectionResult:
    """Looks for labels/combined_windows.json (preferred — gives real
    window boundaries) and falls back to labels/combined_labels.json
    (point labels only; each point is reported as a zero-width window,
    clearly noted as a fallback, not a real window boundary from NAB
    itself)."""
    windows_path = Path(nab_root) / "labels" / "combined_windows.json"
    labels_path = Path(nab_root) / "labels" / "combined_labels.json"
    key = series_relative_path.replace("\\", "/")
    if key.startswith("data/"):
        key = key[len("data/"):]

    if windows_path.is_file():
        try:
            data = json.loads(windows_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return LabelInspectionResult(status=BLOCKED, source_file=str(windows_path), n_windows=0, windows=[], note=str(exc))
        windows = data.get(key)
        if windows is None:
            return LabelInspectionResult(
                status=WARN, source_file=str(windows_path), n_windows=0, windows=[],
                note=f"Series key '{key}' not found in combined_windows.json",
            )
        return LabelInspectionResult(
            status=PASS, source_file=str(windows_path), n_windows=len(windows),
            windows=[tuple(w) for w in windows],
        )

    if labels_path.is_file():
        try:
            data = json.loads(labels_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return LabelInspectionResult(status=BLOCKED, source_file=str(labels_path), n_windows=0, windows=[], note=str(exc))
        points = data.get(key)
        if points is None:
            return LabelInspectionResult(
                status=WARN, source_file=str(labels_path), n_windows=0, windows=[],
                note=f"Series key '{key}' not found in combined_labels.json",
            )
        return LabelInspectionResult(
            status=WARN, source_file=str(labels_path), n_windows=len(points),
            windows=[(p, p) for p in points],
            note="Only point labels available (no combined_windows.json) — each "
                 "reported 'window' is a zero-width point, not a real NAB window boundary.",
        )

    return LabelInspectionResult(
        status=BLOCKED, source_file=None, n_windows=0, windows=[],
        note="Neither combined_windows.json nor combined_labels.json found under labels/",
    )


def build_binary_labels_from_windows(series: pd.Series, windows: list) -> "np.ndarray":
    """Converts real labeled anomaly windows (timestamp pairs) into a
    binary ground-truth array aligned to `series`'s own index — 1 for
    any raw point whose timestamp falls within [window_start,
    window_end] inclusive of ANY window, else 0. Never fabricates a
    label for a point outside every real window; an empty `windows`
    list correctly produces an all-zero array (e.g. for a genuine
    no-anomaly series), never an error.
    """
    import numpy as np

    y_true = np.zeros(len(series), dtype=int)
    for w_start, w_end in windows:
        w_start_ts, w_end_ts = pd.Timestamp(w_start), pd.Timestamp(w_end)
        mask = (series.index >= w_start_ts) & (series.index <= w_end_ts)
        y_true[mask] = 1
    return y_true


# ----------------------------------------------------------------------
# Parameter recommendation (pure function — no I/O, fully testable
# with synthetic row counts)
# ----------------------------------------------------------------------


@dataclass
class ParameterRecommendation:
    status: str                          # PASS or WARN
    row_count: int
    window_size: int
    recommended_t_event: int | None
    recommended_detection_window: int | None       # upper bound on how many post-event rows are available to scan for drift detection
    recommended_post_retrain_window: int | None    # None means "TBD — see Phase 8.1 decision", not "zero"
    minimum_pre_event_required: int
    notes: list

    def to_dict(self) -> dict:
        return {
            "status": self.status, "row_count": self.row_count, "window_size": self.window_size,
            "recommended_t_event": self.recommended_t_event,
            "recommended_detection_window": self.recommended_detection_window,
            "recommended_post_retrain_window": self.recommended_post_retrain_window,
            "minimum_pre_event_required": self.minimum_pre_event_required,
            "notes": self.notes,
        }


def recommend_parameters(
    row_count: int, window_size: int,
    background_sample_size: int = 20, evaluation_sample_size: int = 10,
) -> ParameterRecommendation:
    """Evidence-based recommendation from an ACTUAL inspected row_count
    — never a guessed default. `minimum_pre_event_required` is derived
    from the current default background+evaluation sample sizes
    (Phase 8's ExperimentConfig defaults) plus one window's worth of
    lookback, so it reflects what the pipeline actually needs, not an
    arbitrary number.

    Split rationale: t_event at 60% of the series leaves a majority-
    sized pre-event region (needed for detector training AND both
    fixed samples) while still leaving a meaningful post-event region
    for drift detection, retraining, and post-retrain evaluation. This
    is a starting recommendation for review, not a locked decision —
    it does not resolve the post-retrain-window methodological question
    (Phase 8.1), which remains open here.
    """
    notes = []
    minimum_pre_event_required = background_sample_size + evaluation_sample_size + window_size
    n_windows = max(0, row_count - window_size + 1)

    if n_windows < minimum_pre_event_required * 2:
        return ParameterRecommendation(
            status=WARN, row_count=row_count, window_size=window_size,
            recommended_t_event=None, recommended_detection_window=None,
            recommended_post_retrain_window=None,
            minimum_pre_event_required=minimum_pre_event_required,
            notes=[
                f"Series has only {n_windows} windows at window_size={window_size}; "
                f"needs at least {minimum_pre_event_required * 2} for a usable "
                f"pre/post-event split — NOT RECOMMENDED as-is."
            ],
        )

    t_event = int(row_count * 0.6)
    post_event_windows = max(0, (row_count - window_size + 1) - t_event)
    detection_window = post_event_windows  # upper bound; real detection may stop earlier
    notes.append(
        "recommended_post_retrain_window is intentionally None (TBD) — the Option A "
        "vs. Option B decision from Phase 8.1 is NOT resolved by this inspection alone "
        "unless the actual data makes one option clearly necessary."
    )
    if post_event_windows < minimum_pre_event_required:
        notes.append(
            "WARNING: post-event region is smaller than the minimum needed for a "
            "second fixed-sample-sized region — review before final selection."
        )
        status = WARN
    else:
        status = PASS

    return ParameterRecommendation(
        status=status, row_count=row_count, window_size=window_size,
        recommended_t_event=t_event, recommended_detection_window=detection_window,
        recommended_post_retrain_window=None,
        minimum_pre_event_required=minimum_pre_event_required, notes=notes,
    )

"""Phase 11 — Real NAB baseline analysis and parameter-locking CLI.

Run on a machine with the actual NAB file:

    python3 scripts/analyze_baseline_and_lock_params.py \\
        --nab-root /path/to/NAB \\
        --series data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv \\
        --t-event 2419 \\
        --window-size 10 \\
        --retrain-window-size 20 \\
        --post-retrain-window-size 20



This script NEVER writes to config/experiment.yaml automatically — it
only recommends. Locking config values is a separate, explicit,
reviewed step, consistent with the project's standing rule that
NAB-dependent values are never silently chosen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.baseline_analysis import (
    BLOCKED,
    PASS,
    check_anomaly_window_overlap,
    check_checkpoint_feasibility,
    check_event_index_validity,
    compute_baseline_statistics,
    recommend_drift_magnitudes,
)
from src.data.loader import DatasetLoadError, load_nab_series
from src.data.nab_inspection import discover_nab_root, inspect_labels

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_analysis(nab_root_arg, series_relative_path, t_event, window_size, retrain_window_size, post_retrain_window_size):
    report = {}

    discovery = discover_nab_root(nab_root_arg)
    report["dataset_discovery"] = discovery.to_dict()
    if discovery.status == BLOCKED:
        report["overall_status"] = BLOCKED
        report["overall_reason"] = discovery.reason
        return report

    nab_root = discovery.root_path
    full_path = Path(nab_root) / series_relative_path

    try:
        series = load_nab_series(full_path)
    except DatasetLoadError as exc:
        report["overall_status"] = BLOCKED
        report["overall_reason"] = f"Could not load series: {exc}"
        return report

    report["series"] = {
        "relative_path": series_relative_path, "row_count": len(series),
        "timestamp_start": str(series.index.min()), "timestamp_end": str(series.index.max()),
        "inferred_frequency": str(series.index.inferred_freq),
    }

    label_result = inspect_labels(nab_root, str(Path("data") / series_relative_path) if not series_relative_path.startswith("data") else series_relative_path)
    report["labels"] = label_result.to_dict()

    validity = check_event_index_validity(t_event, len(series))
    report["event_index_validity"] = validity.to_dict()
    if validity.status == BLOCKED:
        report["overall_status"] = BLOCKED
        report["overall_reason"] = validity.reason
        return report

    overlap = check_anomaly_window_overlap(series, t_event, label_result.windows)
    report["anomaly_window_overlap"] = [o.to_dict() for o in overlap]

    baseline = compute_baseline_statistics(series, t_event, label_result.windows)
    report["baseline_statistics"] = baseline.to_dict()

    feasibility = check_checkpoint_feasibility(
        len(series), t_event, window_size, retrain_window_size=retrain_window_size,
        post_retrain_window_size=post_retrain_window_size,
    )
    report["checkpoint_feasibility"] = feasibility.to_dict()

    drift_candidates = recommend_drift_magnitudes(baseline)
    report["drift_magnitude_candidates"] = [c.to_dict() for c in drift_candidates]

    report["overall_status"] = PASS if feasibility.status != BLOCKED else BLOCKED
    report["overall_reason"] = None
    return report


def write_json(report, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))


def write_markdown(report, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 11 — Real NAB Baseline Analysis and Parameter Recommendation\n"]
    lines.append(f"**Overall status: {report.get('overall_status')}**\n")
    if report.get("overall_reason"):
        lines.append(f"Reason: {report['overall_reason']}\n")

    if "series" in report:
        s = report["series"]
        lines.append("\n## Series (directly inspected)\n")
        lines.append(f"- Path: {s['relative_path']}")
        lines.append(f"- Row count: {s['row_count']}")
        lines.append(f"- Timestamp range: {s['timestamp_start']} to {s['timestamp_end']}")
        lines.append(f"- Inferred frequency: {s['inferred_frequency']}")

    if "labels" in report:
        l = report["labels"]
        lines.append("\n## Labels (directly inspected)\n")
        lines.append(f"- Status: {l['status']}, n_windows: {l['n_windows']}")
        for w in l.get("windows", []):
            lines.append(f"  - {w}")

    if "event_index_validity" in report:
        v = report["event_index_validity"]
        lines.append("\n## Event Index Validity\n")
        lines.append(f"- Status: {v['status']} — {v['reason']}")

    if "anomaly_window_overlap" in report:
        lines.append("\n## Anomaly Window Position Relative to t_event\n")
        for o in report["anomaly_window_overlap"]:
            lines.append(f"- {o['window_start']} to {o['window_end']}: {o['position']}")

    if "baseline_statistics" in report:
        b = report["baseline_statistics"]
        lines.append("\n## Baseline Statistics (pre-event region only, directly computed)\n")
        for k, v in b.items():
            lines.append(f"- {k}: {v}")

    if "checkpoint_feasibility" in report:
        f = report["checkpoint_feasibility"]
        lines.append("\n## Checkpoint Feasibility\n")
        for k, v in f.items():
            lines.append(f"- {k}: {v}")

    if "drift_magnitude_candidates" in report:
        lines.append("\n## Drift Magnitude Candidates (RECOMMENDATIONS, not locked)\n")
        for c in report["drift_magnitude_candidates"]:
            lines.append(f"\n### {c['label']}\n")
            for k, v in c.items():
                if k != "label":
                    lines.append(f"- {k}: {v}")

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 11 baseline analysis")
    parser.add_argument("--nab-root", default=None)
    parser.add_argument("--series", required=True, help="Relative path, e.g. data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv")
    parser.add_argument("--t-event", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--retrain-window-size", type=int, default=20)
    parser.add_argument("--post-retrain-window-size", type=int, default=20)
    args = parser.parse_args()

    report = run_analysis(
        args.nab_root, args.series, args.t_event, args.window_size,
        args.retrain_window_size, args.post_retrain_window_size,
    )

    json_path = REPO_ROOT / "artifacts" / "phase11_baseline_and_parameters.json"
    md_path = REPO_ROOT / "docs" / "PHASE11_BASELINE_AND_PARAMETERS.md"
    write_json(report, json_path)
    write_markdown(report, md_path)

    print(f"Overall status: {report.get('overall_status')}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()

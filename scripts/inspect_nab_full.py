"""Full NAB dataset inspection CLI.

Run this on a machine with the actual NAB dataset available:

    python3 scripts/inspect_nab_full.py --nab-root /path/to/NAB

Or, if NAB is checked out in one of the common relative locations
(./NAB, ./data/NAB, ../NAB, ~/NAB, ~/Downloads/NAB):

    python3 scripts/inspect_nab_full.py


Every fact reported is either DIRECTLY INSPECTED (loaded and measured
from an actual file) or explicitly marked as a RECOMMENDATION derived
from those facts — never fabricated, never estimated without saying so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.nab_inspection import (
    BLOCKED,
    NAB_SUBSET_PRIORITY,
    NOT_RUN,
    build_environment_report,
    discover_nab_root,
    discover_series_files,
    inspect_labels,
    inspect_series_file,
    recommend_parameters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOW_SIZE_FOR_RECOMMENDATION = 10  # matches this project's tested default (src/data/preprocessing.py usage throughout); NOT a newly locked NAB-specific value


def run_inspection(nab_root_arg: str | None) -> dict:
    report = {"environment": build_environment_report().to_dict()}

    discovery = discover_nab_root(nab_root_arg)
    report["dataset_discovery"] = discovery.to_dict()

    if discovery.status == BLOCKED:
        report["candidate_series"] = []
        report["overall_status"] = BLOCKED
        report["overall_reason"] = discovery.reason
        return report

    nab_root = discovery.root_path
    candidates = []
    for subset in NAB_SUBSET_PRIORITY:
        files = discover_series_files(nab_root, subset)
        for relative_path in files:
            series_result = inspect_series_file(nab_root, relative_path)
            label_result = inspect_labels(nab_root, relative_path)

            entry = {
                "series": series_result.to_dict(),
                "labels": label_result.to_dict(),
            }
            if series_result.status == "PASS":
                rec = recommend_parameters(series_result.row_count, WINDOW_SIZE_FOR_RECOMMENDATION)
                entry["recommendation"] = rec.to_dict()
            else:
                entry["recommendation"] = {"status": NOT_RUN, "reason": "series inspection failed"}
            candidates.append(entry)

    report["candidate_series"] = candidates
    report["overall_status"] = "PASS" if candidates else BLOCKED
    report["overall_reason"] = None if candidates else "NAB root found but no series files discovered in any known subset"
    return report


def write_json_artifact(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))


def write_markdown_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase 10 — Real NAB Dataset Inspection\n")
    lines.append(f"**Overall status: {report['overall_status']}**\n")
    if report.get("overall_reason"):
        lines.append(f"Reason: {report['overall_reason']}\n")

    lines.append("\n## Environment (directly inspected)\n")
    env = report["environment"]
    lines.append(f"- Python: {env['python_version']}")
    lines.append(f"- OS: {env['operating_system']}")
    lines.append(f"- CPU-only (project policy): {env['cpu_only']}")
    lines.append("\n| Package | Status | Version | Error |")
    lines.append("|---|---|---|---|")
    for dep in env["dependencies"]:
        lines.append(f"| {dep['package']} | {dep['status']} | {dep['version'] or '-'} | {dep['error'] or '-'} |")

    lines.append("\n## Dataset Discovery (directly inspected)\n")
    disc = report["dataset_discovery"]
    lines.append(f"- Status: {disc['status']}")
    lines.append(f"- Root path: {disc['root_path'] or 'NOT FOUND'}")
    if disc.get("reason"):
        lines.append(f"- Reason: {disc['reason']}")

    if report["candidate_series"]:
        lines.append("\n## Candidate Series (directly inspected + recommendations)\n")
        for entry in report["candidate_series"]:
            s = entry["series"]
            l = entry["labels"]
            r = entry["recommendation"]
            lines.append(f"\n### {s['relative_path']}\n")
            lines.append(f"- Status: {s['status']}")
            if s["status"] == "PASS":
                lines.append(f"- Row count (inspected): {s['row_count']}")
                lines.append(f"- Timestamp range: {s['timestamp_start']} to {s['timestamp_end']}")
                lines.append(f"- Inferred frequency: {s['inferred_frequency']}")
                lines.append(f"- Monotonic: {s['is_monotonic']}")
                lines.append(f"- Missing values: {s['n_missing_values']}")
                lines.append(f"- Duplicate timestamps: {s['n_duplicate_timestamps']}")
            else:
                lines.append(f"- Error: {s['error']}")
            lines.append(f"- Labels status: {l['status']} (source: {l['source_file'] or 'none found'})")
            lines.append(f"- Number of labeled windows: {l['n_windows']}")
            if l.get("note"):
                lines.append(f"- Label note: {l['note']}")
            lines.append(f"- Recommendation status: {r.get('status')}")
            if r.get("recommended_t_event") is not None:
                lines.append(f"- Recommended t_event (RECOMMENDATION, not locked): {r['recommended_t_event']}")
                lines.append(f"- Recommended detection window upper bound (RECOMMENDATION): {r['recommended_detection_window']}")
                lines.append("- Recommended post-retrain window: TBD (Phase 8.1 decision not resolved here)")
            for note in r.get("notes", []):
                lines.append(f"  - Note: {note}")
    else:
        lines.append("\n## Candidate Series\n\nNone inspected — dataset discovery was BLOCKED, see above.\n")

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 10 NAB dataset inspection")
    parser.add_argument("--nab-root", default=None, help="Path to a local NAB checkout root")
    args = parser.parse_args()

    report = run_inspection(args.nab_root)

    json_path = REPO_ROOT / "artifacts" / "phase10_nab_inspection.json"
    md_path = REPO_ROOT / "docs" / "PHASE10_NAB_INSPECTION.md"
    write_json_artifact(report, json_path)
    write_markdown_report(report, md_path)

    print(f"Overall status: {report['overall_status']}")
    print(f"JSON artifact: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()

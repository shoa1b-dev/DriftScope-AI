"""Download and inspect real NAB candidate series.


Usage:
    python3 scripts/inspect_nab_candidates.py

This downloads each realAWSCloudwatch CPU-utilization series (the
family confirmed to exist via live GitHub lookup during Phase 1) and
prints a real ValidationReport + baseline statistics for each, so the
window size / drift magnitude / final series choice can be based on
actual numbers rather than assumption.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_nab_series
from src.data.validation import baseline_statistics, validate_series

RAW_BASE = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/"

# Confirmed to exist in data/realAWSCloudwatch/ via live GitHub directory

CANDIDATE_FILES = [
    "ec2_cpu_utilization_24ae8d.csv",
    "ec2_cpu_utilization_53ea38.csv",
    "ec2_cpu_utilization_5f5533.csv",
    "ec2_cpu_utilization_77c1ca.csv",
    "ec2_cpu_utilization_825cc2.csv",
    "ec2_cpu_utilization_ac20cd.csv",
    "ec2_cpu_utilization_c6585a.csv",
    "ec2_cpu_utilization_fe7f93.csv",
]

# Minimum length gate: needs room for pre-drift baseline + post-drift/
# pre-retrain + post-retrain, each of size >= a working window. Using a
# conservative floor here; refine once real lengths are known.
MINIMUM_LENGTH_REQUIRED = 900  # ~3 segments x 300 points, adjust after real inspection

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(filename: str) -> Path:
    out_path = OUT_DIR / filename
    if out_path.exists():
        return out_path
    resp = requests.get(RAW_BASE + filename, timeout=30)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'series':35s} {'len':>6s} {'missing':>8s} {'dupes':>7s} "
          f"{'monotonic':>10s} {'freq':>6s} {'mean':>8s} {'std':>8s} {'gate_pass':>10s}")
    for filename in CANDIDATE_FILES:
        try:
            path = download(filename)
        except Exception as exc:  # noqa: BLE001
            print(f"{filename:35s}  DOWNLOAD FAILED: {exc}")
            continue

        series = load_nab_series(path)
        report = validate_series(series, minimum_length_required=MINIMUM_LENGTH_REQUIRED)
        stats = baseline_statistics(series, end_index=min(len(series) // 3, len(series)))

        print(
            f"{filename:35s} {report.length:6d} {report.n_missing_values:8d} "
            f"{report.n_duplicate_timestamps:7d} {str(report.is_monotonic):>10s} "
            f"{str(report.inferred_frequency):>6s} {stats['mean']:8.2f} {stats['std']:8.2f} "
            f"{str(report.passes_minimum_length):>10s}"
        )

    print("\nPaste this full output back so the final series/window/magnitude "
          "recommendation can be based on real numbers.")


if __name__ == "__main__":
    main()

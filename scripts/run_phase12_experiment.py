"""Phase 12 — Primary Real NAB Experiment orchestration.

Run on a machine with river and shap installed 
    python3 scripts/run_phase12_experiment.py --nab-root ~/Downloads/NAB

Runs the REAL 2 conditions x 5 approved seeds = 10 `run_experiment`
calls (each producing both detectors -> 20 detector-condition results)
against the ACTUAL locked NAB series, using the ACTUAL locked
parameters from config/experiment.yaml. NO monitor or SHAP fakes are
injected here — this is the real, production execution path. If a
specific run legitimately fails (DriftNeverDetectedError,
InsufficientPostRetrainDataError, or a dependency error), that run is
recorded as a structured failure with the exact exception type and
message — the script does NOT crash the whole batch, silently retry
with different parameters, or fabricate a substitute result. The other
runs continue.

"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import DatasetLoadError, load_nab_series
from src.data.nab_inspection import build_binary_labels_from_windows, discover_nab_root, inspect_labels
from src.experiments.experiment_config import APPROVED_SEEDS, ExperimentConfig
from src.experiments.experiment_runner import (
    DriftNeverDetectedError,
    InsufficientPostRetrainDataError,
    MissingDataError,
    TooShortPreDriftRegionError,
    run_experiment,
)
from src.monitoring.drift_monitor import RiverUnavailableError
from src.explanation.shap_explainer import SHAPUnavailableError

REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("drift", "no_drift")

KNOWN_RUN_FAILURE_TYPES = (
    RiverUnavailableError, SHAPUnavailableError, DriftNeverDetectedError,
    InsufficientPostRetrainDataError, MissingDataError, TooShortPreDriftRegionError,
)


def load_locked_config() -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "experiment.yaml").read_text())
    return cfg


def build_experiment_config(cfg: dict, condition: str, seed: int) -> ExperimentConfig:
    """Builds an ExperimentConfig STRICTLY from the locked config.yaml
    values — no parameter is invented, adjusted, or defaulted here."""
    return ExperimentConfig(
        condition=condition,
        seed=seed,
        window_size=cfg["preprocessing"]["window_size"],
        t_event=cfg["drift_injection"]["t_inject"],
        drift_offset=cfg["drift_injection"]["magnitude"]["offset"],
        drift_scale_multiplier=cfg["drift_injection"]["magnitude"]["scale_multiplier"],
        retrain_window_size=cfg["retraining"]["retrain_window_size"],
        post_retrain_window_size=cfg["retraining"]["post_retrain_window_size"],
    )


def flatten_detector_result(condition, seed, detector_name, detector_result, run_status, error_type=None, error_message=None):
    """One flat row per detector-condition-seed combination, for the
    CSV summary. `run_status` is 'completed' or 'failed'; on failure
    every metric field is left as None — never a fabricated 0/NaN."""
    row = {
        "condition": condition, "seed": seed, "detector": detector_name, "status": run_status,
        "error_type": error_type, "error_message": error_message,
        "t_event_window": None, "t_detect": None, "t_retrain": None,
        "secondary_monitor_status": None, "secondary_monitor_detection_index": None,
    }
    for checkpoint in ("pre_drift", "post_drift_pre_retrain", "post_retrain"):
        for metric in ("precision", "recall", "f1"):
            row[f"{checkpoint}_{metric}"] = None
            row[f"{checkpoint}_{metric}_undefined"] = None
    row["cosine_vs_post_drift_mean"] = None
    row["cosine_vs_post_drift_n_invalid"] = None
    row["cosine_vs_post_retrain_mean"] = None
    row["cosine_vs_post_retrain_n_invalid"] = None
    row["jaccard_vs_post_drift_mean"] = None
    row["jaccard_vs_post_retrain_mean"] = None

    if detector_result is None:
        return row

    row["t_event_window"] = detector_result.t_event_window
    row["t_detect"] = detector_result.t_detect
    row["t_retrain"] = detector_result.t_retrain
    row["secondary_monitor_status"] = detector_result.secondary_monitor.status
    row["secondary_monitor_detection_index"] = detector_result.secondary_monitor.detection_index

    for checkpoint in ("pre_drift", "post_drift_pre_retrain", "post_retrain"):
        det = detector_result.detection[checkpoint]
        row[f"{checkpoint}_precision"] = det.precision.value
        row[f"{checkpoint}_precision_undefined"] = det.precision.undefined
        row[f"{checkpoint}_recall"] = det.recall.value
        row[f"{checkpoint}_recall_undefined"] = det.recall.undefined
        row[f"{checkpoint}_f1"] = det.f1.value
        row[f"{checkpoint}_f1_undefined"] = det.f1.undefined

    row["cosine_vs_post_drift_mean"] = detector_result.stability_vs_post_drift.cosine.mean
    row["cosine_vs_post_drift_n_invalid"] = detector_result.stability_vs_post_drift.cosine.n_invalid
    row["cosine_vs_post_retrain_mean"] = detector_result.stability_vs_post_retrain.cosine.mean
    row["cosine_vs_post_retrain_n_invalid"] = detector_result.stability_vs_post_retrain.cosine.n_invalid
    row["jaccard_vs_post_drift_mean"] = detector_result.stability_vs_post_drift.jaccard.mean
    row["jaccard_vs_post_retrain_mean"] = detector_result.stability_vs_post_retrain.jaccard.mean
    return row


def run_all(nab_root_arg: str | None) -> dict:
    cfg = load_locked_config()

    discovery = discover_nab_root(nab_root_arg)
    if discovery.status != "PASS":
        return {"overall_status": "BLOCKED", "reason": discovery.reason, "runs": []}

    nab_root = discovery.root_path
    series_relative_path = cfg["dataset"]["selected_series"]
    full_path = Path(nab_root) / series_relative_path

    try:
        series = load_nab_series(full_path)
    except DatasetLoadError as exc:
        return {"overall_status": "BLOCKED", "reason": f"Could not load series: {exc}", "runs": []}

    label_result = inspect_labels(nab_root, series_relative_path)
    y_true = build_binary_labels_from_windows(series, label_result.windows)

    all_rows = []
    full_results_by_key = {}
    seen_keys = set()

    for condition in CONDITIONS:
        for seed in APPROVED_SEEDS:
            config = build_experiment_config(cfg, condition, seed)
            run_start = time.time()
            try:
                result = run_experiment(series, y_true, config)  # REAL execution, no fakes
                runtime = time.time() - run_start
                for detector_name in ("detector_a", "detector_b"):
                    key = (condition, seed, detector_name)
                    if key in seen_keys:
                        raise RuntimeError(f"DUPLICATE RUN DETECTED: {key}")
                    seen_keys.add(key)
                    detector_result = getattr(result, detector_name)
                    row = flatten_detector_result(condition, seed, detector_name, detector_result, "completed")
                    row["runtime_seconds"] = runtime
                    all_rows.append(row)
                    full_results_by_key[f"{detector_name}_{condition}_{seed}"] = detector_result.to_metadata_dict()
            except KNOWN_RUN_FAILURE_TYPES as exc:
                runtime = time.time() - run_start
                for detector_name in ("detector_a", "detector_b"):
                    key = (condition, seed, detector_name)
                    seen_keys.add(key)
                    row = flatten_detector_result(
                        condition, seed, detector_name, None, "failed",
                        error_type=type(exc).__name__, error_message=str(exc),
                    )
                    row["runtime_seconds"] = runtime
                    all_rows.append(row)
            except Exception as exc:  # noqa: BLE001 — genuinely unexpected, still recorded honestly, not swallowed
                runtime = time.time() - run_start
                for detector_name in ("detector_a", "detector_b"):
                    key = (condition, seed, detector_name)
                    seen_keys.add(key)
                    row = flatten_detector_result(
                        condition, seed, detector_name, None, "failed",
                        error_type=f"UNEXPECTED:{type(exc).__name__}", error_message=str(exc),
                    )
                    row["runtime_seconds"] = runtime
                    all_rows.append(row)
                traceback.print_exc()

    return {
        "overall_status": "COMPLETED",
        "n_rows": len(all_rows),
        "n_completed": sum(1 for r in all_rows if r["status"] == "completed"),
        "n_failed": sum(1 for r in all_rows if r["status"] == "failed"),
        "rows": all_rows,
        "full_results": full_results_by_key,
        "series_relative_path": series_relative_path,
        "n_labeled_windows": label_result.n_windows,
    }


def write_outputs(report: dict):
    out_dir = REPO_ROOT / "artifacts" / "phase12"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "real_experiment_results.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    csv_path = out_dir / "real_experiment_results.csv"
    if report.get("rows"):
        import csv as csv_module
        fieldnames = list(report["rows"][0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in report["rows"]:
                writer.writerow(row)
    else:
        csv_path.write_text("status\nBLOCKED_NO_ROWS\n")

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Phase 12 real NAB experiment")
    parser.add_argument("--nab-root", default=None)
    args = parser.parse_args()

    report = run_all(args.nab_root)
    json_path, csv_path = write_outputs(report)

    print(f"Overall status: {report.get('overall_status')}")
    if report.get("overall_status") == "COMPLETED":
        print(f"Rows: {report['n_rows']} (expect 20) — completed: {report['n_completed']}, failed: {report['n_failed']}")
    else:
        print(f"Reason: {report.get('reason')}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

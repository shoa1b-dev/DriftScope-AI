"""Read-only loaders for the existing research artifacts (no Streamlit import here).

Nothing in this module writes to disk or imports from ``src/`` (the experiment code).
Missing values stay missing (NaN/None); undefined metrics are never converted to 0.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P12_JSON = "artifacts/phase12/real_experiment_results.json"
P12_CSV = "artifacts/phase12/real_experiment_results.csv"
P13_TABLES = "artifacts/phase13/tables"
P13_FIGS = "artifacts/phase13/figures"
CHECKPOINTS = ["pre_drift", "post_drift_pre_retrain", "post_retrain"]
METRICS = ["precision", "recall", "f1"]
DETECTOR_LABELS = {"detector_a": "Detector A · Isolation Forest",
                   "detector_b": "Detector B · MLP Autoencoder"}

REQUIRED_JSON_KEYS = {"overall_status", "n_rows", "n_completed", "n_failed", "rows", "full_results"}
REQUIRED_ROW_COLS = {"condition", "seed", "detector", "status"}


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing or malformed; message is user-readable."""


def abspath(rel: str) -> Path:
    return ROOT / rel


def read_json(rel: str) -> Any:
    p = abspath(rel)
    if not p.exists():
        raise ArtifactError(f"Required artifact not found: {rel}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{rel} is not valid JSON: {exc}") from exc


def read_csv(rel: str) -> pd.DataFrame:
    p = abspath(rel)
    if not p.exists():
        raise ArtifactError(f"Required artifact not found: {rel}")
    return pd.read_csv(p, encoding="utf-8")  # explicit UTF-8 (avoids mojibake of em dashes)


def read_text(rel: str) -> str | None:
    p = abspath(rel)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


# ---------------------------------------------------------------- validation
def validate_phase12(d: dict) -> list[str]:
    """Return a list of human-readable consistency problems (empty = consistent)."""
    issues: list[str] = []
    missing = REQUIRED_JSON_KEYS - set(d)
    if missing:
        return [f"Phase 12 JSON is missing keys: {sorted(missing)}"]
    rows = d["rows"]
    if d["n_rows"] != len(rows):
        issues.append(f"n_rows={d['n_rows']} but {len(rows)} rows present")
    done = sum(r.get("status") == "completed" for r in rows)
    failed = sum(r.get("status") == "failed" for r in rows)
    if done != d["n_completed"] or failed != d["n_failed"]:
        issues.append("completed/failed counts do not match row statuses")
    if len(d["full_results"]) != d["n_completed"]:
        issues.append("full_results entries != n_completed")
    return issues


def validate_csv_matches_json(df: pd.DataFrame, d: dict) -> list[str]:
    issues = []
    if not REQUIRED_ROW_COLS <= set(df.columns):
        return [f"CSV missing columns: {sorted(REQUIRED_ROW_COLS - set(df.columns))}"]
    if len(df) != d["n_rows"]:
        issues.append(f"CSV has {len(df)} rows, JSON has {d['n_rows']}")
    if int((df.status == "completed").sum()) != d["n_completed"]:
        issues.append("CSV and JSON disagree on completed-run count")
    return issues


# ---------------------------------------------------------------- frames
def runs_frame(d: dict) -> pd.DataFrame:
    """One row per attempted detector run, straight from the JSON ``rows`` (failures kept)."""
    df = pd.DataFrame(d["rows"])
    df["run_key"] = [f"{r.detector}_{'drift' if r.condition == 'drift' else 'no_drift'}_{r.seed}"
                     for r in df.itertuples()]
    return df


def metric_series(df: pd.DataFrame, checkpoint: str, metric: str) -> pd.Series:
    """Metric values with *undefined* (and failed-run) entries as NaN, never 0."""
    val, und = f"{checkpoint}_{metric}", f"{checkpoint}_{metric}_undefined"
    s = df[val].astype("float64")
    mask = df[und].fillna(True).astype(bool) if und in df else pd.Series(False, index=df.index)
    return s.where(~mask)


def detection_frame(d: dict) -> pd.DataFrame:
    recs = []
    for key, r in d["full_results"].items():
        for cp in CHECKPOINTS:
            x = r["detection"][cp]
            cm = x["confusion_matrix"]
            rec = dict(run_key=key, detector=x["detector_name"], condition=x["condition"],
                       seed=x["seed"], checkpoint=cp, n_samples=x["n_samples"],
                       n_episodes=x["n_episodes"], threshold=x["threshold"], **cm)
            for m in METRICS:
                v = x[m]
                rec[m] = np.nan if v.get("undefined") or v.get("value") is None else v["value"]
                rec[f"{m}_undefined"] = bool(v.get("undefined"))
            rec["flags_all_samples"] = (cm["tn"] == 0 and cm["fn"] == 0)
            recs.append(rec)
    return pd.DataFrame(recs)


def stability_frame(d: dict) -> pd.DataFrame:
    """Per-explanation-sample cosine / Jaccard (pre_drift reference vs later checkpoint)."""
    recs = []
    for key, r in d["full_results"].items():
        for field, cp in (("stability_vs_post_drift", "post_drift_pre_retrain"),
                          ("stability_vs_post_retrain", "post_retrain")):
            s = r[field]
            cos, jac = s["cosine"]["values"], s["jaccard"]["values"]
            for i, (c, j) in enumerate(zip(cos, jac)):
                recs.append(dict(run_key=key, detector=s["detector_name"], condition=s["condition"],
                                 seed=s["seed"], checkpoint=cp, sample=i, k=s["k"],
                                 cosine=np.nan if c is None else c,
                                 jaccard=np.nan if j is None else j))
    return pd.DataFrame(recs)


def stability_run_frame(d: dict) -> pd.DataFrame:
    """Per-run mean cosine/Jaccard at each measured checkpoint (as stored)."""
    recs = []
    for key, r in d["full_results"].items():
        for field, cp in (("stability_vs_post_drift", "post_drift_pre_retrain"),
                          ("stability_vs_post_retrain", "post_retrain")):
            s = r[field]
            recs.append(dict(run_key=key, detector=s["detector_name"], condition=s["condition"],
                             seed=s["seed"], checkpoint=cp, cosine=s["cosine"]["mean"],
                             jaccard=s["jaccard"]["mean"], n_invalid_cosine=s["cosine"]["n_invalid"],
                             n_samples=s["n_samples"], k=s["k"]))
    return pd.DataFrame(recs)


def attribution_matrix(d: dict, run_key: str, checkpoint: str) -> pd.DataFrame:
    e = d["full_results"][run_key]["explanations"][checkpoint]
    return pd.DataFrame(np.asarray(e["attributions"], dtype=float), columns=e["feature_names"])


def mean_abs_importance(d: dict, run_key: str, checkpoint: str) -> pd.Series:
    return attribution_matrix(d, run_key, checkpoint).abs().mean().rename(checkpoint)


def topk_features(importance: pd.Series, k: int) -> list[str]:
    return list(importance.sort_values(ascending=False).index[:k])


def labeled_episodes(d: dict, run_key: str) -> pd.DataFrame:
    """Labeled-anomaly episodes recorded by the Phase 12 evaluation (window-index space)."""
    r = d["full_results"][run_key]
    recs = []
    for cp in CHECKPOINTS:
        base = r["window_bounds"][cp]["start"]
        for ep in r["detection"][cp]["delay"]["episode_results"]:
            e = ep["episode"]
            recs.append(dict(checkpoint=cp, stored_start=e["start"], stored_end=e["end"],
                             window_start=base, detected=ep["detected"],
                             derived_global_start=base + e["start"], derived_global_end=base + e["end"]))
    return pd.DataFrame(recs)


def drift_events(d: dict) -> pd.DataFrame:
    df = runs_frame(d)
    keep = ["condition", "seed", "detector", "status", "error_type", "error_message",
            "t_event_window", "t_detect", "t_retrain", "secondary_monitor_status",
            "secondary_monitor_detection_index"]
    return df[keep]


# ---------------------------------------------------------------- phase 13
def load_validation_tables() -> dict[str, pd.DataFrame]:
    out = {}
    folder = abspath(P13_TABLES)
    if folder.exists():
        for p in sorted(folder.glob("*.csv")):
            out[p.stem] = pd.read_csv(p, encoding="utf-8")
    return out


def p_value(stats: pd.DataFrame, startswith: str) -> float | None:
    m = stats[stats.comparison.str.startswith(startswith)]
    return float(m.p_value.iloc[0]) if len(m) else None


# ---------------------------------------------------------------- dataset
def find_dataset(d: dict | None = None) -> Path | None:
    rel = (d or {}).get("series_relative_path", "data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv")
    cands = [ROOT / "data" / "raw" / rel, ROOT / "data" / rel, ROOT / "NAB" / rel,
             ROOT / "data" / "raw" / Path(rel).name, ROOT / "data" / "NAB" / rel, ROOT.parent / "NAB" / rel]
    return next((c for c in cands if c.exists()), None)


def find_labels() -> Path | None:
    for base in [ROOT / "data" / "raw", ROOT / "data", ROOT / "NAB", ROOT / "data" / "NAB", ROOT.parent / "NAB"]:
        for sub in ("labels/combined_windows.json", "combined_windows.json"):
            if (base / sub).exists():
                return base / sub
    return None


def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"timestamp", "value"} <= set(df.columns):
        raise ArtifactError(f"{path.name} lacks timestamp/value columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_label_windows(label_path: Path, series_rel: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    key = series_rel if series_rel in data else next((k for k in data if k.endswith(Path(series_rel).name)), None)
    if key is None:
        return []
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in data[key]]


# ---------------------------------------------------------------- reproducibility
def parse_requirements(text: str) -> tuple[dict[str, str], str | None]:
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", text, flags=re.M))
    py = re.search(r"Python:\s*([0-9.]+)", text)
    return pins, (py.group(1) if py else None)


def sha256_file(rel: str) -> str | None:
    p = abspath(rel)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def expected_hashes() -> dict[str, str]:
    """Parse EXPECTED_SHA256 out of scripts/verify_final_package.py (text only; never imported)."""
    txt = read_text("scripts/verify_final_package.py") or ""
    return dict(re.findall(r'"([\w/.\-]+)":\s*"([0-9a-f]{64})"', txt))


def list_artifacts() -> list[str]:
    out = []
    for base in ("artifacts", "config"):
        for p in sorted(abspath(base).rglob("*")) if abspath(base).exists() else []:
            if p.is_file() and p.suffix.lower() in {".json", ".csv", ".yaml", ".png"}:
                out.append(p.relative_to(ROOT).as_posix())
    return out

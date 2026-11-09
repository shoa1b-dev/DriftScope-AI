"""Read-only integrity check of the FINAL package.

 This script:
  * does NOT import anything from src/,
  * does NOT run any detector, SHAP, River, or the experiment,
  * does NOT write or modify any file.

It only reads the stored result artifacts and the Phase 13 deliverables and
checks that they are present, internally consistent, and unchanged.

Usage (from the project root):
    python scripts/verify_final_package.py
Exit code 0 = all checks passed; 1 = at least one check failed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# SHA-256 of the authoritative files as audited (also listed in docs/FINAL_AUDIT.md).
EXPECTED_SHA256 = {
    "artifacts/phase12/real_experiment_results.json":
        "79acb5f9b9afc85493f6ec1fc7e6835096ab46a6d11275725f96aec8f7550e4f",
    "artifacts/phase12/real_experiment_results.csv":
        "2325ab956685aee28fa2a100a7450b966d51059774487eef8624c7019ab3c65e",
    "docs/PHASE13_FINAL_RESULTS.md":
        "098c569d941c6795452666ecc589454a6a873f13300d3db3e3c6f688abe07588",
}
TABLES = ["A_drift_condition_summary", "B_no_drift_condition_summary", "C_detector_a_summary",
          "D_detector_b_summary", "E_successful_drift_runs_detail", "F_drift_detection_failures",
          "G_statistical_tests"]
FIGURES = ["fig1_detection_f1_by_checkpoint", "fig2_explanation_stability_by_detector",
           "fig3_stability_recovery_slopegraph", "fig4_adwin_detection_timing"]

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("PASS  " if ok else "FAIL  ") + msg)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    # 1. Authoritative files unchanged
    for rel, want in EXPECTED_SHA256.items():
        p = ROOT / rel
        check(p.exists() and sha256(p) == want, f"unchanged SHA-256: {rel}")

    # 2. Real results: counts, error types, uniqueness, JSON == CSV
    jpath = ROOT / "artifacts/phase12/real_experiment_results.json"
    cpath = ROOT / "artifacts/phase12/real_experiment_results.csv"
    data = json.loads(jpath.read_text(encoding="utf-8"))
    rows = data["rows"]
    check(len(rows) == 20 and data["n_rows"] == 20, "JSON has 20 attempted rows")
    check(sum(r["status"] == "completed" for r in rows) == 16 and data["n_completed"] == 16, "16 completed rows")
    failed = [r for r in rows if r["status"] == "failed"]
    check(len(failed) == 4 and data["n_failed"] == 4, "4 failed rows")
    check({r["error_type"] for r in failed} == {"DriftNeverDetectedError"},
          "all failures are DriftNeverDetectedError (no RiverUnavailable/SHAPUnavailable dry-run errors)")
    check(sorted((r["seed"], r["detector"]) for r in failed)
          == [(101, "detector_a"), (101, "detector_b"), (303, "detector_a"), (303, "detector_b")]
          and all(r["condition"] == "drift" for r in failed),
          "failures = drift condition, seeds 101 & 303, both detectors")
    keys = {(r["condition"], r["seed"], r["detector"]) for r in rows}
    check(len(keys) == 20, "20 unique (condition, seed, detector) keys")
    check(sorted({r["seed"] for r in rows}) == [13, 42, 101, 202, 303], "seeds = [13, 42, 101, 202, 303]")
    check(all(r["status"] == "completed" for r in rows if r["condition"] == "no_drift"), "all 10 no_drift rows completed")

    csv_lines = cpath.read_text(encoding="utf-8").strip().splitlines()
    check(len(csv_lines) == 21, "CSV has header + 20 data rows")
    try:
        import pandas as pd
        c = pd.read_csv(cpath)
        same = (c.groupby("status").size().to_dict() == {"completed": 16, "failed": 4}
                and {(a, b, d) for a, b, d in zip(c.condition, c.seed, c.detector)} == keys)
        check(same, "CSV and JSON describe the same 20 runs (16 completed / 4 failed)")
        ok = c[c.status == "completed"]
        check(int(ok.post_retrain_recall_undefined.sum()) == 16 and int(ok.post_retrain_f1_undefined.sum()) == 16,
              "post-retrain recall and F1 flagged undefined in 16/16 completed rows")
        check(int(ok.post_retrain_precision_undefined.sum()) == 0 and (ok.post_retrain_precision == 0).all(),
              "post-retrain precision defined and 0.0 in 16/16 completed rows")
        check((ok.cosine_vs_post_retrain_mean < ok.cosine_vs_post_drift_mean).all(),
              "cosine stability lower post-retrain than post-drift in 16/16 completed rows")
        check(ok.cosine_vs_post_drift_mean.min() >= 0.991, "min post-drift cosine >= 0.991")
    except ImportError:
        print("SKIP  pandas not installed: CSV content checks skipped")

    # 3. Phase 13 deliverables present
    for t in TABLES:
        check((ROOT / f"artifacts/phase13/tables/{t}.csv").exists(), f"table present: {t}.csv")
    for f in FIGURES:
        check((ROOT / f"artifacts/phase13/figures/{f}.png").exists(), f"figure present: {f}.png")

    # 4. Report wording corrections present (whitespace-normalised search)
    rep = re.sub(r"\s+", " ", (ROOT / "docs/PHASE13_FINAL_RESULTS.md").read_text(encoding="utf-8"))
    check("The observed difference in explanation stability is strongly associated with detector identity "
          "in this two-detector experiment" in rep, "report contains the detector-identity wording correction")
    check("Non-independence caveat" in rep, "report contains the non-independence caveat")

    # 5. Core project files present
    for rel in ["config/experiment.yaml", "pyproject.toml", "scripts/run_phase12_experiment.py",
                "src/experiments/experiment_runner.py", "tests/unit/test_experiment_runner.py", ".gitignore",
                "requirements-lock.txt"]:
        check((ROOT / rel).exists(), f"present: {rel}")

    # 6. Things that must NOT be in the repository
    for rel in [".venv", "build", ".idea", "NAB"]:
        check(not (ROOT / rel).exists(), f"not present (should be excluded): {rel}/")
    check(not any(ROOT.rglob("__pycache__")), "no __pycache__ directories")

    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed" + ("" if n_fail == 0 else f" — {n_fail} FAILED"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

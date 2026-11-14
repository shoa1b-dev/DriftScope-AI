# Final Reconciliation Audit

## 1. Verdict

| Question | Answer |
|---|---|
| Real experiment results present and authentic? | **Yes** — in the Phase 12 archive (not the Phase 13 archive). 20 rows / 16 completed / 4 `DriftNeverDetectedError`; JSON and CSV agree cell-for-cell. |
| Is the dry-run result (`RiverUnavailableError`/`SHAPUnavailableError`) present anywhere? | **No** — only described in a stale doc (`PHASE12_REAL_EXPERIMENT.md`, now bannered). |
| Phase 13 statistics reproducible from the raw results? | **Yes** — Table G recomputed independently (all p-values match); Tables A–F match the raw results. |
| Both Phase 13 wording corrections present? | **Yes** — detector-identity wording and non-independence caveat. |
| Phase 12.1 `collect_environment_metadata()` patch present? | **No.** Not in either archive. Package has 343 tests, not 347. |
| Scientific caveats the report does not state? | **Yes** — see section 4 (K2, K3, K4 matter most). |

## 2. Reconciliation matrix

| Component | Phase 12 archive | Phase 13 archive | Authoritative | Status |
|---|---|---|---|---|
| Source code (`src/`) | present (`build/lib/src` is an identical stale copy) | absent | Phase 12 `src/` | Verified |
| Tests | 343 test functions (unit + integration) | absent | Phase 12 `tests/` | Verified (343 ran, OK, 6 skipped) |
| Config | `config/experiment.yaml` | absent | Phase 12 | Verified vs locked design (seeds, 2419, +3.85, 100/100 windows, [4,2,4]); contains stale pre-experiment comments |
| Runner script | `run_phase12_experiment.py` (no metadata function) | absent | Phase 12 | Verified; Phase 12.1 patch **missing** |
| Dependency declaration | `pyproject.toml` (unpinned) + bundled `.venv` | absent | `pyproject.toml` + new `requirements-lock.txt` | Pins reconstructed from venv metadata |
| `.gitignore` | none at project root | none | new `.gitignore` | Missing → added |
| Phase 1–12 docs | present | absent | Phase 12 `docs/` | `PHASE12_REAL_EXPERIMENT.md` was **stale** → bannered |
| Result JSON | **real** (20 rows, 286,310 bytes) | absent | Phase 12 | Verified |
| Result CSV | **real** (20 rows) | absent | Phase 12 | Verified |
| Phase 13 report | absent | present | Phase 13 | Verified; unmodified |
| Phase 13 figures (4) | absent | present | Phase 13 | Verified (all four visually inspected) |
| Phase 13 tables A–G | absent | present | Phase 13 | Verified (see K4 on A–D) |
| README | stale "Phase 1" README | newer, but wrong in places | Phase 13 README, corrected | Conflicting → rewritten |
| Environment metadata in results | absent | claimed present | none | Missing (as the report itself states for the JSON) |
| Phase 13 analysis/figure script | absent | absent | none | Missing (K5) |
| Approved proposal .docx/.pdf | absent | absent | none | Missing (referenced by README) |
| `.idea/`, `.venv/`, `build/`, `*.egg-info/`, `__pycache__/` | present | absent | excluded | Not for GitHub |

## 3. Evidence for the key findings

**Real results (SHA-256):**

```
79acb5f9b9afc85493f6ec1fc7e6835096ab46a6d11275725f96aec8f7550e4f  artifacts/phase12/real_experiment_results.json
2325ab956685aee28fa2a100a7450b966d51059774487eef8624c7019ab3c65e  artifacts/phase12/real_experiment_results.csv
098c569d941c6795452666ecc589454a6a873f13300d3db3e3c6f688abe07588  docs/PHASE13_FINAL_RESULTS.md
```

- 20 unique (condition, seed, detector) keys; seeds [13, 42, 101, 202, 303]; 10 `drift` + 10 `no_drift`.
- 4 failures: drift, seeds 101 and 303, both detectors, all `DriftNeverDetectedError`.
- Post-retrain: precision defined and 0.0 in 16/16; recall and F1 flagged undefined in 16/16.
- Cosine(pre-drift vs post-drift) min 0.9912; cosine lower post-retrain than post-drift in 16/16.
- Table G recomputed with SciPy: Wilcoxon p = 0.03125 / 0.0625 / 0.001953 / 0.001953; Mann-Whitney drift-vs-no-drift U=40, p=0.3132; detector A-vs-B U=0, p=0.0001554; Spearman rho=0.9411, p=0.0051.
- Dataset (from `NAB.zip`): 4,032 rows, 5-minute interval, 0 missing, 0 duplicate timestamps, 3 labeled windows; `int(0.6*4032)=2419`; `t_event_window=2410` is 2419 minus the 10-sample feature window (config and results agree).
- Dependency evidence (bundled venv): Python 3.11.9, River 0.26.1, SHAP 0.51.0, scikit-learn 1.9.1, NumPy 2.4.6, pandas 3.0.6.

## 5. What was verified, and how

| Check | Method | Result |
|---|---|---|
| Archive inventories | `unzip -l`, filtered | Done; Phase 12 zip has 18,415 `.venv` entries, Phase 13 zip 19 entries |
| Result JSON/CSV content | Parsed with Python/pandas | Section 3 |
| Table G | Recomputed with SciPy | All 9 rows match |
| Tables A–F | Recomputed from raw rows | A–D, E, F match |
| Figures 1–4 | Viewed | Consistent with data |
| Wording corrections | Whitespace-normalised search | Both present |
| Test suite | `python -m unittest discover -s tests -v` on a scratch copy | `Ran 343 tests ... OK (skipped=6)`; results files unchanged afterwards |
| Final package | `scripts/verify_final_package.py` | 42/42 checks passed |
| **Not done** | — | Real-River/real-SHAP tests (not installed); re-running the experiment; independent review of the scientific design |

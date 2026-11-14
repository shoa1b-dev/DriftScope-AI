---

# Phase 12 — Primary Real NAB Experiment: Orchestration Built and Dry-Run Validated

## 1. Exact experiment matrix (as configured, using the locked Phase 11 values)

2 conditions (`drift`, `no_drift`) × 5 seeds (13, 42, 101, 202, 303) × 2 detectors = **20 detector-condition runs**, all against `data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv`, `t_event=2419`, `window_size=10`, `drift_offset=3.85`, `retrain_window_size=100`, `post_retrain_window_size=100` — read directly from `config/experiment.yaml`, never hardcoded separately in the script.

## 2. Exact number of successful/failed runs 
**0 successful, 20 failed** — all 20 rows structurally correct, honestly typed:
- 10 rows (`drift` condition, all 5 seeds × both detectors): `RiverUnavailableError` — fails at real `AdwinDriftMonitor()` construction, before SHAP is ever reached.
- 10 rows (`no_drift` condition, all 5 seeds × both detectors): `SHAPUnavailableError` — `no_drift` never touches the drift monitor at all, so it correctly proceeds all the way to the real SHAP explanation stage before failing there.

This split is itself a useful structural confirmation: it proves the `no_drift` condition genuinely never depends on `river` (consistent with the Phase 8.1 design), and that both dependency checks are reached at exactly the point the architecture predicts, not swallowed or reached prematurely.

## 3–5. Detection / Stability / SHAP execution summaries

**None available** — every metric field in every row is `null`, exactly as designed (`flatten_detector_result` never fabricates a value for a failed run). No detection, stability, or SHAP result of any kind exists yet.

## 6. Invalid attribution counts

Not applicable — SHAP was never successfully invoked.

## 7. Warnings/errors

Two distinct, correctly-typed errors as described in §2 — both expected given the known environment constraint, not unexpected failures.

## 8. Exact test results

```
python3 -m unittest discover -s tests -v
Ran 343 tests in 25.636s
OK (skipped=6)
```
**337 passed, 6 skipped (unchanged), 0 failed.** New this phase: `test_build_binary_labels.py` (5 tests) and `test_phase12_orchestration.py` (7 tests, using monkeypatched `run_experiment` to simulate mixed success/failure — proving the script's own bookkeeping is correct independent of real dependencies).

## 9. Generated artifacts

- `src/data/nab_inspection.py` — added `build_binary_labels_from_windows` (converts real labeled windows into a binary ground-truth array; verified against the real file: 405 total anomalous points, 270 pre-event, 135 post-event)
- `scripts/run_phase12_experiment.py` — the real orchestration script
- `tests/unit/test_build_binary_labels.py`, `tests/unit/test_phase12_orchestration.py`
- `artifacts/phase12/real_experiment_results.json`, `artifacts/phase12/real_experiment_results.csv` — **from this dry run**, 20 rows, all `status: failed`, correctly typed errors, no fabricated metrics
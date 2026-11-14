# Phase 3 — Drift Injection, Drift Monitoring, Retraining Trigger

## Files created / modified

| File | Status |
|---|---|
| `src/data/drift_injection.py` | Implemented and verified (synthetic data) |
| `src/monitoring/drift_monitor.py` | Implemented; ADWIN/Page-Hinkley wrappers unverified (river unavailable) |
| `src/monitoring/retrain_trigger.py` | Implemented and verified (synthetic data + mock monitor) |
| `tests/support/mock_drift_monitor.py` | Test fixture only — fake detection rule, never used in production code |
| `tests/unit/test_drift_injection.py` | New, 11 tests, all passing |
| `tests/unit/test_drift_monitor.py` | New, 9 tests (7 run + pass, 2 real-ADWIN/PH tests skipped) |
| `tests/unit/test_retrain_trigger.py` | New, 11 tests, all passing |
| `config/experiment.yaml` | Annotated drift_monitor section with verified/unverified status |

## Design decisions and why

**Drift injection is configurable, not hardcoded**, because the approved proposal explicitly defers the magnitude and injection point to real dataset inspection (Phase 10 §1 / your instruction). The mechanism itself (`new = old * scale + offset` from `t_inject` onward) is deterministic — no randomness in the arithmetic — which is what makes "reproducible behavior" trivial to guarantee: identical parameters always produce identical output (tested directly in `test_identical_params_produce_identical_output`).

**ADWIN is primary** because that's your explicit, confirmed decision, matching the CAIN 2024 anchor study's use of adaptive-windowing drift detection. **Page-Hinkley is secondary/robustness-check only** — implemented with the exact same interface as ADWIN (via the shared `DriftMonitorBase`), but nowhere does the code or config treat it as a substitute; both classes' docstrings state this explicitly, and any future results reporting must carry the same labeling forward.

**The monitor interface is independent of detector classes** — `DriftMonitorBase.update(value, index)` only ever sees a scalar score, never a `BaseDetector` instance. This is what lets the identical monitor class watch Detector A's or Detector B's score stream unmodified, which is what makes the RO4 two-detector comparison methodologically fair (per the approved proposal's Phase 11 §6 justification).

**Detection and retraining are separate events, enforced in code, not just in naming.** `RetrainTrigger.process(...)` only ever sets `detection_index`. A separate, explicit `confirm_retrain(index)` call is required to set `retrain_index` — and it raises if called before detection, or called twice for the same detection. This makes the three-timestamp discipline (`t_inject`/`t_detect`/`t_retrain`) a structural guarantee rather than a convention someone could accidentally violate later.

**`retrain_window_size` is not guessed.** `RetrainTrigger.get_retrain_slice(...)` raises `ValueError` if `retrain_window_size` is still `None` — there is no fallback default. This is tested directly (`test_none_retrain_window_size_raises_on_slice_request`).

## What was actually verified 

**Verified (real execution, synthetic data):**
- Drift injection: offset-only, scale-only, combined, reproducibility, all 5 invalid-parameter cases.
- Retrain trigger: detection→pending→confirm-retrain flow, the detection/retrain index separation, double-confirm rejection, confirm-before-detection rejection, window-size guard, reset behavior.
- Drift monitor **interface contract** (via `MockDriftMonitor`, a fake threshold rule): not-detected-initially, detects-once, returns-True-only-on-transition (this is where a real bug was caught and fixed — see below), reset, re-detect-after-reset.
- The honest-failure path: `AdwinDriftMonitor()`/`PageHinkleyDriftMonitor()` correctly raise `RiverUnavailableError` rather than silently substituting a fake.

## Still not implemented

Checkpoint framework, SHAP explanation generation, stability metrics, detection-performance evaluation, full experiment runner — per your explicit restriction, none of these were started this phase.

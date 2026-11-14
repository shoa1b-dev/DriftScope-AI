# Phase 8 — Experiment Runner and End-to-End Orchestration

## Files created / modified

| File | Status |
|---|---|
| `src/experiments/experiment_config.py` | New — implemented and verified |
| `src/experiments/experiment_runner.py` | New — implemented and verified (synthetic orchestration); real ADWIN/Page-Hinkley/SHAP execution unverified |
| `src/experiments/result_writer.py` | New — implemented and verified |
| `tests/unit/test_experiment_config.py` | New, 12 tests |
| `tests/unit/test_experiment_runner.py` | New, 23 tests |
| `tests/unit/test_result_writer.py` | New, 5 tests |
| `docs/PHASE8_EXPERIMENT_RUNNER.md` | This file |

## Exact test command

```
python3 -m unittest discover -s tests -v
```

## Exact counts

**Collected: 225 · Passed: 221 · Skipped: 4 · Failed: 0**

(4 skips unchanged: 2 real-ADWIN/Page-Hinkley from Phase 3, 2 real-SHAP from Phase 5.)

## Fully verified synthetic functionality

- **Complete drift-condition run**, both detectors, all 17 stages, using an injected fake monitor and fake SHAP compute function.
- **Complete no-drift-condition run** — confirmed it never invokes a monitor factory at all (a spy factory records zero calls), `t_detect` stays `None`, event source is correctly labeled `"matched_arbitrary_point"`, secondary-monitor status is `"not_run"` (not merely unavailable — genuinely never attempted), and recall is correctly flagged `undefined` rather than reported as a real 0.0 miss.
- **Both fixed samples (background + evaluation) proven pre-drift-only**: every source index in both samples is checked to be `< t_event_window`, and both detectors are proven to share the identical sample (same signature), not independently redrawn ones.
- **Correct checkpoint names/completeness**: all three checkpoint labels present in both `explanations` and `detection` dicts for both detectors (implicitly also proves correct ordering, since `CheckpointManager` would have raised `CheckpointOrderError`/`DuplicateCheckpointError` otherwise — the run would not have completed at all if ordering were wrong).
- **Reproducibility**: identical seed + identical data + identical fakes → identical `t_detect` and identical confusion-matrix counts. Different approved seeds correctly recorded and distinguished.
- **Error handling**: missing data (`None` series/labels, length mismatch, non-binary labels), too-short pre-drift region, insufficient post-retrain data (constructed via a deterministic countdown fake to hit the exact boundary), drift never detected (with the failing detector's name confirmed present in the error message).
- **Dependency-blocked paths raise correctly, not silently**: omitting the monitor factory for a drift-condition run raises `RiverUnavailableError`; omitting the SHAP compute function raises `SHAPUnavailableError` — both from the real underlying modules, not re-wrapped or swallowed.
- **No-fabrication checks**: injected fake SHAP values traced through unmodified to the final result; detection episode counts shown to differ meaningfully based on where the synthetic anomaly was placed (not a fixed placeholder).
- **Result serialization**: one JSON file per detector, deterministic naming (`{detector}_{condition}_{seed}.json`), round-trip read-back, and a regression check that no live model object ever appears in the written file.

## Bugs found and fixed

One, in a **test's** constructed scenario, not the runner: `test_insufficient_post_retrain_data` initially tried to combine "passes the initial too-short check" with "leaves no room after retrain" using immediate detection — algebraically impossible given `window_size >= 2` and `stride = 1` (worked through directly: the two conditions contradict each other when detection is immediate). Fixed by adding a small local `_CountdownDriftMonitor` test fake that detects on a precisely chosen call number, giving direct control over exactly which index triggers detection without relying on real detector score magnitudes. Re-ran the full suite after the fix. The runner module itself had zero bugs — every test targeting `experiment_runner.py`'s actual logic passed on first execution (confirmed via the two standalone smoke-test scripts run before the formal test suite was even written).

## Design decisions needing your approval

1. **One `run_experiment` call = one condition + one seed, but BOTH detectors** — matching the stage list's literal ordering ("Train Detector A" then "Train Detector B" as sequential stages within one run), while keeping the two detectors' internal processing fully independent (separate `CheckpointManager`, separate model instances, separate monitors). Flagging because the alternative (one call per detector) was also structurally possible and would have meant computing/injecting drift twice for the same experiment.
2. **`config.t_event` serves two roles** — the raw drift-injection point for "drift", and the matched-arbitrary-point for "no_drift" — reusing one index rather than drawing a second independent point for the control, specifically to keep both conditions' checkpoint timelines numerically aligned per the proposal's control design.
3. **Retrain slice is clamped to never dip before `t_event_window`** for the drift condition (`retrain_slice_start = max(t_event_window, t_detect - retrain_window_size + 1)`), ensuring the retrain data is always genuinely post-drift, even if `retrain_window_size` is larger than the gap between drift onset and detection.
4. **`t_retrain = t_detect + 1`** — a documented, simple convention for "immediate" retraining that still keeps the two indices numerically distinct, per your Phase 4 requirement that detection and retraining are always separate.
5. **Page-Hinkley (secondary) failure handling differs from ADWIN (primary)**: if the primary monitor factory is unavailable/fails, the whole run fails (correct — it drives real checkpoint/retrain timing). If the *secondary* monitor is unavailable (`RiverUnavailableError`), this is caught internally and recorded as `secondary_monitor_status="unavailable"` rather than failing the whole run — because Page-Hinkley is explicitly non-blocking per your Phase 3 decision. This is a deliberate, narrow exception to "don't silently swallow errors," scoped only to the one component that was already established as optional/non-gating.
6. **Detection windows per checkpoint**: `pre_drift` evaluates over windows `[0, t_event_window)`, `post_drift_pre_retrain` over `[t_event_window, t_detect]`, `post_retrain` over `[t_retrain, end)` — i.e., "the rest of the available data" for the post-retrain window, not a fixed-size window matched to the others. This was the simplest well-defined choice for this integration phase; a fixed, matched post-retrain window size is a natural refinement once real NAB data determines what's actually available.

## Interface incompatibilities discovered

None. All Phase 2–7 module interfaces were used exactly as previously built and tested — no existing module needed a signature change, and the one place I double-checked against assumption (`preprocessing.py`'s exact function signatures, since that file predates this visible conversation) matched what the runner needed once verified directly against the source.

## Confirmation: no fabricated results

No real NAB data, no real ADWIN/Page-Hinkley detection, and no real SHAP attribution appear anywhere in this phase. Every number in every test is either synthetic and hand-constructed, or a direct, traceable pass-through of an injected fake explicitly labeled as fake. The two tests that exercise the *real* dependency paths only confirm that they fail honestly (raise the documented error) — they do not, and cannot, claim real execution occurred.

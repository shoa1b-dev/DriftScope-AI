# Phase 8.1 — Review of Phase 8 Runner Design Decisions

## Decision 1 — Page-Hinkley unavailability: KEPT, with a structural clarity improvement

**Rationale:** ADWIN is the confirmed primary drift detector — it drives the actual checkpoint/retrain timing, so its unavailability genuinely invalidates the run and must fail loudly (`RiverUnavailableError` propagates, unchanged). Page-Hinkley is confirmed secondary/robustness-check only — it never gates anything, so treating its unavailability as run-fatal would be inconsistent with that confirmed status. This behavior is unchanged in substance, but the result structure is improved: `secondary_monitor_status`/`secondary_monitor_detection_index` (two separate fields) are replaced with a single `SecondaryMonitorResult(status, detection_index)` object, so the two can never be read independently and misread. `status` is exactly one of `"not_run"` (no_drift — the concept doesn't apply), `"ran"` (real execution occurred — `detection_index` is either a real index or `None` meaning a genuine "didn't detect" finding), or `"unavailable"` (dependency missing — `detection_index` is always `None`, for an entirely different reason than the "ran, no detection" case). Five new tests directly prove all three states are distinguishable, including a contrast test proving ADWIN's unavailability is *not* caught the same way.

## Decision 2 — Reusing `t_event`: KEPT, with metadata now fully surfaced

**Rationale unchanged:** reusing the same index keeps drift and no-drift checkpoint timelines numerically aligned, as the approved proposal's control design requires; no second random point is drawn, and `no_drift` never touches `drift_injection.py`. **What changed:** in Phase 8, the actual per-checkpoint `index_source` labels were computed but never surfaced outside the internal `CheckpointRecord` objects — a real gap, since nothing in the final result actually proved the labeling was correct beyond one summary string. Fixed by adding `DetectorRunResult.checkpoint_events`, a dict of `{event_index, index_source}` per checkpoint, now included in the JSON metadata. New regression tests confirm `no_drift` never produces `"adwin_detection"` or `"retrain_trigger"` anywhere, with a direct contrast test proving the *same* code path correctly *does* use `"adwin_detection"` under `"drift"` — so the distinction is proven real, not just untested.

## Decision 3 — Retraining slice boundary: formula clarified, base logic now genuinely reused

**Exact formula (drift condition):** the base window is obtained by calling `RetrainTrigger.get_retrain_slice` itself (feeding it index positions as its "values" so its own arithmetic determines the base start) — this was a real gap in Phase 8: the runner had re-implemented the same formula independently instead of reusing the module, which is exactly what Phase 8's own instructions said not to do. Fixed: the base start now comes directly from `RetrainTrigger.get_retrain_slice`. An **additional** clamp, `max(t_event_window, base_start)`, is applied on top — reported explicitly as a deliberate extension (the generic trigger utility has no concept of "which points are pre-event"; that's an experiment-level concept). **Inclusive/exclusive:** the slice is `[clamped_start : t_detect]` inclusive on both ends (Python slice `[start:t_detect+1]`). **No future data:** the slice never includes anything after `t_detect`. **No forbidden pre-event data:** the clamp guarantees `clamped_start >= t_event_window` always, for the drift condition. **No-drift condition** intentionally mirrors the same base formula directly rather than instantiating `RetrainTrigger` (which requires a prior detection event its `confirm_retrain` contract demands, and the control correctly never has one) — and is **not** clamped to `t_event_window`, since there is no pre/post-event distinction for a genuine no-drift control; this asymmetry is now explicitly tested (`test_no_drift_retrain_slice_may_dip_before_event_window`). New tests cover: `retrain_window_size` far larger than available data (clamp holds), a short-window case, and the no-drift contrast case.

## Decision 4 — `t_retrain = t_detect + 1`: convention documented, boundary case tested

**Documented explicitly** (module docstring) as a synthetic orchestration convention modeling "immediate retrain," not a scientific claim about real detection-to-retrain latency — the real latency, once implemented against real data, may differ. New tests cover: detection at the final valid scanned index (raises `InsufficientPostRetrainDataError`, with the message now stating explicitly that the event is "NOT silently moved") and detection with a valid next index (succeeds, `t_retrain == t_detect + 1` confirmed).

## Decision 5 — Post-retrain evaluation window: recommend Option B in principle, implemented as a configurable fallback

**Recommendation: Option B (fixed, comparable window)** is the methodologically correct target — it's actually what the *already-approved* proposal specifies (Phase 13 §13: post-retrain windows "matched to W where series length allows"), so Option A (all remaining data, as Phase 8 shipped) was a Phase-8 simplification that diverged from the approved design, not a neutral default. Option A creates a real comparability problem: different seeds/detectors would get different-sized post-retrain windows purely because `t_detect` varies, which weakens the paired Wilcoxon comparisons the Testing & Evaluation Plan already locked in. **Implemented:** `config.post_retrain_window_size: int | None = None` — when set, the window is `[t_retrain, min(t_retrain + size, end))`, clamped (not erroring) when the requested size exceeds available data, and rejected only if the result would be empty. When left `None` (the current default), the runner falls back to all-remaining-data, explicitly documented as a fallback, not the preferred design — no numeric value is chosen here, satisfying "do not select a final numeric value yet." `DetectorRunResult.window_bounds` now records the exact `{start, end, n_samples}` actually used for **all three** checkpoints, not just post-retrain.

## Decision 6 — Both detectors in one runner call: confirmed correctly isolated, now regression-tested

No code changes were needed — `detector_a`/`detector_b` were already fully separate objects/managers/results in Phase 8. Three new tests make this an explicit, checked guarantee rather than an implicit property: object-identity distinctness between the two `DetectorRunResult`s and their sub-results, every result field's own `detector_name`/similar field matching the detector it belongs to (no cross-labeling), and a direct check that the two detectors' `model_type` differ (`IsolationForest` vs. `MLPRegressor`) as a concrete sanity anchor.

## Files changed

`src/experiments/experiment_config.py` (new `post_retrain_window_size` field + validation), `src/experiments/experiment_runner.py` (rewritten: `SecondaryMonitorResult`, `checkpoint_events`, `window_bounds`, reused `get_retrain_slice`-based clamp, post-retrain window logic), `tests/unit/test_experiment_runner.py` (existing test updated for renamed fields; 22 new tests added across six new test classes), `docs/PHASE8_1_RUNNER_DESIGN_REVIEW.md` (this file).

## Exact test command and counts

```
python3 -m unittest discover -s tests -v
```

**Collected: 244 · Passed: 240 · Skipped: 4 · Failed: 0**

## Bugs found and fixed

1. **Real integration gap (not a test bug):** the retrain-slice base formula was independently re-implemented in the runner rather than reusing `RetrainTrigger.get_retrain_slice`, contrary to Phase 8's own "do not duplicate their logic" instruction. Fixed by genuinely calling the existing module's method for the base window.
2. **Missing metadata surface (not a test bug):** per-checkpoint `index_source` labels existed internally but were never exposed in `DetectorRunResult`'s public result/metadata — meaning Decision 2's "never fabricate a drift event for the control" guarantee, while structurally enforced by Phase 4's `CheckpointManager`, wasn't actually *visible* or independently testable at the runner's own output level. Fixed by adding `checkpoint_events`.
3. **Test-only bug:** one new test asserted a lowercase substring ("not silently moved") against an error message that (correctly) used "NOT" in caps for emphasis — a case-sensitivity mismatch in my own test, not a module defect. Fixed the assertion.

## Remaining methodological concerns

- `post_retrain_window_size` remains unset (`None`) by default, so real experiments will still use the Option A fallback until a real value is chosen post-NAB-inspection — this is correct per your instruction not to select a value yet, but it means Decision 5's methodological concern (cross-seed comparability) is not yet *resolved* for a real run, only *made configurable* so it can be resolved once real data is available. Worth remembering to actually set this field before the first real multi-seed experiment.
- The no-drift condition's retrain slice is deliberately unclamped relative to the event boundary (Decision 3) — correct given no-drift has no drift event to protect against leaking around, but this asymmetry between the two conditions' slice logic is subtle enough that it's worth keeping the explicit contrast test (`test_no_drift_retrain_slice_may_dip_before_event_window`) as a permanent guard against someone "fixing" it into symmetry later without realizing why it's intentionally asymmetric.

## Confirmation

No real NAB data, no real River execution, no real SHAP execution, and no fabricated result were produced anywhere in this review. All changes were exercised exclusively through synthetic data and injected fakes; the two tests targeting real-dependency paths (`RiverUnavailableError` propagation for ADWIN) confirm only that the failure path is correct, not that real execution occurred.

# Phase 4 — Checkpoint Framework and Leakage-Safe Explanation Sample

## Files created / modified

| File | Status |
|---|---|
| `src/explanation/fixed_sample.py` | Implemented and verified (synthetic data) |
| `src/checkpoints/checkpoint_manager.py` | Implemented and verified (synthetic data, both real detector classes) |
| `tests/unit/test_fixed_sample.py` | New, 13 tests, all passing |
| `tests/unit/test_checkpoint_manager.py` | New, 15 tests, all passing |
| `docs/PHASE4_CHECKPOINTS.md` | This file |

## Why the explanation sample is fixed, and how leakage is prevented

The approved proposal (Phase 11 §13) requires the same evaluation points to be explained at all three checkpoints, drawn only from the pre-drift region — otherwise a stability change could reflect *different explained data* rather than a *changed model*. Two layers enforce this, not just one:

1. **Structural slicing**: `build_fixed_sample_from_full_series(all_features, t_inject, ...)` slices to `all_features[:t_inject]` *before* sampling — even if a caller passes the full pre+post-drift array, post-drift rows are never in the pool being sampled from. Tested directly with a synthetic array where pre-drift rows are `0.0` and post-drift rows are `999.0`: the resulting sample contains only zeros.
2. **Enforced immutability**: the sample's `features` array is marked read-only (`setflags(write=False)`) at construction. Attempting to mutate it raises `ValueError` — verified directly, not just documented as a convention.

## How model snapshots are isolated

`CheckpointManager.capture(...)` stores `copy.deepcopy(detector)`, not a reference. This was tested directly for **both** detector classes: capture a checkpoint, then call `.fit(...)` again on the *live* detector with very different data (simulating retraining), then confirm the stored snapshot's `anomaly_score()` output is bit-identical to what it was at capture time — while the live detector's output has genuinely changed. Both `test_detector_a_snapshot_survives_retraining` and `test_detector_b_snapshot_survives_retraining` pass.

## How the three checkpoints are represented

`VALID_CHECKPOINTS = ("pre_drift", "post_drift_pre_retrain", "post_retrain")`, enforced in strict order: capturing a checkpoint before its required predecessor(s) raises `CheckpointOrderError`; capturing the same checkpoint twice raises `DuplicateCheckpointError`; checking completeness before all three exist raises `IncompleteCheckpointsError` via `require_complete()` (or can be checked non-destructively via `.missing_checkpoints`/`.is_complete`).

## How the no-drift control is handled

The manager does **not** hardcode what a "post_drift_pre_retrain" checkpoint means for the `no_drift` condition, and does not invent a drift event for it. Instead, any checkpoint capture that supplies a non-`None` `event_index` **must** also supply `extra_metadata["index_source"]` — this is a checked contract (`UnlabeledEventIndexError` if missing), not a documentation note someone could forget. For the `drift` condition, callers are expected to pass `"index_source": "adwin_detection"`; for the `no_drift` control, `"index_source": "matched_arbitrary_point"` — both conditions use the same three checkpoint names (for aligned, comparable trajectories, per the approved proposal's control design), but the metadata always makes explicit whether an index came from real detection or a matched point. `TestConditionHandling.test_no_drift_condition_requires_index_source_same_as_drift` verifies the control gets no free pass on this rule.

## Serialization and naming

Deterministic checkpoint IDs follow `{detector}_{condition}_{seed}_{checkpoint}` exactly (e.g. `detector_a_drift_42_pre_drift`), tested directly. `CheckpointRecord.to_metadata_dict()` produces a JSON-serializable payload (verified via an actual `json.dumps`/`json.loads` round trip) that excludes the model snapshot, since a live model object isn't JSON-safe; the snapshot itself is a Python object suitable for separate pickling later, when persistence is actually needed. No results or model files were written into the repository — `results/` remains empty, consistent with existing project policy.

## Exact command and test result

```
python3 -m unittest discover -s tests -v
```
```
Ran 80 tests in 1.994s

OK (skipped=2)
```
78 ran and passed, 2 skipped 
## Verified 

**Implemented and verified (real execution, synthetic data):** fixed-sample construction and immutability, pre-drift-only enforcement, checkpoint ordering/duplication/completeness rules, unlabeled-event-index rejection (including for the no-drift control), model-snapshot independence for **both** Detector A and Detector B, deterministic naming, JSON metadata round-trip.

**Not implemented (per explicit restriction this phase):** SHAP explanation generation, stability metrics, detection-performance evaluation, full experiment runner.

## Issues found and fixed

None this phase — all 80 tests passed on first full run. (Phase 3's `update()` transition bug remains fixed and covered by regression tests, which continue to pass.)

## Still not implemented

SHAP explanation generation, stability metrics (cosine similarity, top-k overlap), detection-performance evaluation, full multi-seed experiment runner. No NAB-dependent values (window size, injection point, drift magnitude, retrain-window size) have been locked — all remain `null`/TBD in `config/experiment.yaml`, pending your laptop's real dataset inspection.

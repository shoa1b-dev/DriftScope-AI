# Phase 5 — SHAP Explanation Generation

## Files created / modified

| File | Status |
|---|---|
| `src/explanation/explanation_result.py` | New — implemented and verified |
| `src/explanation/shap_explainer.py` | New — dispatch/validation/normalization implemented and verified; real shap execution unverified |
| `src/detectors/detector_a_isolation_forest.py` | Modified — added `raw_model` property (exposes the underlying sklearn estimator for `TreeExplainer`) |
| `src/detectors/detector_b_autoencoder.py` | Modified — added `raw_model` property (for inspection; NOT what gets explained — see design decisions) |
| `tests/unit/test_shap_explainer.py` | New, 24 tests |
| `docs/PHASE5_SHAP_EXPLANATION.md` | This file |

## Exact command executed

```
python3 -m unittest discover -s tests -v
```

## Exact test count

- **Collected:** 101
- **Passed:** 97
- **Skipped:** 4 (2 real-ADWIN/Page-Hinkley from Phase 3, unchanged; 2 real-SHAP tests, new this phase)
- **Failed:** 0

## Implemented and verified (real execution, this environment)

- Backend dispatch: `get_explanation_target("detector_a", ...)` returns `TreeExplainer` + the raw sklearn `IsolationForest`; `get_explanation_target("detector_b", ...)` returns the permutation backend label + the detector's own `anomaly_score` method.
- **Direct proof that Detector B's explanation target is the anomaly score, not an arbitrary output**: `test_detector_b_target_is_the_anomaly_score_method_not_raw_model` calls the dispatched target and compares its output byte-for-byte against `detector.anomaly_score(...)` called directly.
- Feature-name preservation and validation (empty / mismatched-length rejected).
- Output shape normalization, including unwrapping a `shap.Explanation`-like object (tested with a fake object exposing `.values`, since real shap isn't installed).
- Checkpoint-label compatibility — the same `generate_explanation` call works for all three checkpoints.
- Fixed-sample immutability validation — a deliberately tampered writeable sample is rejected; a structurally empty sample is rejected.
- Same-sample-across-checkpoints — explanation signature is identical across all three checkpoints for one run.
- Metadata correctness and JSON serializability, including a determinism check (identical inputs → identical metadata dict).

## Design decisions needing your review

1. **Detector B's SHAP background/evaluation data.** `shap.Explainer(target, X, algorithm="permutation")` uses the same `X` (the fixed sample) as both the permutation background distribution and the evaluation set. This is a simplification — a separate, larger background set is more standard SHAP practice, but the approved proposal's fixed-sample design doesn't specify a separate background. I used the fixed sample for both to keep this phase self-contained; flagging this as a decision you may want to revisit once real SHAP runs are possible, rather than treating it as already settled.
2. **`raw_model` property added to both detector classes** (not just Detector A, which strictly needs it for `TreeExplainer`). Added symmetrically for interface consistency and future inspection, even though Detector B's explanation path doesn't use it. Minor, but it is a change to files from an earlier approved phase, so flagging per your standing practice of not silently modifying completed work.
3. **Metadata now includes attributions as plain numbers** (`ExplanationResult.to_metadata_dict()`), unlike `CheckpointRecord.to_metadata_dict()` which excludes the model snapshot. This is intentional — attributions are plain numeric results, not a live object — but wanted to name the asymmetry explicitly rather than let it pass silently.

## Bugs found and fixed

None this phase — all 101 tests passed on the first full run.

## Still not implemented

Stability metrics, detection evaluation, experiment runner. No NAB-dependent values locked — all remain `null`/TBD in `config/experiment.yaml`.

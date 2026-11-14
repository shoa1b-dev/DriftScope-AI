# Phase 5.1 — Review and Correction: SHAP Background-Sample Design

## 1. Final decision: Option B (separate background and evaluation samples)

## 2. Rationale

The Phase 5 implementation used the same 10-point fixed sample as both the permutation background distribution and the set of points explained at each checkpoint. On review, this conflates two statistically different roles: the background is meant to represent the general pre-drift distribution (and a stable expectation estimate benefits from more points), while the evaluation sample is deliberately small and fixed specifically so the *same* points can be tracked across checkpoints. Collapsing them into one small sample meant the background distribution was likely too sparse to give a stable permutation baseline. Option B separates these roles while changing nothing about the leakage-prevention design: both samples are still built exclusively from the pre-drift region via the same `build_fixed_sample_from_full_series` mechanism, both are immutable, both are selected once before drift and reused unchanged at every checkpoint. Overlap between the two samples' indices is permitted (standard SHAP practice — background sets commonly include points also being explained); what would NOT be permitted, and still isn't, is either sample drawing from post-drift data.

## 3. Files changed

| File | Change |
|---|---|
| `src/explanation/fixed_sample.py` | Added `build_background_and_evaluation_samples(...)` convenience function |
| `src/explanation/explanation_result.py` | Added `background_sample_signature` field; clarified metadata-asymmetry docstring |
| `src/explanation/shap_explainer.py` | `generate_explanation` now takes `background_sample` (required for the permutation backend, N/A for tree); new `MissingBackgroundSampleError`; `raw_value_fn`/`_run_real_shap` signature extended to `(backend, target, eval_X, background_X)`; `_validate_sample` factored out and reused for both samples |
| `src/detectors/detector_a_isolation_forest.py` | `raw_model` now returns a fresh `copy.deepcopy` on every access (was a live reference) |
| `src/detectors/detector_b_autoencoder.py` | `raw_model` **removed** (unused by the SHAP pipeline); added `model_class_name` (safe, string-only accessor) for the one legitimate remaining need (metadata) |
| `src/checkpoints/checkpoint_manager.py` | Docstring only — clarified the metadata-asymmetry rationale, no behavior change |
| `config/experiment.yaml` | Added `explanation.evaluation_sample_size` and `explanation.background_sample_size` (both `null`/TBD — genuinely dataset-dependent, not guessed) |
| `tests/unit/test_shap_explainer.py` | Rewritten: new tests for background-sample requirement/rejection, both-samples-reused-across-checkpoints, `raw_model` isolation, `raw_model` absence on Detector B, and the metadata-asymmetry regression test |

## 4. Exact test count

**Collected: 110 · Passed: 106 · Skipped: 4 · Failed: 0**

```
python3 -m unittest discover -s tests -v
...
Ran 110 tests in 4.388s

OK (skipped=4)
```

## 5. Passed / skipped / failed breakdown

106 passed (9 new/changed this review, plus all 97 carried over unchanged from Phase 5 minus the ones rewritten). 4 skipped — unchanged set: 2 real-ADWIN/Page-Hinkley (Phase 3), 2 real-SHAP (Phase 5). 0 failed.

## 6. Bugs found

None — this was a design correction, not a bug fix. No test failed at any point during the change; the rewritten tests were written to match the new interface before being run, and passed on the first full run after the rewrite.

## 7. Resolution of the three design concerns

1. **Background sample: resolved.** Option B implemented — separate, independently-seeded, immutable, pre-drift-only background and evaluation samples; both proven to be reused identically across all three checkpoints (new test), and proven to have distinct signatures from each other (new test).
2. **`raw_model` property: resolved.** Kept on Detector A (necessary for `TreeExplainer`, no equivalent alternative), now hardened to return a deepcopy on every access rather than a live reference. Removed from Detector B (never used by the SHAP pipeline — confirmed by tracing the actual dispatch logic, which uses `anomaly_score` instead).
3. **Metadata asymmetry: resolved (was already intentional, now explicitly documented and regression-tested).** Both `to_metadata_dict()` docstrings now state the rule directly: live model objects never appear in metadata; plain numeric results (like SHAP attributions) do. A new test (`test_checkpoint_record_metadata_never_contains_model_object`) locks this in going forward.

## 8. Real SHAP execution status

Unchanged and clearly marked: still **unverified** in this environment (`shap` not installed, no network egress). The two real-execution tests (`TestRealSHAPExecution`) remain ready-to-run and are reported as skipped, not passing.

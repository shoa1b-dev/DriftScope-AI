# Phase 6 — Explanation Stability Metrics

## Files created / modified

| File | Status |
|---|---|
| `src/metrics/explanation_stability.py` | New — implemented and verified (synthetic vectors) |
| `tests/unit/test_explanation_stability.py` | New, 39 tests |
| `docs/PHASE6_STABILITY_METRICS.md` | This file |

## Exact test command

```
python3 -m unittest discover -s tests -v
```

## Exact counts

**Collected: 149 · Passed: 145 · Skipped: 4 · Failed: 0**

(4 skips unchanged from prior phases: 2 real-ADWIN/Page-Hinkley, 2 real-SHAP.)

## Implemented and verified (real execution, synthetic data)

**Cosine similarity:** identical→1.0, orthogonal→0.0, opposite→−1.0 (all exact, not approximate — `test_cosine_values_are_exact_computed_result_not_approximated` checks `== 1.0`, not "close to"), a hand-computed 45° case (`sqrt(2)/2`), shape mismatch, zero-vector rejection (one-sided and both-sided), NaN/Inf rejection, matrix mean aggregation, both `on_invalid` modes (`raise` stops immediately; `mark_nan` records NaN and excludes it from the mean, tested to actually change the resulting mean correctly).

**Top-k Jaccard:** correct top-k selection by absolute magnitude (including a case where the largest-magnitude value is negative, proving sign is ignored correctly), k validation (`k<1`, `k>n_features`), deterministic tie-breaking (explicit tied-value test), the all-zero-vector case proven deterministic across repeated calls (not an error, per your requirement), feature-name/vector length mismatch, identical→1.0, disjoint→0.0, a hand-computed partial-overlap case (1/3), matrix aggregation, non-finite rejection.

**Top-level API:** both approved comparisons work; the two unapproved pairs tested are rejected — including explicitly testing that the *reversed* direction of an approved pair is still rejected (direction matters, not just membership); five consistency checks (detector/condition/seed/sample-signature/feature-names) each independently tested; metadata field correctness; JSON serialization, including the NaN→`null` conversion under `mark_nan` mode (verified to produce actually-valid JSON, not just "doesn't crash").

## Bugs found and fixed

One, in the **test file**, not the module under test: `test_known_hand_computed_cosine_value` used `math.sqrt` without importing `math`, causing a `NameError`. Fixed by adding the import; re-ran the full suite afterward to confirm the fix and that nothing else was masked by it. The module code (`explanation_stability.py`) itself had no bugs — all its tests passed on the first run once the import was fixed.

## Design decisions made (documented, not flagged as needing separate approval — but noting them for your review)

1. **Zero-vector policy for cosine similarity**: raises `ZeroVectorError` rather than returning a default (0.0, 1.0, or NaN) inline, because none of those defaults is objectively "correct" and picking one silently could misrepresent a real finding. Batch-level callers opt into either strict (`raise`) or lenient (`mark_nan`, explicit exclusion from the mean, with the excluded count reported) — never a silent average that hides how many samples were actually valid.
2. **Jaccard has no zero-vector failure mode**, unlike cosine — an all-zero attribution row still produces a deterministic top-k selection (lowest-index features win ties), so it's treated as a valid case, not an error. This asymmetry between the two metrics is intentional and documented in the module docstring.
3. **Aggregation method is per-sample-then-mean only** — no flattened/global alternative was implemented, per your "preferred approach" instruction. Documented in the module docstring as a design choice, not merely an implementation default.
4. **Tie-breaking rule**: ties in absolute attribution magnitude (including all-zero vectors) are broken by ascending original feature index. Simple, fully deterministic, and doesn't depend on feature name alphabetization or array/sort-algorithm quirks.

## Confirmation: no fabricated experiment results

Every numeric value used across all 39 new tests is either a hand-constructed synthetic vector with a known, hand-computable answer (e.g. the 45°/`sqrt(2)/2` case) or a deliberately constructed edge case (zero vectors, NaN, ties). No SHAP attribution, detection metric, or drift-recovery finding is claimed or implied anywhere in this phase — the module has never been run against real explanation output, since no real SHAP execution has occurred yet in this project.

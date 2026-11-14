# Phase 7 — Detection Evaluation Metrics

## Files created / modified

| File | Status |
|---|---|
| `src/metrics/detection_evaluation.py` | New — implemented and verified (synthetic labels/predictions) |
| `tests/unit/test_detection_evaluation.py` | New, 38 tests |
| `docs/PHASE7_DETECTION_EVALUATION.md` | This file |

## Exact test command

```
python3 -m unittest discover -s tests -v
```

## Exact counts

**Collected: 187 · Passed: 183 · Skipped: 4 · Failed: 0**

(4 skips unchanged: 2 real-ADWIN/Page-Hinkley, 2 real-SHAP.)

## Implemented and verified (real execution, synthetic data)

- **Confusion matrix**: correct TP/TN/FP/FN counts on a hand-traced 6-point example, Python-int typing, JSON serialization, length-mismatch rejection, empty-input rejection, non-binary label/prediction rejection (both sides tested separately), boolean-label acceptance.
- **Precision/recall/F1**: normal case against a hand-computed fraction (2/3 for all three), and three separate zero-denominator cases each tested for both the `0.0` value *and* the `undefined=True` flag — including the exact no-drift-control scenario (zero actual positives → recall undefined, not a real miss), plus a direct "never NaN or Inf" sweep.
- **Thresholding**: `>=` boundary-inclusive and `>` boundary-exclusive behavior both tested at the exact threshold value; invalid comparison operator rejected.
- **Episode extraction**: single episode, multiple separated episodes, adjacent points merging into one episode, episode at sequence start, episode at sequence end, all-anomalous sequence, and the no-drift control (all-zero → empty tuple, never a fabricated episode). Invalid episode construction (`end < start`) rejected.
- **Episode-level zero-delay/delay evaluation**: zero-delay case, positive-delay case (exact value, not approximate), missed-episode case (delay/index both `None`, never a placeholder number), a prediction *before* the episode correctly not counted, and — directly testing your definition — a prediction *after* the episode end correctly not counted either (episode reported as missed, per the "only search within [start, end]" definition).
- **Multi-episode aggregation**: mixed detected/missed episodes, mean delay computed only over detected episodes (verified against a hand-computed average), the "all missed" case correctly returning `None` for the mean rather than fabricating `0`, and the no-episodes case.
- **Top-level `evaluate_detection`**: full pipeline on a drift-condition example, the no-drift-control case (zero episodes, recall flagged undefined), threshold value carried into metadata, full metadata JSON round-trip.

## Implemented but unverified

Nothing new — this module has no dependency on `river` or `shap`.

## Not implemented (per your restriction)

Experiment runner, NAB loading, drift-monitor changes, retraining changes, SHAP execution, stability-metric changes, real experiments, ZIP.

## Bugs found and fixed

One, in the **test file**: `test_metadata_json_serializable` asserted `n_episodes == 1` for ground truth `[1, 0, 1, 1]`, but that sequence genuinely contains 2 episodes (index 0 alone, then indices 2–3) — the module's episode extraction was correct; my test's expected value was wrong. Fixed the assertion to `2`, re-ran the full suite to confirm. The module itself (`detection_evaluation.py`) had no bugs — every test targeting the module logic passed on first run.

## Design decisions needing your approval

1. **Zero-delay/delay search window is bounded to `[episode.start, episode.end]` inclusive** — a prediction occurring after an episode's end is never attributed to that episode, even if no closer prediction exists. This directly implements your instruction ("if no prediction occurs during the episode, mark the episode as missed"), but is worth confirming explicitly since a looser definition (search open-ended from episode start onward) was also structurally possible and would have changed several test outcomes.
2. **MetricValue's `undefined` flag pattern** for precision/recall/F1 — rather than a single policy of "always 0.0" or "always raise," every zero-denominator case returns `0.0` *and* an explicit `undefined: bool`. This satisfies "do not silently produce NaN/infinity" while still letting downstream reporting distinguish a genuine 0-performance case from a structurally-undefined one (most importantly, the no-drift control's recall). Flagging this because it's a slightly richer contract than "return 0.0" alone, which is what the phase instructions' example wording suggested as one valid option.
3. **Label validation is strict**: `confusion_matrix_counts` and `extract_episodes_from_binary` both reject anything that isn't already exactly 0/1 or False/True — no automatic threshold-based conversion happens inside them. Converting continuous scores to binary is only possible via the separate, explicit `threshold_scores` function. This keeps "how a score became a label" always visible and auditable, rather than letting it happen implicitly inside evaluation.

## Confirmation: no fabricated experiment results

Every value in all 38 new tests is either hand-traced from a small synthetic sequence with a computable answer, or a deliberately constructed edge case (empty input, all-zero ground truth, missed episodes, boundary threshold values). No real NAB ground truth, detector prediction, or delay/recovery finding is claimed or implied anywhere in this phase.

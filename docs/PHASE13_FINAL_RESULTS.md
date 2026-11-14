# Phase 13 — Final Results Analysis, Statistical Evaluation & Research Completion

**Explanation Stability Under Concept Drift: An Empirical Evaluation of
Detection-Performance and SHAP-Attribution Recovery in CPU-Only
Time-Series Anomaly Detection**

This document is the final, consolidated analysis of the real Phase 12
experiment. It uses **only** the authoritative real-execution artifacts:

- `artifacts/phase12/real_experiment_results.json`
- `artifacts/phase12/real_experiment_results.csv`

produced by `python scripts/run_phase12_experiment.py --nab-root "E:\NAB"`
on the laptop environment (River 0.26.1, SHAP 0.51.0, Windows, CPU-only).


All tables in this document are also saved, unrounded, in
`artifacts/phase13/tables/*.csv`, and all figures in
`artifacts/phase13/figures/*.png`, so every number here is traceable
back to the serialized result.

---

## Step 1 — Result Integrity

Verified directly against the JSON/CSV before any analysis:

| Check | Result |
|---|---|
| Total rows | 20 (10 condition×seed × 2 detectors) |
| Conditions present | `drift`, `no_drift` — 10 each |
| Seeds present | 13, 42, 101, 202, 303 — all 5 |
| Detectors present | `detector_a`, `detector_b` — 10 each |
| Status | 16 `completed`, 4 `failed` |
| Failure type | `DriftNeverDetectedError` only (no `RiverUnavailableError` / `SHAPUnavailableError`) |
| Duplicate (condition, seed, detector) keys | 0 |
| JSON/CSV consistency | All 20 rows identical on condition/seed/detector/status between the two files |
| Checkpoint completeness (completed rows) | `pre_drift`, `post_drift_pre_retrain`, `post_retrain` all populated for all 16; all null for the 4 failed rows |
| SHAP attribution validity | Real, varying attribution values in all 16 completed rows; `n_invalid` = 0 in every case (the all-zero-attribution edge case was never triggered this run) |
| Config recorded matches locked design | `t_event_window=2410` (window derived from `t_event=2419`), retrain/post-retrain windows = 100, feature set `[mean, std, min, max, slope]` — all match `config/experiment.yaml` |

**The 4 failures** are `drift` condition, seeds **101** and **303**, both
detectors, each with message *"primary monitor never detected drift
within the available post-event data (scanned indices 2410..4022)"*.
Per the locked research design, this is a **legitimate experimental
outcome** — the drift was genuinely injected into the series; ADWIN
simply never fired for those two seeds. It is not treated as evidence
that no drift occurred, and the drift magnitude/ADWIN configuration
were not altered to force a detection.

Integrity check: **PASSED**. Analysis below proceeds on the real data.

---

## Step 2 — Descriptive Analysis

Full unrounded tables: `artifacts/phase13/tables/A`–`F_*.csv`.

### A. Drift condition (10 attempted, 6 completed / 4 failed)

| metric | n | mean | median | std | min | max |
|---|---|---|---|---|---|---|
| pre_drift_f1 | 6 | 0.2089 | 0.2070 | 0.0085 | 0.2015 | 0.2202 |
| pre_drift_precision | 6 | 0.1391 | 0.1377 | 0.0297 | 0.1120 | 0.1693 |
| pre_drift_recall | 6 | 0.6549 | 0.6574 | 0.3780 | 0.3037 | 1.0000 |
| post_drift_pre_retrain_f1 | 6 | 0.2953 | 0.3019 | 0.1126 | 0.1760 | 0.3976 |
| post_drift_pre_retrain_precision | 6 | 0.1775 | 0.1815 | 0.0777 | 0.0965 | 0.2482 |
| post_drift_pre_retrain_recall | 6 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| post_retrain_f1 / precision / recall | 6 | **undefined — see Step 3** | | | | |
| cosine_vs_post_drift_mean | 6 | 0.9997 | 0.9999 | 0.0004 | 0.9991 | 1.0000 |
| cosine_vs_post_retrain_mean | 6 | 0.5064 | 0.6932 | 0.4557 | -0.1116 | 0.9128 |
| jaccard_vs_post_drift_mean | 6 | 0.9889 | 1.0000 | 0.0272 | 0.9333 | 1.0000 |
| jaccard_vs_post_retrain_mean | 6 | 0.5778 | 0.4333 | 0.2697 | 0.3667 | 1.0000 |

### B. No-drift condition (10/10 completed)

| metric | n | mean | median | std | min | max |
|---|---|---|---|---|---|---|
| pre_drift_f1 | 10 | 0.2086 | 0.2070 | 0.0078 | 0.2015 | 0.2202 |
| post_drift_pre_retrain_f1 | 10 | 0.0000 (undefined for recall/F1 in all 10 — no anomalies in the matched-point window) | | | | |
| post_retrain_f1 | 10 | **undefined — see Step 3** | | | | |
| cosine_vs_post_drift_mean | 10 | 0.9989 | 1.0000 | 0.0027 | 0.9912 | 1.0000 |
| cosine_vs_post_retrain_mean | 10 | 0.3244 | 0.3102 | 0.4713 | -0.1640 | 0.8306 |
| jaccard_vs_post_drift_mean | 10 | 0.9800 | 1.0000 | 0.0450 | 0.8667 | 1.0000 |
| jaccard_vs_post_retrain_mean | 10 | 0.4567 | 0.3667 | 0.2816 | 0.1333 | 0.8667 |

### C/D. Detector A vs Detector B (all 8 completed rows each, both conditions pooled)

| metric | Detector A mean (std) | Detector B mean (std) |
|---|---|---|
| pre_drift_f1 | 0.2160 (0.0032) | 0.2015 (0.0000) |
| pre_drift_precision | 0.1653 (0.0028) | 0.1120 (0.0000) |
| pre_drift_recall | 0.3116 (0.0061) | **1.0000 (0.0000)** |
| post_drift_pre_retrain_precision | 0.0400 (0.0555) | 0.0931 (0.1284) |
| cosine_vs_post_drift_mean | 1.0000 (0.0000) | 0.9985 (0.0029) |
| **cosine_vs_post_retrain_mean** | **-0.0183 (0.2471)** | **0.8036 (0.0678)** |
| jaccard_vs_post_retrain_mean | 0.3167 (0.1168) | 0.6875 (0.2636) |

Detector B (the autoencoder) has **exact** perfect recall (1.0, zero
variance) at the pre-drift checkpoint across every single run — it
flags every labeled anomaly present in the pre-drift evaluation region,
at low precision. Detector A (Isolation Forest) never does. This
asymmetry persists into every downstream comparison below, most
strikingly in explanation stability (bolded row).

### E. Successful drift-detection runs (seeds 13, 42, 202 × 2 detectors)

| seed | detector | t_detect | t_retrain | pre_drift F1 | post_drift F1 | cosine(post_drift) | cosine(post_retrain) |
|---|---|---|---|---|---|---|---|
| 13 | detector_a | 3593 | 3594 | 0.2124 | 0.2061 | 1.0000 | 0.5826 |
| 13 | detector_b | 2953 | 2954 | 0.2015 | 0.3976 | 0.9995 | 0.8037 |
| 42 | detector_a | 3657 | 3658 | 0.2202 | 0.1965 | 1.0000 | -0.0127 |
| 42 | detector_b | 2953 | 2954 | 0.2015 | 0.3976 | 0.9991 | 0.9128 |
| 202 | detector_a | 3817 | 3818 | 0.2162 | 0.1760 | 1.0000 | -0.1116 |
| 202 | detector_b | 2953 | 2954 | 0.2015 | 0.3976 | 0.9998 | 0.8637 |

`t_retrain = t_detect + 1` holds exactly in every row, as designed.

### F. Drift-detection failures

| seed | detector | error_type |
|---|---|---|
| 101 | detector_a | DriftNeverDetectedError |
| 101 | detector_b | DriftNeverDetectedError |
| 303 | detector_a | DriftNeverDetectedError |
| 303 | detector_b | DriftNeverDetectedError |

No failure is silently excluded from any table above — the drift
condition tables explicitly report n=6 (of 10 attempted), and this
table lists all 4 by seed and detector.

---

## Step 3 — Detection Performance

**A finding that shapes everything downstream: `post_retrain`
precision/recall/F1 are `undefined` in all 16/16 completed runs**,
in both conditions. Checking the `*_undefined` flags directly:

- `post_retrain_recall_undefined = True` for all 16 rows → the
  100-sample post-retrain evaluation window contained **zero true
  labeled anomalies** in every single run, for both conditions.
- `post_retrain_precision_undefined = False`, value `0.0`, for all 16
  → each detector *did* raise at least one alarm in that window, and
  every alarm was a false positive (0 true positives / n predicted
  positives = 0.0, precision is well-defined here, just zero).

**This means detection performance genuinely cannot be evaluated with
precision/recall/F1 at the post-retrain checkpoint in this experiment** —
not because performance collapsed to zero, but because the metric's
denominator (true positives available) is structurally absent in every
100-sample post-retrain window. Per the project's own methodological
rule, these values are **not manufactured**; they are reported as
undefined. This is a genuine limitation of the checkpoint design
combined with label sparsity (see Step 9), not a scientific finding
about detector behavior.

At the **post-drift/pre-retrain** checkpoint, by contrast, all 6
successful drift runs have well-defined metrics, and **recall is
exactly 1.0 in all 6** — the post-drift/pre-retrain window fully
captures the one post-event NAB labeled anomaly window. Precision
varies by detector: detector_a 0.097–0.115, detector_b flat at 0.2482
for all three seeds (identical, because detector_b's t_detect is
identical — 2953 — across all three successful seeds, producing the
same window and the same score behavior).

Wilcoxon signed-rank, F1 pre-drift vs. post-drift/pre-retrain
(drift condition, n=6): **statistic=6.0, p=0.4375 — not significant.**
The reason is visible in the raw differences: detector_b's F1 rises in
all 3 of its runs (+0.196 each), while detector_a's F1 falls in all 3
of its runs (-0.006 to -0.040). The effect is not random noise — it is
a **detector-identity split**, not a general "drift hurts F1" or
"drift helps F1" pattern.

No-drift condition: post_drift_pre_retrain recall/F1 are undefined in
all 10 rows (the matched-arbitrary-point window contains no labeled
anomalies), so there is no valid no-drift baseline for this specific
checkpoint's F1 — only the drift condition's post-drift/pre-retrain
checkpoint happens to land on real anomaly data, an artifact of where
`t_event` and the one post-event NAB window fall, not a property of
the detectors.

---

## Step 4 — Explanation Stability

Both approved comparisons are anchored to the **pre-drift** explanation
as baseline (this is the only pairing the implementation supports —
`APPROVED_COMPARISONS` in `src/metrics/explanation_stability.py`
explicitly rejects any other checkpoint pair, including
`post_drift_pre_retrain → post_retrain` directly). That direct
comparison is therefore **not computed** anywhere in this project and
is not fabricated here; where useful below, the two baseline-anchored
measures are used together to infer the direction of change between
them.

**Primary metric — cosine similarity:**

- `pre_drift → post_drift_pre_retrain`: essentially at ceiling
  everywhere — drift: mean 0.9997 (min 0.9991); no_drift: mean 0.9989
  (min 0.9912). Explanations barely move at all immediately after
  drift injection, before any retraining happens, in **both**
  conditions.
- `pre_drift → post_retrain`: much more variable and, on average,
  far lower — drift: mean 0.5064 (range -0.1116 to 0.9128); no_drift:
  mean 0.3244 (range -0.1640 to 0.8306).

**Every one of the 16 completed runs (16/16) shows a lower
`cosine_vs_post_retrain` than `cosine_vs_post_drift`** — i.e.
explanations are *more* different from the pre-drift baseline after
retraining than they were right after drift injection, before
retraining. Wilcoxon signed-rank:

| comparison | n | statistic | p |
|---|---|---|---|
| drift: cosine(post_drift) vs cosine(post_retrain) | 6 | 0.0 | **0.0312** |
| no_drift: cosine(post_drift) vs cosine(post_retrain) | 10 | 0.0 | **0.0020** |

Both are the most extreme possible result for their sample size (all
differences same sign), and both are significant at α=0.05.

**Secondary metric — top-k Jaccard** shows the same pattern:

| comparison | n | statistic | p |
|---|---|---|---|
| drift: jaccard(post_drift) vs jaccard(post_retrain) | 6 | 0.0 | 0.0625 (not significant at α=0.05, though 5/6 differences are negative and one is exactly 0) |
| no_drift: jaccard(post_drift) vs jaccard(post_retrain) | 10 | 0.0 | **0.0020** |

**This pattern — stability dropping after retraining relative to the
pre-drift explanation — happens regardless of whether real drift was
injected** (it appears in `no_drift` as strongly as in `drift`). A
Mann-Whitney U test comparing `cosine_vs_post_retrain` between drift
(n=6) and no_drift (n=10) found **no significant difference**
(U=40.0, p=0.3132). Retraining itself, not the drift event, is
associated with the explanation shift.

---

## Step 5 — Detection vs. Explanation Relationship

The standout finding of this project is a **detector-identity effect**,
not a drift effect. Splitting `cosine_vs_post_retrain` by detector
across all 16 completed runs:

- **detector_a** (Isolation Forest): range -0.1640 to 0.5826, mean **-0.018**
- **detector_b** (Autoencoder): range 0.7078 to 0.9128, mean **0.804**

The two ranges **do not overlap at all** (detector_a's maximum, 0.583,
is below detector_b's minimum, 0.708). Mann-Whitney U:
**U=0.0, p=0.0002** — the most extreme possible separation for n=8
vs n=8. After retraining, detector_a's SHAP explanations for the fixed
evaluation sample become essentially uncorrelated (or weakly
anti-correlated) with their pre-drift explanations, while detector_b's
explanations remain substantially similar to pre-drift.

A Spearman correlation between `post_drift_pre_retrain_precision` and
`cosine_vs_post_retrain_mean` (drift condition, n=6) gives **rho=0.94,
p=0.0051** — but this is **confounded with detector identity**:
detector_b's precision (flat 0.2482) and its high cosine stability
co-occur simply because both are properties of the same detector,
not because higher post-drift precision *causes* better post-retrain
explanation stability. With only two detectors and three seeds each,
this experiment cannot separate "detector type" from "detection
quality" as explanations for the correlation. It is reported honestly
as descriptive, not as a validated causal or even purely correlational
finding independent of detector identity.

**Per the instructions for this step, we explicitly do not assume
explanation recovery must track detection recovery** — and the data
supports treating them as separate axes: detection metrics are
undefined post-retrain for every run (Step 3), while explanation
stability is measurable and detector-dependent post-retrain (this
step). The two phenomena are evaluated on different evidence and
should not be merged into a single "recovery" verdict.

---

## Step 6 — Drift Detection (ADWIN) Behavior

- **Detection success rate**: 6/10 detector-runs (60%); **failure
  rate**: 4/10 (40%).
- Failures are **seed-level, not detector-level**: both detectors
  failed together for seed 101 and for seed 303; both succeeded
  together for seeds 13, 42, 202.
- **detector_a** `t_detect`: 3593, 3657, 3817 (mean 3689, std 115.4) —
  varies noticeably across its three successful seeds.
- **detector_b** `t_detect`: 2953, 2953, 2953 (std **0.0**) — identical
  across all three of its successful seeds. This is consistent with
  the drift injection itself being deterministic given `t_event` and
  the fixed `+3.85` offset; detector_b's reconstruction-error signal
  crosses ADWIN's detection threshold at the same point regardless of
  seed, while detector_a's Isolation Forest scores (whose internal
  splits do depend on `random_state`) shift ADWIN's exact detection
  index more.
- `t_retrain = t_detect + 1` holds exactly for all 6 successful runs
  (see Step 2E table).
- Secondary monitor (Page-Hinkley): `status="ran"` for all 6 completed
  drift rows and `"not_run"` for all 10 no_drift rows (correctly never
  invoked, per the locked design). Of the 6 "ran" rows, only 2 report
  a `secondary_monitor_detection_index` (both = 2933, both
  detector_b, seeds 42 and 202) — Page-Hinkley, being secondary and
  non-blocking, did not itself detect drift in the other 4 successful
  runs even though ADWIN (primary) did.

**The four undetected detector-runs are not evidence that no drift
occurred.** The offset was injected identically for every seed; ADWIN
simply did not cross its detection threshold within the scanned
post-event region (indices 2410–4032) for seeds 101 and 303. See
`artifacts/phase13/figures/fig4_adwin_detection_timing.png`.

---

## Step 7 — Statistical Analysis

Full table: `artifacts/phase13/tables/G_statistical_tests.csv`.

| Comparison | n | Test | Statistic | p | Interpretation |
|---|---|---|---|---|---|
| DRIFT: F1 pre_drift vs post_drift_pre_retrain | 6 | Wilcoxon signed-rank | 6.0 | 0.4375 | Not significant; effect is split exactly by detector, not directional overall |
| DRIFT: precision pre_drift vs post_drift_pre_retrain | 6 | Wilcoxon | 6.0 | 0.4375 | Not significant, same detector-split pattern |
| DRIFT: cosine(post_drift) vs cosine(post_retrain) | 6 | Wilcoxon | 0.0 | **0.0312** | Significant; stability lower post-retrain in all 6/6 |
| DRIFT: jaccard(post_drift) vs jaccard(post_retrain) | 6 | Wilcoxon | 0.0 | 0.0625 | Not significant at α=0.05 (borderline; 5/6 negative, 1 tied at 0) |
| NO_DRIFT: cosine(post_drift) vs cosine(post_retrain) | 10 | Wilcoxon | 0.0 | **0.0020** | Significant; stability lower post-retrain in 10/10 |
| NO_DRIFT: jaccard(post_drift) vs jaccard(post_retrain) | 10 | Wilcoxon | 0.0 | **0.0020** | Significant; same direction in 10/10 |
| cosine_vs_post_retrain: drift (n=6) vs no_drift (n=10) | 6 vs 10 | Mann-Whitney U | 40.0 | 0.3132 | Not significant — the post-retrain instability is not drift-specific |
| cosine_vs_post_retrain: detector_a (n=8) vs detector_b (n=8) | 8 vs 8 | Mann-Whitney U | 0.0 | **0.0002** | Significant, complete separation — the dominant effect in this dataset |
| post_drift_precision vs cosine_vs_post_retrain (drift) | 6 | Spearman | rho=0.9411 | 0.0051 | Descriptive only; confounded with detector identity (see Step 5) |
| Post-retrain F1 (any comparison) | — | — | — | — | **Not tested** — undefined in all 16 completed runs (Step 3); no valid observations exist |

**Comparisons explicitly not tested for insufficient observations:**
the 4-run failure subset (seeds 101/303) has no completed metrics to
test at all; no comparison in this project has fewer than 6 paired
observations run through Wilcoxon, per the pre-registered rule against
forcing significance testing onto too-small samples.

No claim of statistical significance is made anywhere in this document
beyond what the p-values above actually support, and every non-significant
result is reported as such rather than omitted.

**Non-independence caveat for the cosine/Jaccard Wilcoxon tests.** The
two `cosine(post_drift) vs cosine(post_retrain)` and two
`jaccard(post_drift) vs jaccard(post_retrain)` tests above use:

- drift: n=6 = **3 detected seeds × 2 detectors**
- no_drift: n=10 = **5 seeds × 2 detectors**

so detector identity is repeated within each group rather than each
observation being an independent, exchangeable draw. Given the very
strong detector-identity effect found separately (Section 5's
Mann-Whitney U, p=0.0002), a meaningful share of each group's internal
consistency is attributable to only two underlying detector instances,
not to six or ten independent experimental units. The reported
Wilcoxon statistics and p-values (0.0312, 0.0625, 0.00195, 0.00195)
are mathematically correct for the observed paired values, but because
of this repeated-detector structure they should **not** be read as
stronger population-level evidence than this experiment's design
permits. The underlying **descriptive** finding is unaffected by this
caveat: the metric decreased from post-drift/pre-retrain to
post-retrain in every one of the 16 completed runs, regardless of how
the paired significance tests are qualified.

---

## Step 8 — Figures

All in `artifacts/phase13/figures/`, generated directly from the result
arrays (no illustrative/decorative figures):

1. **`fig1_detection_f1_by_checkpoint.png`** — mean F1 per checkpoint,
   drift vs. no-drift, with n-valid/n-total annotated per bar (makes
   the post-retrain undefined-everywhere finding visually explicit).
2. **`fig2_explanation_stability_by_detector.png`** — scatter of
   cosine(post_drift) vs. cosine(post_retrain) for all 16 completed
   runs, colored by detector, shaped by condition — shows the
   detector_a/detector_b separation directly.
3. **`fig3_stability_recovery_slopegraph.png`** — paired
   before/after slopegraph per run, split by condition — shows the
   16/16 uniform-direction drop and the detector split simultaneously.
4. **`fig4_adwin_detection_timing.png`** — `t_detect` per seed and
   detector, with an ✕ marking the 4 `DriftNeverDetectedError` cases.

---

## Step 9 — Limitations

1. Of the 3 NAB labeled anomaly windows in this series, only **1**
   falls after `t_event=2419`; the other 2 are pre-event. This was
   identified during Phase 11 baseline analysis and was not used as a
   reason to move `t_event`.
2. Only **3 of 5** drift seeds (13, 42, 202) produced a successful
   ADWIN detection; seeds 101 and 303 legitimately never triggered
   detection within the available post-event data.
3. The drift-condition **effective post-detection/retraining sample is
   therefore n=3 seeds (6 detector-runs)**, not 5/10 — all statistics
   above that condition on successful detection are explicitly scoped
   to n=6, never silently reported as if n=10.
4. **Every post-retrain evaluation window (100 samples, both
   conditions, all 16 completed runs) contained zero true labeled
   anomalies**, making post-retrain precision well-defined-but-zero
   and recall/F1 undefined in 100% of runs. This is a structural
   consequence of the 100-sample post-retrain window size combined
   with the sparsity of labeled anomalies in this series — not a
   detector-performance finding. **Detection recovery after retraining
   cannot be assessed with this metric in this experiment.**
5. The experiment uses **one** primary NAB series
   (`ec2_cpu_utilization_fe7f93.csv`); findings are not claimed to
   generalize to other NAB series or to real-world drift more broadly.
6. **CPU-only** constraints bound model/explainer complexity
   (Isolation Forest and a small `[4,2,4]` MLP autoencoder; permutation-
   based SHAP for the autoencoder rather than a gradient-based
   explainer).
7. The drift studied here is a **controlled, injected** offset
   (+3.85, locked, robust 3×IQR), not naturally occurring concept
   drift; conclusions are about this specific injected-drift design,
   not concept drift in general.
8. SHAP stability is evaluated **only** under the project's fixed
   pre-drift-only background/evaluation sample methodology (Option B);
   different sampling choices could yield different stability
   readings.
9. The strong detector-identity effect (Step 5) is observed with
   exactly **two** detector architectures; it cannot be attributed to
   "autoencoders vs. tree ensembles" in general from n=2 architecture
   types.
10. The Spearman correlation in Step 5 is reported as descriptive only
    because of confounding with detector identity, per Step 5's
    discussion — it is not used to support any causal claim in Step 10.

---

## Step 10 — Final Scientific Interpretation

**1. What happened to detection performance under drift?**
At the post-drift/pre-retrain checkpoint (the only checkpoint besides
pre-drift with valid detection metrics in the drift condition), recall
was perfect (1.0) in all 6 successful runs — the post-event labeled
anomaly window was fully captured. F1 changed in opposite directions
for the two detectors (detector_b improved, detector_a slightly
declined), so there is **no single directional answer** to "did drift
hurt detection" — it depends on detector architecture, and the overall
paired test (n=6) was not significant.

**2. What happened to explanation stability under drift?**
Immediately after drift injection (post-drift/pre-retrain, before any
retraining), explanations were **essentially unchanged** from
pre-drift (cosine ≈0.999–1.000 in every run, both conditions). Drift
injection alone, prior to retraining, does not measurably disturb
these SHAP explanations in this experiment.

**3. Did retraining show evidence of recovery?**
For detection: **cannot be determined** — post-retrain
precision/recall/F1 are undefined in all 16 completed runs (Step 3/9).
For explanation stability: the opposite of recovery was observed —
stability *relative to the pre-drift baseline* was **significantly
lower** after retraining than immediately after drift, in both the
drift condition (p=0.0312) and the no-drift condition (p=0.0020), and
in 16/16 runs individually. Retraining does not restore explanations
to their pre-drift state in this experiment; if anything it moves them
further away, and this happens whether or not real drift was present.

**4. Did explanation stability recover to the same degree as detection
performance?** This question cannot be answered comparatively, because
detection performance at the relevant post-retrain checkpoint is
undefined rather than measured. What can be said is that explanation
stability's post-retrain behavior is strongly **detector-dependent**
(detector_a collapses toward ~0, detector_b stays high, non-overlapping,
p=0.0002) — a far larger and more robust effect than anything found in
the (limited) detection-performance data.

**5. What happened when ADWIN failed to detect drift?** For seeds 101
and 303, no retraining was ever triggered (the pipeline correctly never
reaches the retrain/post-retrain checkpoints for a run with no
detection), so these runs contribute detection and explanation data
**only** at whatever checkpoints don't depend on detection — in
practice, none of the drift-condition post-detection metrics exist for
these two seeds at all. They are preserved in the results as failed
rows, not as zero/undefined placeholder successes.

**6. Conclusions strongly supported by this evidence:**
- Drift injection alone (before retraining) does not measurably disturb
  SHAP explanation stability in this setup (cosine ≈1.0 universally).
- Explanation stability relative to the pre-drift baseline degrades
  significantly after retraining, independent of whether real drift
  occurred (drift and no-drift are statistically indistinguishable on
  this measure, p=0.31).
- The observed difference in explanation stability is strongly
  associated with detector identity in this two-detector experiment —
  far stronger than any drift/no-drift or seed effect observed
  (p=0.0002, complete separation). As noted in Step 9 (limitation 9),
  this association cannot be generalized to "autoencoders vs. tree
  ensembles" in general from only two detector instances.
- ADWIN detection timing is essentially deterministic for detector_b
  across seeds but seed-sensitive for detector_a.

**7. Conclusions that remain uncertain:**
- Whether detection performance itself recovers after retraining —
  undetermined due to universal label absence in the post-retrain
  window.
- Whether the detector-identity effect on explanation stability
  reflects something general about autoencoders vs. tree-ensemble
  anomaly detectors, or is specific to this MLP architecture, this
  SHAP backend pairing (TreeExplainer vs. permutation), and this
  dataset.
- Whether the apparent correlation between post-drift precision and
  post-retrain stability (Step 5) is anything beyond the detector-type
  confound.
- All drift-condition findings are based on only 3 of 5 seeds; a
  different pair of seeds legitimately failing to detect could change
  which seeds contribute to these statistics.

No causal claims beyond these are made; this experiment's design
(n=2 detectors, 5 seeds, 1 series, 1 drift magnitude) supports
description and hypothesis generation, not strong causal inference.

---

## Step 11 — Reproducibility Record (Step 12 of the brief)

| Item | Value |
|---|---|
| River | 0.26.1 (confirmed via `import river; river.__version__` on the execution machine, Phase 12 preflight) |
| SHAP | 0.51.0 (confirmed the same way) |
| OS | Windows |
| Environment | Python virtual environment (`.venv`) |
| Dataset | Real NAB, `data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv` |
| Execution mode | CPU-only |
| Execution command | `python scripts\run_phase12_experiment.py --nab-root "E:\NAB"` |
| Python version embedded in the result JSON | **Not present** — the original `real_experiment_results.json` predates the Phase 12.1 metadata change and does not contain a `python_version`/`river_version`/`shap_version` field. Per Phase 12.1's decision, this run was **not repeated** solely to add that field; the versions above are recorded here, in the documentation, from the verified Phase 12 preflight evidence instead of being retroactively written into the existing artifact. |

This document does **not** claim that dependency versions are embedded
in `real_experiment_results.json` — they are not. `scripts/run_phase12_experiment.py`
now collects this metadata (via `collect_environment_metadata()`, added
in Phase 12.1) for any *future* run of this script; it was deliberately
not backfilled into the existing, already-analyzed result file, since
doing so would misrepresent what that file actually contains.

---

## Final Deliverables Index

| # | Deliverable | Location |
|---|---|---|
| 1 | Final analyzed results tables | `artifacts/phase13/tables/A`–`G_*.csv` |
| 2 | Statistical analysis output | `artifacts/phase13/tables/G_statistical_tests.csv` + Step 7 above |
| 3 | Final figures | `artifacts/phase13/figures/fig1`–`4_*.png` |
| 4 | Updated research documentation | This file; `README.md` updated to point here |
| 5 | Final methodology summary | Steps 1–2 above (uses locked design from `docs/PHASE11_LOCKED_PARAMETERS.md`, unchanged) |
| 6 | Final limitations section | Step 9 above |
| 7 | Final scientific interpretation | Step 10 above |
| 8 | Final conclusion | Step 10, items 6–7 |
| 9 | Reproducibility information | Step 11 above |
| 10 | Final project status | See chat: **PHASE 13 — FINAL RESEARCH COMPLETION REPORT** |

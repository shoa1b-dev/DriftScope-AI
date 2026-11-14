# Phase 11 — Real NAB Baseline Analysis and Parameter Recommendation

**Overall status: PASS**


## Series (directly inspected)

- Path: data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv
- Row count: 4032
- Timestamp range: 2014-02-14 14:27:00 to 2014-02-28 14:22:00
- Inferred frequency: 5min

## Labels (directly inspected)

- Status: PASS, n_windows: 3
  - ('2014-02-17 00:37:00.000000', '2014-02-17 11:47:00.000000')
  - ('2014-02-21 18:27:00.000000', '2014-02-22 05:37:00.000000')
  - ('2014-02-23 09:42:00.000000', '2014-02-23 20:52:00.000000')

## Event Index Validity

- Status: PASS — t_event is within valid bounds

## Anomaly Window Position Relative to t_event

- 2014-02-17 00:37:00.000000 to 2014-02-17 11:47:00.000000: before_event
- 2014-02-21 18:27:00.000000 to 2014-02-22 05:37:00.000000: before_event
- 2014-02-23 09:42:00.000000 to 2014-02-23 20:52:00.000000: after_event

## Baseline Statistics (pre-event region only, directly computed)

- baseline_start_index: 0
- baseline_end_index_exclusive: 2419
- count: 2419
- mean: 6.05346837536172
- std: 12.170641250570794
- variance: 148.12450845009542
- minimum: 1.806
- maximum: 99.668
- median: 2.654
- quantiles: {'q10': 2.05, 'q25': 2.216, 'q50': 2.654, 'q75': 3.5, 'q90': 6.1540000000000274}
- coefficient_of_variation: 2.0105236363514574
- n_labeled_anomalies_in_baseline: 2
- anomaly_treatment_note: 2 labeled anomaly window(s) overlap the baseline region (index [0, 2419)). Anomaly points are NOT excluded from these statistics — they remain part of the real pre-event data as recorded. This should be reviewed before finalizing t_event.

## Checkpoint Feasibility

- status: PASS
- t_event_window: 2410
- n_total_windows: 4023
- n_pre_event_windows: 2410
- n_post_event_windows: 1613
- minimum_post_event_required: 42
- reason: Sufficient post-event windows for all three checkpoints without overlap

## Drift Magnitude Candidates (RECOMMENDATIONS, not locked)


### 1_std_shift

- offset: 12.170641250570794
- scale_multiplier: 1.0
- expected_shifted_mean: 18.224109625932513
- distinguishable_from_noise: False
- plausible_range: True
- notes: ['A 1-std shift may be hard to distinguish from ordinary baseline variation.', "CAUTION: baseline std is computed from a region containing 2 real labeled anomaly window(s) — std may be substantially inflated above the series' typical variability. Compare against the IQR-based candidates below."]

### 2_std_shift

- offset: 24.34128250114159
- scale_multiplier: 1.0
- expected_shifted_mean: 30.39475087650331
- distinguishable_from_noise: True
- plausible_range: True
- notes: ["CAUTION: baseline std is computed from a region containing 2 real labeled anomaly window(s) — std may be substantially inflated above the series' typical variability. Compare against the IQR-based candidates below."]

### 3_std_shift

- offset: 36.51192375171238
- scale_multiplier: 1.0
- expected_shifted_mean: 42.5653921270741
- distinguishable_from_noise: True
- plausible_range: True
- notes: ["CAUTION: baseline std is computed from a region containing 2 real labeled anomaly window(s) — std may be substantially inflated above the series' typical variability. Compare against the IQR-based candidates below."]

### combined_offset_and_scale

- offset: 18.25596187585619
- scale_multiplier: 1.3
- expected_shifted_mean: 26.125470763826428
- distinguishable_from_noise: True
- plausible_range: True
- notes: ['Combined offset+scale — moderate shift on both axes, per the approved primary drift mechanism.', 'CAUTION: uses baseline std, which may be anomaly-inflated — see note above.']

### 1.5_iqr_shift_robust

- offset: 1.9259999999999997
- scale_multiplier: 1.0
- expected_shifted_mean: 4.58
- distinguishable_from_noise: True
- plausible_range: True
- notes: ['Robust candidate: uses median/IQR instead of mean/std, so it is not distorted by a small number of extreme baseline points.']

### 3_iqr_shift_robust

- offset: 3.8519999999999994
- scale_multiplier: 1.0
- expected_shifted_mean: 6.505999999999999
- distinguishable_from_noise: True
- plausible_range: True
- notes: ['Robust candidate: uses median/IQR instead of mean/std, so it is not distorted by a small number of extreme baseline points.']

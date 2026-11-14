# Phase 11 — Locked Parameters (Final)

Everything below is either an **observed real-data fact**, a **calculated value derived from real data**, a **design decision** (yours, recorded with rationale), or an explicitly labeled **limitation**. Nothing here is fabricated or estimated.

## Observed Real-Data Facts

- File: `data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv`
- Row count: 4032 (0 missing, 0 duplicate timestamps, monotonic, 5-min frequency)
- Timestamp range: 2014-02-14 14:27:00 to 2014-02-28 14:22:00
- Real labeled anomaly windows (3 total, from `combined_windows.json`):
  1. 2014-02-17 00:37:00 – 2014-02-17 11:47:00
  2. 2014-02-21 18:27:00 – 2014-02-22 05:37:00
  3. 2014-02-23 09:42:00 – 2014-02-23 20:52:00
- Baseline region [0, 2419) contains real values as high as 99.668, 75.246, 72.784 — directly overlapping windows 1 and 2. 149 of 2419 baseline points exceed 20 (typical baseline values are ~2–3.5).

## Calculated Values (from the real data above)

| Statistic | Value |
|---|---|
| Baseline count | 2419 |
| Baseline mean | 6.053 |
| Baseline std | 12.171 |
| Baseline median | 2.654 |
| Baseline min / max | 1.806 / 99.668 |
| Baseline q25 / q75 | 2.216 / 3.5 |
| Baseline IQR | 1.284 |
| Baseline CV | 2.011 |
| `t_event_window` (windowed-index boundary, window_size=10) | 2410 |
| Total windows | 4023 |
| Pre-event windows | 2410 |
| Post-event windows | 1613 |
| Window position relative to t_event=2419 | window 1: before · window 2: before · window 3: after |

## Locked Design Decisions

| Field | Old | New | Rationale | Evidence |
|---|---|---|---|---|
| `dataset.selected_series` | `null` | `ec2_cpu_utilization_fe7f93.csv` | Clean (0 missing/dup), richest label coverage (3 windows) among clean CPU-utilization series, matches CAIN 2024 continuity | Phase 10/11 inspection |
| `preprocessing.window_size` | `null` | `10` | Used throughout this validation; matches project's established default since Phase 2 | Real windowing produced 4023 windows / N_FEATURES=5 correctly |
| `drift_injection.t_inject` | `null` | `2419` | **Kept at the original 60% heuristic, explicitly NOT moved to increase post-event label coverage** — per direct instruction, this stays a documented limitation, not an optimization target | 1613 real post-event windows confirmed available |
| `drift_injection.magnitude.offset` | `null` | `3.85` | Robust 3×IQR candidate. The baseline std (12.171) is inflated by 2 real anomaly windows inside the pre-event region — using it directly would make the "drift" partly just re-injecting the baseline's own contamination. The IQR-based magnitude is not distorted by those 2 points | Real baseline IQR = 1.284; 3×1.284 = 3.85 |
| `drift_injection.sensitivity_magnitude` | (did not exist) | offset=24.34, `status: documented_not_active` | Real 2×std candidate, recorded as a future robustness check — **not** added to the active experiment matrix | Real baseline std = 12.171; 2×12.171 = 24.34 |
| `retraining.retrain_window_size` | `null` | `100` | Validated: retrain-slice clamp correctly holds at `t_event_window` for any near-event detection; 1613 available post-event windows give ample headroom for 100 | Direct boundary-math validation, this phase |
| `retraining.post_retrain_window_size` | (did not exist) | `100` | Validated: `[t_retrain, t_retrain+100)` fits comfortably within 1613 available windows (>16× margin) | Direct boundary-math validation, this phase |

**Detection window:** no separate config field exists for this; the effective ADWIN scan range is bounded by the 1613 real post-event windows, confirmed more than sufficient. Nothing to change here — retained as-is, per instruction.

## Boundary Validation Performed This Phase (real execution, not simulated)

Run directly against the real file in this environment:
1. Real drift injection at `t_inject=2419`, `offset=3.85` — confirmed `n_points_affected=1613`.
2. Real windowing — confirmed `(4023, 5)` feature matrix.
3. Real `t_event_window=2410` — matches the checkpoint-feasibility check exactly.
4. Real `PrefixFittedScaler` fit/transform on the actual data — no errors.
5. Real fixed background/evaluation samples built from the real scaled features — **both proven pre-event-only** (background max index 2310, evaluation max index 2270, both `< 2410`) and **both immutable** (`writeable=False`).
6. Real retrain-slice clamp math — confirmed it holds at `t_event_window` for a near-event detection scenario.
7. Real post-retrain window math — confirmed `[2416, 2516)`, 100 samples, well within the 1613 available.

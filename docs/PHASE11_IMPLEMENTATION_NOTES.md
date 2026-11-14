# Phase 11 — Implementation Notes

## Update: real data received and analyzed

**One real finding required extending the tool**: the baseline region (index [0, 2419)) contains 2 of the file's 3 real labeled anomaly windows, and directly checking the raw values confirmed this — 149 of 2419 baseline points exceed 20 (typical baseline values are ~2–3.5), with the top 5 values (72–99.7) landing exactly inside the two labeled windows. This means baseline mean (6.05) and std (12.17) are substantially inflated relative to the series' typical (median 2.65) behavior — not a data error, just a real property of this file. `recommend_drift_magnitudes` was extended to (a) explicitly flag std-based candidates with a `CAUTION` note whenever the baseline is anomaly-contaminated, and (b) add robust, median/IQR-based candidates as an alternative, tested directly against a small reproduction of this exact scenario.

## Original implementation summary (tool build, still applies)

`src/data/baseline_analysis.py`, `scripts/analyze_baseline_and_lock_params.py` — see below for what each part does; unchanged in structure, extended in the drift-magnitude function as described above.

## Files created/modified

`src/data/baseline_analysis.py` (extended), `scripts/analyze_baseline_and_lock_params.py`, `tests/unit/test_baseline_analysis.py` (30 tests, was 26), `docs/PHASE11_BASELINE_AND_PARAMETERS.md` (now real), `artifacts/phase11_baseline_and_parameters.json` (now real), this file.

## Test results (final, after real-data extension)

```
python -m unittest discover -s tests -v
Ran 318 tests in 24.673s
OK (skipped=6)
```
**312 passed, 6 skipped (unchanged real-dependency-unavailable tests), 0 failed.**

## Bugs found and fixed this phase

Same two as the tool-build pass (a missing dataclass argument on one error path; a wrong CV test expectation) — both caught and fixed before real data arrived. No new bugs surfaced when running against the real file.

## Configuration changes

**None yet, by design.** Two real decision points (below) need your choice before `config/experiment.yaml` is updated — locking them now would be picking on your behalf, which Phase 11 explicitly said not to do.

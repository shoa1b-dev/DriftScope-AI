# Explanation Stability Under Concept Drift

*Explanation Stability Under Concept Drift: An Empirical Evaluation of
Detection-Performance and SHAP-Attribution Recovery in CPU-Only
Time-Series Anomaly Detection.*

**Status: research complete; package reconciled and audited.**
The real experiment was executed on real NAB data with real River
(ADWIN / Page-Hinkley) and real SHAP, then analyzed and written up.

**Start here:**

## What's implemented

| Component | Status |
|---|---|
| `config/experiment.yaml` — full experiment schema | Implemented; values locked
| `src/data/` — NAB loader, validation, inspection | Implemented, unit-tested |
| `src/detectors/` — Isolation Forest, MLP autoencoder `[4,2,4]` | Implemented, unit-tested |
| `src/monitoring/` — ADWIN (primary) / Page-Hinkley (secondary) | Implemented, unit-tested |
| `src/checkpoints/` — pre_drift / post_drift_pre_retrain / post_retrain, fixed pre-drift-only SHAP samples | Implemented, unit-tested |
| `src/explanation/` — SHAP (TreeExplainer / permutation) | Implemented, unit-tested 
| `src/metrics/` — cosine + top-k Jaccard stability, precision/recall/F1 with explicit `undefined` handling | Implemented, unit-tested |
| `src/experiments/` — experiment runner + config | Implemented, unit-tested |
| `scripts/run_experiment.py` — real-experiment orchestration | Implemented; **was executed once** on real NAB + real River + real SHAP |


## Running the tests

From the project root:

```bash
python -m unittest discover -s tests -v
```

The package contains **343** test functions. 



## The real experiment 

- **20** detector-run rows attempted (2 conditions × 5 seeds × 2 detectors)
- **16 completed**, **4 failed** with `DriftNeverDetectedError` (drift
  condition, seeds 101 and 303, both detector rows) — recorded as failures,
  not imputed and not hidden. 


Dependency versions are **not** embedded inside `real_experiment_results.json`.
`requirements-lock.txt` records the versions found in the virtual environment
that was bundled with the project, which is the best available evidence.

## Reproducibility (locked)

- Seeds: `[13, 42, 101, 202, 303]` — recorded in `config/experiment.yaml`.
- `t_event = 2419` (`t_event_window = 2410` after the 10-sample feature window),
  primary drift offset `+3.85` (robust 3×IQR), sensitivity offset `+24.34`
  (documented, never used in the primary experiment).
- CPU-only throughout — no GPU dependency anywhere in this project.

## Project structure

```
README.md
pyproject.toml                  project metadata + dependency list
requirements-lock.txt           versions reconstructed from the bundled venv
.gitignore
config/experiment.yaml          locked experiment schema
src/                            implementation (data, detectors, monitoring,
                                checkpoints, explanation, metrics, experiments)
scripts/                        real-data inspection, Phase 12 runner, and
                                verify_final_package.py (read-only checker)
tests/                          unit/ + integration/ + support/
data/raw/                       place downloaded CSVs here (git-ignored)
results/                        scratch output folder (git-ignored)
```



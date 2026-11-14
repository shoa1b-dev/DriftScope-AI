# Reproducibility Guide

This guide explains how to **inspect and verify** the existing results and run
the test suite **without re-running the real experiment**.

## 1. Set up an environment (Windows PowerShell)

```powershell
python --version                      # the project was run on Python 3.11.9
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` holds the exact versions found in the original
environment (River 0.26.1, SHAP 0.51.0, ...). If an exact pin cannot be
installed on your machine, `pip install -e .` installs the unpinned list from
`pyproject.toml` instead; results may then differ slightly from the originals.

## 2. Run the unit tests (safe — does not touch the results)

```powershell
python -m unittest discover -s tests -v
```

Expected: 343 tests. Without `river`/`shap` installed, 6 are skipped (they
need the real libraries). The tests use synthetic data and temporary files;
they do not need the NAB dataset.

## 3. Verify the stored results and deliverables (read-only)

```powershell
python scripts\verify_final_package.py
```

This checks the result files against their recorded SHA-256 hashes, confirms
20 attempted / 16 completed / 4 failed (`DriftNeverDetectedError`), checks that
the JSON and CSV agree, and confirms the 7 tables, 4 figures and report are
present. It never imports `src/` and never writes anything.

You can also open the files directly:

- `artifacts/phase12/real_experiment_results.csv` — one row per detector run
- `artifacts/phase13/tables/` — tables A–G
- `artifacts/phase13/figures/` — figures 1–4
- `docs/PHASE13_FINAL_RESULTS.md` — the final report

## 4. The NAB dataset (not included)

The Numenta Anomaly Benchmark is a separate public project
(<https://github.com/numenta/NAB>). Only needed if you intend to re-run the
experiment or the inspection scripts; it is **not** needed for the tests or for
inspecting the stored results.

- The single series used is `data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv`
  (4,032 rows, 5-minute spacing) plus NAB's `labels/combined_windows.json`
  (3 labeled windows for that series).
- The runner takes the location of your NAB checkout as `--nab-root`.
- Keep the dataset outside this repository (or under `data/raw/`, whose CSVs
  are git-ignored). Check NAB's own license before redistributing any of it.

## 5. If you ever do want to re-run the experiment

Do it only deliberately, on a copy of the project, so the original artifacts
are not overwritten:

```powershell
Copy-Item -Recurse . ..\drift-stability-rerun
cd ..\drift-stability-rerun
python scripts\run_phase12_experiment.py --nab-root "E:\NAB"
```

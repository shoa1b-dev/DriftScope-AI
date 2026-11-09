# DRIFTSCOPE AI — Interactive Research Observatory

A read-only Streamlit dashboard that presents the empirical benchmark of how SHAP explanations of two anomaly
detectors behave under concept drift and retraining. It reads stored benchmark results and statistical
validation tables; it never retrains models, recomputes SHAP, or writes to the result files.

## Install and launch (Windows PowerShell, from the project root)
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```
`streamlit` and `plotly` are the only additions to the research dependencies. The dark theme is configured in
`.streamlit/config.toml` (run from the project root so it is picked up) and refined by `driftscope/theme.py`.

## Sections
| # | Section | Content |
|---|---|---|
| 01 | Research Overview | Study summary, run counts, headline statistics, scope and limitations |
| 02 | Dataset Explorer | Interactive NAB EC2 CPU series with labeled anomaly windows (requires the NAB CSV under `data/raw/`) |
| 03 | Drift Monitor | Injected drift vs ADWIN / Page-Hinkley detections per seed; non-detections stay visible |
| 04 | Detector Lab | Isolation Forest vs MLP autoencoder; undefined metrics kept undefined |
| 05 | SHAP Explorer | Stored attributions: importance, strip plot, heatmap, checkpoint and top-k comparison |
| 06 | Retraining & Recovery | Explanation stability across checkpoints with detector / condition / seed filters |
| 07 | Statistical Center | Validation results, p-values, derived effect sizes and recomputation check |

## Architecture
`app.py` → `driftscope/` (`data_loader.py` pure data access, `analysis.py` derived statistics, `cache.py`,
`theme.py` CSS and chart template, `components/ui.py`, `pages/*.py`).

## Tests
```powershell
python -m unittest discover -s tests\dashboard -t . -v
```

## Notes
- Detector B's decision threshold is 0.0, so it flags every sample; its detection metrics are shown with that context.
- SHAP values are shown as stored; feature values are not stored, so the strip plot is not colored by feature value.

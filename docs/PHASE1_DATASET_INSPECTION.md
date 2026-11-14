# Phase 1 — Dataset Inspection Status

## Environment constraint (verified, not assumed)

This development environment has no network egress. Verified two ways:

```bash
$ pip install --dry-run --break-system-packages mcp
ERROR: Could not find a version that satisfies the requirement mcp (from versions: none)
ERROR: No matching distribution found for mcp
```

```bash
$ python3 scripts/inspect_nab_candidates.py
...

## What was verified through live web retrieval (real, not fabricated)

Using web search/fetch tools,
the following was confirmed directly from the official `numenta/NAB`
GitHub repository during Phase 1:

- `data/realAWSCloudwatch/` contains these CPU-utilization series:
  `ec2_cpu_utilization_24ae8d.csv`, `_53ea38`, `_5f5533`, `_77c1ca`,
  `_825cc2`, `_ac20cd`, `_c6585a`, `_fe7f93` (plus disk/network series
  not relevant to this study).
- `ec2_cpu_utilization_24ae8d.csv` is confirmed **4,033 lines (4,032
  data rows + header), 103 KB** — from GitHub's own file metadata, not
  an estimate.
- Real labeled anomaly timestamps for that series, from NAB's official
  `labels/combined_labels.json`: `2014-02-26 22:05:00` and
  `2014-02-27 17:15:00`.
- A genuine, independently documented data-quality issue in NAB itself:
  GitHub issues #351 and #385 show `combined_labels.json` and
  `combined_windows.json` giving **inconsistent** anomaly timestamps for
  the same series. This is real, citable evidence supporting the
  "NAB labeling-quality concerns" limitation already in the locked
  proposal (Phase 16) — not something this project is claiming for the
  first time.


**No window size, drift magnitude, or final series choice is locked in
`config/experiment.yaml` until that real output is available.**

"""Result writer — persists an ExperimentResult's metadata as JSON.

STATUS: IMPLEMENTED, unit-tested (writes to a temp directory in tests
— nothing is written into the repository itself, per project policy;
`results/` stays empty until a real experiment is actually run).

Deterministic naming follows the same convention used throughout this
project: {detector}_{condition}_{seed}.json — one file per detector,
matching CheckpointRecord's own checkpoint_id pattern (Phase 4), so
results on disk are discoverable the same way checkpoints are.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.experiments.experiment_runner import ExperimentResult


def result_file_name(detector_name: str, condition: str, seed: int) -> str:
    return f"{detector_name}_{condition}_{seed}.json"


def write_experiment_result(result: ExperimentResult, output_dir: str | Path) -> dict[str, str]:
    """Writes one JSON file per detector. Returns {detector_name: path}
    for the files actually written. Raises (does not silently ignore)
    if output_dir cannot be created or a file cannot be written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for detector_name, detector_result in (
        ("detector_a", result.detector_a), ("detector_b", result.detector_b)
    ):
        filename = result_file_name(detector_name, result.condition, result.seed)
        path = output_dir / filename
        payload = {
            "condition": result.condition,
            "seed": result.seed,
            "config": result.config.to_metadata_dict(),
            "detector_result": detector_result.to_metadata_dict(),
        }
        path.write_text(json.dumps(payload, indent=2))
        written[detector_name] = str(path)

    return written


def read_experiment_result_file(path: str | Path) -> dict:
    """Reads back a written result file as a plain dict (not
    reconstructing live objects — model snapshots are never part of
    this JSON payload, consistent with the metadata-only serialization
    convention established in Phase 4/5)."""
    return json.loads(Path(path).read_text())

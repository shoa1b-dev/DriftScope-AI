"""NAB-format time-series loader.

STATUS: IMPLEMENTED and unit-tested against synthetic data (see
tests/unit/test_loader.py).

NAB CSV format (per the official repository): two columns, `timestamp`
and `value`, no index column, ISO-like timestamp strings
(e.g. "2014-02-26 21:42:00").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DatasetLoadError(Exception):
    """Raised when a NAB-format CSV cannot be loaded or is structurally invalid."""


def load_nab_series(path: str | Path) -> pd.Series:
    """Load a single NAB-format CSV into a timestamp-indexed pandas Series.

    Raises DatasetLoadError (not a silent fallback) if:
    - the file does not exist
    - the expected columns are missing
    - the timestamp column cannot be parsed
    """
    path = Path(path)
    if not path.exists():
        raise DatasetLoadError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context, not swallowed
        raise DatasetLoadError(f"Could not parse CSV at {path}: {exc}") from exc

    expected_cols = {"timestamp", "value"}
    if not expected_cols.issubset(set(df.columns)):
        raise DatasetLoadError(
            f"Expected columns {expected_cols}, found {set(df.columns)} in {path}"
        )

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as exc:  # noqa: BLE001
        raise DatasetLoadError(f"Could not parse timestamps in {path}: {exc}") from exc

    series = df.set_index("timestamp")["value"]
    series.name = path.stem
    return series

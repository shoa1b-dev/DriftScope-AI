"""Drift injection — parametric offset/scale shift.

STATUS: IMPLEMENTED, unit-tested against synthetic data.

Implements the approved proposal's PRIMARY drift mechanism (Phase 11 §3):
a parametric offset+scale shift applied from a configurable injection
point onward. This is deliberately deterministic (no randomness in the
shift arithmetic itself), which is what makes it trivially reproducible:
same series + same (t_inject, offset, scale_multiplier) always produces
bit-identical output. A `random_state` parameter is still accepted and
recorded for forward-compatibility (e.g. if stochastic magnitude jitter
is added later), but is NOT used by the current deterministic mechanism.

NAB-specific magnitude/injection-point values are NOT set anywhere in
this module or its tests — every example below uses synthetic,
arbitrary numbers to prove the mechanism works, not real dataset values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DriftInjectionError(Exception):
    """Raised for invalid injection parameters — never silently clamped."""


@dataclass(frozen=True)
class DriftInjectionRecord:
    """Every field here is something that actually happened — this is
    the log record referenced throughout the approved proposal as
    'logged, controllable parameters', not a summary invented after
    the fact."""

    t_inject: int
    injected_at_label: object       # the series' own index label at t_inject (e.g. a timestamp)
    offset: float
    scale_multiplier: float
    n_points_affected: int
    random_state: int | None


def inject_offset_scale_drift(
    series: pd.Series,
    t_inject: int,
    offset: float = 0.0,
    scale_multiplier: float = 1.0,
    random_state: int | None = None,
) -> tuple[pd.Series, DriftInjectionRecord]:
    """Apply a parametric offset+scale shift to `series` from index
    `t_inject` (inclusive) onward.

    new_value = old_value * scale_multiplier + offset   (for i >= t_inject)

    Raises DriftInjectionError (not a silent no-op or clamp) if:
    - t_inject is outside (0, len(series)) — injecting at 0 would leave
      no pre-drift baseline, injecting at/beyond the end would leave no
      post-drift region
    - scale_multiplier <= 0 (would invert or collapse the signal, not a
      meaningful "drift")
    """
    n = len(series)
    if not (0 < t_inject < n):
        raise DriftInjectionError(
            f"t_inject must satisfy 0 < t_inject < {n} (series length); got {t_inject}"
        )
    if scale_multiplier <= 0:
        raise DriftInjectionError(
            f"scale_multiplier must be > 0; got {scale_multiplier}"
        )

    modified = series.copy()
    modified.iloc[t_inject:] = modified.iloc[t_inject:] * scale_multiplier + offset

    record = DriftInjectionRecord(
        t_inject=t_inject,
        injected_at_label=series.index[t_inject],
        offset=offset,
        scale_multiplier=scale_multiplier,
        n_points_affected=n - t_inject,
        random_state=random_state,
    )
    return modified, record

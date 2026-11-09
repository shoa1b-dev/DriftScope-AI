"""Documented derived calculations (clearly labelled as derived in the UI)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_table(runs: pd.DataFrame, condition: str, col_a: str, col_b: str) -> pd.DataFrame:
    sub = runs[(runs.condition == condition) & (runs.status == "completed")]
    return sub[["detector", "seed", col_a, col_b]].dropna()


def rank_biserial_paired(a: np.ndarray, b: np.ndarray) -> float | None:
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return None
    ranks = stats.rankdata(np.abs(diff))
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 5000, seed: int = 0):
    """Percentile bootstrap 95% CI with a fixed seed. Returns None if n < 3."""
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def min_possible_wilcoxon_p(n: int) -> float:
    """Smallest two-sided exact Wilcoxon p-value attainable with n untied pairs."""
    return 2.0 / (2 ** n)


def reproduce_tests(runs: pd.DataFrame) -> pd.DataFrame:
    """Re-run the Phase 13 tests from Phase 12 per-run values as an integrity cross-check."""
    rows = []
    for cond, tag in (("drift", "DRIFT"), ("no_drift", "NO_DRIFT")):
        for m in ("cosine", "jaccard"):
            t = paired_table(runs, cond, f"{m}_vs_post_drift_mean", f"{m}_vs_post_retrain_mean")
            if len(t) >= 2:
                res = stats.wilcoxon(t.iloc[:, 2], t.iloc[:, 3])
                rows.append(dict(key=f"{tag}: {m}(post_drift) vs {m}(post_retrain)", n=len(t),
                                 statistic=float(res.statistic), p_value=float(res.pvalue)))
    done = runs[runs.status == "completed"]
    d_ = done[done.condition == "drift"].cosine_vs_post_retrain_mean
    n_ = done[done.condition != "drift"].cosine_vs_post_retrain_mean
    if len(d_) and len(n_):
        r = stats.mannwhitneyu(d_, n_, alternative="two-sided")
        rows.append(dict(key="cosine_vs_post_retrain: drift vs no_drift", n=f"{len(d_)} vs {len(n_)}",
                         statistic=float(r.statistic), p_value=float(r.pvalue)))
    a = done[done.detector == "detector_a"].cosine_vs_post_retrain_mean
    b = done[done.detector == "detector_b"].cosine_vs_post_retrain_mean
    if len(a) and len(b):
        r = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(key="cosine_vs_post_retrain: detector_a vs detector_b", n=f"{len(a)} vs {len(b)}",
                         statistic=float(r.statistic), p_value=float(r.pvalue)))
    return pd.DataFrame(rows)

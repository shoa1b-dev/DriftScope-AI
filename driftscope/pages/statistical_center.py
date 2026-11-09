import numpy as np
import plotly.graph_objects as go
import streamlit as st

from driftscope import analysis as A, cache, data_loader as L
from driftscope.components.ui import chart, notice, show_df
from driftscope.theme import C, plotly_layout


def interpret(row) -> str:
    p = row.p_value
    if p != p:
        return "p-value not available"
    tag = "below" if p < 0.05 else "above"
    return f"p = {p:.4g} is {tag} 0.05 (unadjusted; n = {row.n}). Not an effect size."


def render():
    d = cache.benchmark()
    tabs = cache.validation_tables()
    st.subheader("07 · Statistical Research Center")
    stats = tabs.get("G_statistical_tests")
    if stats is None:
        notice("warn", "Statistical validation table not found.", "The summary of test results is unavailable.")
        return
    notice("warn", "Read cautiously.",
           "Samples are small (n = 6–16). Per-run values from two detectors share seeds and the same series, so "
           "observations are not fully independent. Nine tests are shown without multiplicity correction. "
           "Wilcoxon with n = 6 cannot give p below "
           f"{A.min_possible_wilcoxon_p(6):.5f}. A p-value is not an effect size.")
    st.markdown("##### Statistical validation results")
    view = stats.copy()
    view["interpretation"] = view.apply(interpret, axis=1)
    show_df(view)
    st.markdown("##### P-value overview")
    v = stats.sort_values("p_value")
    fig = go.Figure(go.Bar(x=v.p_value, y=[f"{c[:60]} (n={n})" for c, n in zip(v.comparison, v.n)], orientation="h",
                           marker_color=[C["cyan"] if p < .05 else C["muted"] for p in v.p_value],
                           hovertemplate="%{y}<br>p=%{x:.6f}<extra></extra>"))
    fig.add_vline(x=0.05, line=dict(color=C["warn"], dash="dash"), annotation_text="0.05")
    lay = plotly_layout("p-values (log scale)", 440); lay["xaxis"]["type"] = "log"; lay["margin"]["l"] = 320
    fig.update_layout(**lay)
    chart(fig, key="pvals")

    runs = L.runs_frame(d)
    done = runs[runs.status == "completed"]
    st.markdown("##### Explanation stability comparisons (per-run values)")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for det, col in (("detector_a", C["cyan"]), ("detector_b", C["violet"])):
            s = done[done.detector == det]
            fig.add_trace(go.Box(y=s.cosine_vs_post_retrain_mean, name=det, marker_color=col, boxpoints="all"))
        fig.update_layout(**plotly_layout("Post-retrain cosine by detector (8 vs 8)", 360))
        chart(fig, key="boxdet")
    with c2:
        fig = go.Figure()
        for cond, col in (("drift", C["warn"]), ("no_drift", C["teal"])):
            s = done[done.condition == cond]
            fig.add_trace(go.Box(y=s.cosine_vs_post_retrain_mean, name=f"{cond} (n={len(s)})", marker_color=col, boxpoints="all"))
        fig.update_layout(**plotly_layout("Post-retrain cosine by condition", 360))
        chart(fig, key="boxcond")

    st.markdown("##### Effect sizes and intervals (derived in the dashboard from per-run benchmark values)")
    rows = []
    for cond in ("drift", "no_drift"):
        t = A.paired_table(runs, cond, "cosine_vs_post_drift_mean", "cosine_vs_post_retrain_mean")
        diff = (t.iloc[:, 2] - t.iloc[:, 3]).to_numpy()
        ci = A.bootstrap_ci(diff)
        rows.append(dict(comparison=f"{cond}: cosine post_drift − post_retrain (paired)", n=len(t),
                         rank_biserial=A.rank_biserial_paired(t.iloc[:, 2].to_numpy(), t.iloc[:, 3].to_numpy()),
                         mean_difference=float(np.mean(diff)),
                         ci95_low=None if ci is None else ci[0], ci95_high=None if ci is None else ci[1]))
    a = done[done.detector == "detector_a"].cosine_vs_post_retrain_mean.to_numpy()
    b = done[done.detector == "detector_b"].cosine_vs_post_retrain_mean.to_numpy()
    rows.append(dict(comparison="post-retrain cosine: detector_a vs detector_b", n=f"{len(a)} vs {len(b)}",
                     rank_biserial=A.cliffs_delta(a, b), mean_difference=float(a.mean() - b.mean()),
                     ci95_low=None, ci95_high=None))
    import pandas as pd
    show_df(pd.DataFrame(rows))
    st.caption("Rank-biserial (paired) / Cliff's delta (independent). Percentile bootstrap CI (5,000 resamples, "
               "seed 0) is for the paired mean difference; with n = 6 or 10 it is only indicative. "
               "No interval is shown where data are insufficient.")
    st.markdown("##### Independent recomputation of the tests from per-run benchmark values")
    rep = A.reproduce_tests(runs)
    show_df(rep)
    st.caption("Recomputed with SciPy; values match the validation results above.")
    sp = stats[stats.test.str.contains("Spearman")]
    if len(sp):
        notice("warn", "Spearman result is confounded.", f"{sp.note.iloc[0]}. Precision and post-retrain cosine "
               "both differ strongly by detector (n = 6), and Detector B precision is affected by the zero-threshold issue.")

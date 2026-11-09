import plotly.graph_objects as go
import streamlit as st

from driftscope import cache, data_loader as L
from driftscope.components.ui import chart, notice, show_df
from driftscope.theme import DET_COLORS, plotly_layout

LABEL = {"post_drift_pre_retrain": "post-drift / pre-retrain", "post_retrain": "post-retrain"}


def render():
    d = cache.benchmark()
    sr = L.stability_run_frame(d)
    st.subheader("06 · Retraining & Recovery Lab")
    st.write("Stability is each checkpoint's SHAP explanation compared with the fixed **pre-drift** "
             "explanation (so pre_drift is the reference and is not plotted as a measurement).")
    c1, c2, c3, c4 = st.columns(4)
    dets = c1.multiselect("Detector", ["detector_a", "detector_b"], default=["detector_a", "detector_b"])
    conds = c2.multiselect("Condition", ["drift", "no_drift"], default=["drift", "no_drift"])
    seeds = sorted(sr.seed.unique())
    ss = c3.multiselect("Seed", seeds, default=seeds)
    metric = c4.radio("Metric", ["cosine", "jaccard"], horizontal=True)
    sub = sr[sr.detector.isin(dets) & sr.condition.isin(conds) & sr.seed.isin(ss)]
    if sub.empty:
        notice("warn", "No runs match these filters.", "Failed drift seeds (101, 303) have no explanations.")
        return
    fig = go.Figure()
    for rk, g in sub.groupby("run_key"):
        g = g.set_index("checkpoint").reindex(["post_drift_pre_retrain", "post_retrain"])
        r0 = g.iloc[0]
        fig.add_trace(go.Scatter(x=[LABEL[c] for c in g.index], y=g[metric], mode="lines+markers", name=rk,
                                 line=dict(color=DET_COLORS[g.detector.iloc[0]], dash="solid" if g.condition.iloc[0] == "drift" else "dot"),
                                 hovertemplate=rk + "<br>%{x}: %{y:.4f}<extra></extra>"))
    fig.update_layout(**plotly_layout(f"{metric} similarity to pre-drift explanation (solid = drift, dotted = no-drift)", 460),
                      showlegend=len(sub.run_key.unique()) <= 12)
    chart(fig, key="recov")
    n_runs = sub.run_key.nunique()
    pv = sub.pivot(index="run_key", columns="checkpoint", values=metric)
    lower = int((pv["post_retrain"] < pv["post_drift_pre_retrain"]).sum())
    st.markdown(f"**Selected runs (n = {n_runs}):** {lower} show lower {metric} at post-retrain.")
    full = sr.pivot(index="run_key", columns="checkpoint", values="cosine")
    notice("info", "Existing finding.",
           f"Across all {len(full)} completed runs, cosine stability is lower at post-retrain than at "
           f"post-drift/pre-retrain in {int((full['post_retrain'] < full['post_drift_pre_retrain']).sum())}. "
           "This shows a consistent decrease in <i>this</i> experiment — including the no-drift control, so it "
           "is not specific to drift. It does not establish that retraining universally harms explainability: "
           "it is one dataset, two detectors, five seeds, a 100-sample retraining window and 10 explanation "
           "samples, and the post-retrain model is a different fit than the reference model.")
    show_df(sub.drop(columns=["run_key"]))

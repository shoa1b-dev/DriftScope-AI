import plotly.graph_objects as go
import streamlit as st

from driftscope import cache, data_loader as L
from driftscope.components.ui import chart, notice, show_df
from driftscope.theme import DET_COLORS, plotly_layout

DESC = {"detector_a": ("Isolation Forest", "sklearn IsolationForest; random_state = per-seed; TreeExplainer for SHAP."),
        "detector_b": ("MLPRegressor autoencoder", "sklearn MLPRegressor, hidden layers [4, 2, 4]; random_state = "
                       "per-seed; reconstruction error as score; Permutation SHAP backend.")}


def render():
    d = cache.benchmark()
    det = L.detection_frame(d)
    st.subheader("04 · Detector Lab")
    notice("info", "Model Architecture Note: Autoencoder Threshold Calibration.",
           "In the benchmark configuration the autoencoder's decision threshold is 0.0 and samples are flagged when "
           "<code>score ≥ threshold</code>. Reconstruction error is non-negative, so Detector B classifies "
           f"<b>every</b> sample as anomalous (tn = fn = 0 in {int(det[det.detector=='detector_b'].flags_all_samples.sum())} of "
           f"{int((det.detector=='detector_b').sum())} evaluated windows). Its detection metrics therefore reflect "
           "this setting and are not an indication of detection quality; explanation stability is reported "
           "separately. Threshold calibration is a natural next step and is proposed future work — it has not "
           "been evaluated here.")
    c1, c2 = st.columns(2)
    for col, k in zip((c1, c2), ("detector_a", "detector_b")):
        with col:
            st.markdown(f"##### {L.DETECTOR_LABELS[k]}")
            st.write(DESC[k][1])
            st.caption(f"Decision threshold: {sorted(float(x) for x in det[det.detector==k].threshold.unique())}")
    f1, f2, f3 = st.columns(3)
    cond = f1.selectbox("Condition", ["all", "drift", "no_drift"])
    metric = f2.selectbox("Metric", L.METRICS)
    seeds = sorted(det.seed.unique())
    seed_sel = f3.multiselect("Seeds", seeds, default=seeds)
    sub = det[det.seed.isin(seed_sel)]
    if cond != "all":
        sub = sub[sub.condition == cond]
    fig = go.Figure()
    for k in ("detector_a", "detector_b"):
        s = sub[sub.detector == k]
        g = s.groupby("checkpoint", sort=False)[metric]
        fig.add_trace(go.Box(x=s.checkpoint, y=s[metric], name=L.DETECTOR_LABELS[k], marker_color=DET_COLORS[k],
                             boxpoints="all", jitter=0.4, pointpos=0))
    lay = plotly_layout(f"{metric} by checkpoint — undefined values omitted, not plotted as 0", 420)
    fig.update_layout(**lay, boxmode="group")
    chart(fig, key="detbox")
    und = sub.groupby(["detector", "checkpoint"])[f"{metric}_undefined"].agg(["sum", "count"]).reset_index()
    und.columns = ["detector", "checkpoint", "n_undefined", "n_runs"]
    st.markdown(f"##### Undefined-{metric} counts (sample size shown)")
    show_df(und)
    st.caption("Post-retrain windows contain no labeled anomalies, so recall and F1 are undefined there. "
               "Some summary tables average undefined values as 0.0; the counts above show how many are undefined.")
    st.markdown("##### Seed-level detail with confusion matrices")
    cols = ["run_key", "checkpoint", "n_samples", "tp", "tn", "fp", "fn", "precision", "recall", "f1", "threshold"]
    show_df(sub[cols])
    with st.expander("Known limitations"):
        st.markdown("- Detector comparisons concern only these two implementations.\n"
                    "- Detector B detection metrics reflect the 0.0 decision threshold.\n"
                    "- Detection performance and explanation stability are separate questions.")

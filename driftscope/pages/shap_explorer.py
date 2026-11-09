import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driftscope import cache, data_loader as L
from driftscope.components.ui import chart, notice, show_df
from driftscope.theme import C, plotly_layout


def render():
    d = cache.benchmark()
    st.subheader("05 · SHAP Explanation Explorer")
    st.caption("Uses the SHAP attributions stored with the empirical benchmark results. Nothing is recomputed.")
    keys = list(d["full_results"].keys())
    c1, c2, c3 = st.columns(3)
    det = c1.selectbox("Detector", ["detector_a", "detector_b"], format_func=lambda k: L.DETECTOR_LABELS[k])
    cond = c2.selectbox("Condition", ["drift", "no_drift"])
    avail = [k for k in keys if k.startswith(f"{det}_{cond}_")]
    seeds = sorted(int(k.rsplit("_", 1)[1]) for k in avail)
    seed = c3.selectbox("Seed", seeds)
    key = f"{det}_{cond}_{seed}"
    e0 = d["full_results"][key]["explanations"]["pre_drift"]
    st.markdown(f'<span class="ds-tag">{e0["shap_backend"]}</span><span class="ds-tag">{e0["model_type"]}</span>'
               f'<span class="ds-tag">sample shape {e0["sample_shape"]}</span>'
               f'<span class="ds-tag">eval sample sig {e0["explanation_sample_signature"]}</span>', unsafe_allow_html=True)
    notice("info", "What is stored.",
           "10 explanation samples × 5 features per checkpoint. Feature <i>values</i> are not stored, so "
           "plots cannot be colored by feature value as in a standard SHAP beeswarm.")
    cp = st.selectbox("Checkpoint", L.CHECKPOINTS)
    A = L.attribution_matrix(d, key, cp)
    imp = A.abs().mean().sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=C["cyan"],
                           hovertemplate="%{y}: mean|SHAP|=%{x:.4f}<extra></extra>"))
    fig.update_layout(**plotly_layout(f"Mean |SHAP| — {key} @ {cp}", 340))
    chart(fig, key="shapbar")
    t1, t2 = st.tabs(["Strip plot of SHAP values", "Attribution heatmap"])
    with t1:
        fig = go.Figure()
        for f in A.columns:
            fig.add_trace(go.Scatter(x=A[f], y=[f] * len(A),
                                     mode="markers", name=f, showlegend=False,
                                     marker=dict(color=C["violet"], size=9, opacity=.8),
                                     text=[f"sample {i}" for i in range(len(A))],
                                     hovertemplate="%{text}<br>SHAP=%{x:.4f}<extra>" + f + "</extra>"))
        fig.add_vline(x=0, line_color=C["border"])
        fig.update_layout(**plotly_layout("Per-sample SHAP values by feature (not colored by feature value)", 340))
        chart(fig, key="shapstrip")
    with t2:
        fig = go.Figure(go.Heatmap(z=A.values, x=list(A.columns), y=[f"sample {i}" for i in range(len(A))],
                                   colorscale="RdBu", zmid=0, colorbar=dict(title="SHAP")))
        fig.update_layout(**plotly_layout("Attribution heatmap (samples × features)", 420))
        chart(fig, key="shapheat")
    st.markdown("##### Checkpoint comparison")
    imps = {c: L.mean_abs_importance(d, key, c) for c in L.CHECKPOINTS}
    fig = go.Figure()
    for c, col in zip(L.CHECKPOINTS, (C["cyan"], C["teal"], C["violet"])):
        fig.add_trace(go.Bar(x=imps[c].index, y=imps[c].values, name=c, marker_color=col))
    fig.update_layout(**plotly_layout("Mean |SHAP| per feature across checkpoints", 360), barmode="group")
    chart(fig, key="shapcmp")
    k = int(d["full_results"][key]["stability_vs_post_retrain"]["k"])
    rank = {c: L.topk_features(imps[c], k) for c in L.CHECKPOINTS}
    st.markdown(f"##### Top-{k} feature ranking (derived from the mean |SHAP| above)")
    show_df(pd.DataFrame({"checkpoint": list(rank), f"top-{k}": [", ".join(v) for v in rank.values()]}))
    st.caption("This aggregate ranking is a dashboard calculation. The stored Jaccard values in the stability "
               "tables are per-sample and may differ from it.")

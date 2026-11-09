import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driftscope import cache, data_loader as L
from driftscope.components.ui import chart, cards, notice, show_df
from driftscope.theme import C, DET_COLORS, plotly_layout


def _status_table(df):
    """Presentation-only view: failures stay visible with a plain-language outcome."""
    def idx(v):
        return "—" if pd.isna(v) else str(int(v))
    out = pd.DataFrame({
        "seed": df.seed.astype(int), "detector": df.detector,
        "outcome": ["Drift detected" if s == "completed" else "Not detected in scanned range" for s in df.status],
        "event window": df.t_event_window.map(idx), "ADWIN detection": df.t_detect.map(idx),
        "retrain trigger": df.t_retrain.map(idx),
        "Page-Hinkley": ["not detected" if (s == "completed" and pd.isna(v)) else idx(v)
                         for s, v in zip(df.status, df.secondary_monitor_detection_index)]})
    return out


def render():
    d = cache.benchmark()
    ev = L.drift_events(d)
    st.subheader("03 · Drift Monitor")
    notice("info", "Three different things.",
           "<b>Injected drift</b> (configured offset +3.85 at the event window), <b>detected drift</b> (ADWIN "
           "firing on each detector's own score stream), and <b>labeled anomalies</b> (NAB windows) are not "
           "interchangeable.")
    drift = ev[ev.condition == "drift"]
    ok = drift[drift.status == "completed"]
    cards([("Primary monitor", "River ADWIN", "on detector anomaly-score stream"),
           ("Secondary monitor", "Page-Hinkley", f"fired in {int(ok.secondary_monitor_detection_index.notna().sum())} of {len(ok)} drift runs"),
           ("Injected event window", str(int(ok.t_event_window.dropna().iloc[0])) if len(ok) else "n/a", "window index; offset +3.85")])
    st.write("")
    sel = st.multiselect("Detectors", ["detector_a", "detector_b"], default=["detector_a", "detector_b"])
    fig = go.Figure()
    shown = drift[drift.detector.isin(sel)].copy()
    shown["label"] = shown.detector.str[-1].str.upper() + " · seed " + shown.seed.astype(str)
    order = list(shown.label)
    inj = ok.t_event_window.dropna()
    if len(inj):
        fig.add_vline(x=float(inj.iloc[0]), line=dict(color=C["warn"], dash="dash"),
                      annotation_text="injected drift (event window)", annotation_font_color=C["warn"])
    for r in shown.itertuples():
        if r.status == "completed":
            col = DET_COLORS[r.detector]
            fig.add_trace(go.Scatter(x=[r.t_detect], y=[r.label], mode="markers", name="ADWIN detection",
                                     marker=dict(color=col, size=13, symbol="diamond"), legendgroup="adwin",
                                     showlegend=False,
                                     hovertemplate=f"{r.label}<br>ADWIN detected at window {int(r.t_detect)}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[r.t_retrain], y=[r.label], mode="markers", showlegend=False,
                                     marker=dict(color=C["teal"], size=8, symbol="line-ns-open"),
                                     hovertemplate=f"{r.label}<br>retrain trigger at {int(r.t_retrain)}<extra></extra>"))
            if r.secondary_monitor_detection_index == r.secondary_monitor_detection_index and r.secondary_monitor_detection_index is not None:
                fig.add_trace(go.Scatter(x=[r.secondary_monitor_detection_index], y=[r.label], mode="markers",
                                         showlegend=False, marker=dict(color=C["violet"], size=10, symbol="x"),
                                         hovertemplate=f"{r.label}<br>Page-Hinkley at {int(r.secondary_monitor_detection_index)}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=[2410, 4022], y=[r.label, r.label], mode="lines", showlegend=False,
                                     line=dict(color=C["bad"], dash="dot", width=2),
                                     hovertemplate=f"{r.label}: scanned 2410..4022, <b>no detection</b><extra></extra>"))
    for name, sym, col in (("ADWIN detection", "diamond", C["cyan"]), ("Retrain trigger", "line-ns-open", C["teal"]),
                           ("Page-Hinkley detection", "x", C["violet"])):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, marker=dict(symbol=sym, color=col, size=10)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="scanned, never detected",
                             line=dict(color=C["bad"], dash="dot")))
    lay = plotly_layout("Drift timeline per seed and detector (window index)", 460)
    lay["xaxis"]["title"] = "window index"
    lay["yaxis"].update(categoryorder="array", categoryarray=order[::-1])
    fig.update_layout(**lay)
    chart(fig, key="drifttl")
    path = L.find_dataset(d)
    if path is None:
        st.caption("The NAB series is not available locally, so observations are not overlaid (see Dataset Explorer).")
    failed = drift[drift.status == "failed"]
    if len(failed):
        notice("bad", f"{len(failed)} detector rows failed — {failed.seed.nunique()} seeds "
               f"({sorted(failed.seed.unique())}).",
               "ADWIN did not detect drift within the scanned range (windows 2410–4022). This is one "
               "non-detection per seed, recorded on both detector rows. No detection event is drawn for them.")
    st.markdown("##### Seed-level status")
    show_df(_status_table(ev[ev.condition == "drift"]))
    with st.expander("Labeled anomaly episodes recorded in the benchmark evaluation"):
        key = st.selectbox("Run", list(d["full_results"].keys()), key="ep_run")
        show_df(L.labeled_episodes(d, key))
        st.caption("Stored coordinates are as saved by the evaluation; 'derived_global' adds the window start "
                   "(assumes episode indices are relative to the evaluation window; pre-drift windows start at 0).")

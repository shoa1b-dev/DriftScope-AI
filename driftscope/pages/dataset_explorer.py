import plotly.graph_objects as go
import streamlit as st

from driftscope import cache, data_loader as L
from driftscope.components.ui import chart, notice
from driftscope.theme import C, plotly_layout


def render():
    d = cache.benchmark()
    st.subheader("02 · Dataset Explorer")
    path = L.find_dataset(d)
    if path is None:
        notice("warn", "NAB time series not available locally.",
               "The Numenta Anomaly Benchmark file <code>ec2_cpu_utilization_fe7f93.csv</code> is not bundled with "
               "this repository. Place it under <code>data/raw/</code> and reload to enable the interactive "
               "explorer. No substitute data is generated.")
        st.write("Dataset facts recorded in the benchmark: 4,032 observations, 5-minute sampling, "
                 f"{d.get('n_labeled_windows')} labeled anomaly windows.")
        return
    df = cache.series(str(path))
    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", f"{len(df):,}")
    step = df.timestamp.diff().mode()
    c2.metric("Sampling interval", f"{int(step.iloc[0].total_seconds() // 60)} min" if len(step) else "n/a")
    c3.metric("Span", f"{df.timestamp.min():%b %d} – {df.timestamp.max():%b %d, %Y}")
    lo, hi = st.slider("Focus time range", df.timestamp.min().to_pydatetime(), df.timestamp.max().to_pydatetime(),
                       (df.timestamp.min().to_pydatetime(), df.timestamp.max().to_pydatetime()))
    sub = df[(df.timestamp >= lo) & (df.timestamp <= hi)]
    fig = go.Figure(go.Scatter(x=sub.timestamp, y=sub.value, mode="lines", line=dict(color=C["cyan"], width=1.3),
                               name="value", hovertemplate="%{x}<br>value=%{y:.3f}<extra></extra>"))
    lp = L.find_labels()
    if lp is None:
        notice("warn", "Label file not found.",
               "The NAB label file is not present locally, so anomaly windows are not drawn. "
               "Nothing is inferred from detector output.")
    else:
        wins = L.load_label_windows(lp, d["series_relative_path"])
        for a, b in wins:
            fig.add_vrect(x0=a, x1=b, fillcolor=C["bad"], opacity=0.2, line_width=0)
        notice("info", f"{len(wins)} labeled anomaly windows", "drawn from the NAB label file (red bands).")
    fig.update_layout(**plotly_layout("EC2 CPU utilization", 460), xaxis_rangeslider_visible=True)
    chart(fig, key="series")

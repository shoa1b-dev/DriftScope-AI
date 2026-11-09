"""Design tokens, global CSS and the shared Plotly template."""
from __future__ import annotations

C = dict(bg="#0A1020", surface="#111A2E", surface2="#16223A", border="#22304D", text="#E6ECF8",
         muted="#8FA0C0", cyan="#22D3EE", teal="#2DD4BF", violet="#A78BFA", warn="#F5B84B",
         bad="#F87171", ok="#4ADE80")
DET_COLORS = {"detector_a": C["cyan"], "detector_b": C["violet"]}
COND_COLORS = {"drift": C["warn"], "no_drift": C["teal"]}
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"

CSS = f"""
<style>
/* ---------- base canvas ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
  background:{C['bg']} !important; color:{C['text']} !important; font-family:{FONT}; }}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
  background:transparent !important; }}
.stAppDeployButton, #MainMenu, footer {{ display:none !important; visibility:hidden !important; }}
.block-container {{ padding-top:2.2rem; padding-bottom:3rem; max-width:1280px; }}
h1,h2,h3,h4,h5,h6 {{ color:{C['text']} !important; letter-spacing:-0.01em; }}
p, li, label, span, div[data-testid="stMarkdownContainer"] {{ color:{C['text']}; }}
[data-testid="stCaptionContainer"], .stCaption, small {{ color:{C['muted']} !important; }}
a {{ color:{C['cyan']} !important; }}
hr {{ border-color:{C['border']} !important; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background:{C['surface']} !important; border-right:1px solid {C['border']}; }}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] [data-stale="true"] {{ opacity:1 !important; }}
[data-testid="stSidebar"] h3 {{ color:{C['cyan']} !important; letter-spacing:.14em; font-weight:700; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:.35rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding:.45rem .6rem; border-radius:8px; width:100%;
  border:1px solid transparent; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:{C['surface2']}; border-color:{C['border']}; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ color:#F4F7FD !important; font-size:.98rem !important;
  font-weight:600 !important; letter-spacing:.01em; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background:rgba(34,211,238,.12);
  border-color:rgba(34,211,238,.45); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color:{C['cyan']} !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color:#B7C4DD !important; }}

/* ---------- header / cards / notices ---------- */
.ds-header {{ display:flex; align-items:center; gap:14px; padding:16px 20px; margin-bottom:16px;
  background:{C['surface']}; border:1px solid {C['border']}; border-radius:14px; }}
.ds-title {{ font-size:1.4rem; font-weight:700; letter-spacing:.16em; color:{C['cyan']}; }}
.ds-sub {{ color:{C['muted']}; font-size:.88rem; }}
.ds-card {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px; padding:14px 16px; height:100%; }}
.ds-card:hover {{ border-color:rgba(34,211,238,.45); }}
.ds-card .label {{ color:{C['muted']}; font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; }}
.ds-card .value {{ font-size:1.6rem; font-weight:700; color:{C['text']}; margin-top:4px; }}
.ds-card .note {{ color:{C['muted']}; font-size:.8rem; margin-top:4px; }}
.ds-notice {{ border-radius:10px; padding:11px 15px; margin:10px 0; border:1px solid {C['border']};
  background:{C['surface']}; font-size:.9rem; color:{C['text']}; }}
.ds-notice.warn {{ border-left:4px solid {C['warn']}; }}
.ds-notice.bad {{ border-left:4px solid {C['bad']}; }}
.ds-notice.info {{ border-left:4px solid {C['cyan']}; }}
.ds-notice.ok {{ border-left:4px solid {C['ok']}; }}
.ds-notice b {{ color:#FFFFFF; }}
.ds-footer {{ color:{C['muted']}; font-size:.75rem; text-align:center; margin-top:2.2rem;
  padding-top:1rem; border-top:1px solid {C['border']}; }}
.ds-tag {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem;
  border:1px solid rgba(34,211,238,.35); color:#CFE9F2; margin-right:6px; background:rgba(34,211,238,.06); }}

/* ---------- code ---------- */
code, [data-testid="stMarkdownContainer"] code {{ background:{C['surface2']} !important; color:#7EE8D8 !important;
  border:1px solid {C['border']}; border-radius:5px; padding:1px 6px; }}
pre, [data-testid="stCode"], [data-testid="stCode"] pre {{ background:{C['surface']} !important; border:1px solid {C['border']}; border-radius:10px; }}

/* ---------- metrics ---------- */
[data-testid="stMetric"] {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px; padding:12px 16px; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{ color:{C['muted']} !important; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{ color:{C['text']} !important; opacity:1 !important; }}

/* ---------- inputs ---------- */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] {{ color:#C9D5EC !important; font-weight:500; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {{
  background:{C['surface2']} !important; border:1px solid {C['border']} !important; color:{C['text']} !important; border-radius:8px; }}
div[data-baseweb="select"] *, div[data-baseweb="input"] input {{ color:{C['text']} !important; }}
div[data-baseweb="select"] svg {{ fill:{C['muted']} !important; }}
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, ul[role="listbox"], div[data-baseweb="menu"] {{
  background:{C['surface2']} !important; color:{C['text']} !important; border:1px solid {C['border']}; }}
ul[role="listbox"] li, div[data-baseweb="menu"] li {{ background:{C['surface2']} !important; color:{C['text']} !important; }}
ul[role="listbox"] li:hover, div[data-baseweb="menu"] li:hover {{ background:rgba(34,211,238,.15) !important; }}
span[data-baseweb="tag"] {{ background:rgba(34,211,238,.18) !important; border:1px solid rgba(34,211,238,.4); border-radius:6px; }}
span[data-baseweb="tag"] * {{ color:{C['text']} !important; }}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{ background:{C['cyan']} !important; }}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stThumbValue"] {{ color:{C['text']} !important; }}
[data-testid="stRadio"] label p {{ color:{C['text']} !important; }}
button[kind], .stDownloadButton button {{ background:{C['surface2']} !important; color:{C['text']} !important;
  border:1px solid {C['border']} !important; border-radius:8px; }}
button[kind]:hover {{ border-color:{C['cyan']} !important; color:{C['cyan']} !important; }}

/* ---------- tabs / expanders / tables ---------- */
button[data-baseweb="tab"] {{ background:transparent !important; }}
button[data-baseweb="tab"] p {{ color:{C['muted']} !important; font-weight:600; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color:{C['cyan']} !important; }}
div[data-baseweb="tab-highlight"] {{ background:{C['cyan']} !important; }}
div[data-baseweb="tab-border"] {{ background:{C['border']} !important; }}
[data-testid="stExpander"] {{ background:{C['surface']} !important; border:1px solid {C['border']} !important; border-radius:10px; }}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {{ background:transparent !important; color:{C['text']} !important; }}
[data-testid="stExpander"] summary:hover {{ color:{C['cyan']} !important; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border:1px solid {C['border']}; border-radius:10px; overflow:hidden;
  background:{C['surface']}; }}
[data-testid="stDataFrame"] > div {{ background:{C['surface']} !important; }}
[data-testid="stTable"] table, [data-testid="stTable"] th, [data-testid="stTable"] td {{ background:{C['surface']} !important;
  color:{C['text']} !important; border-color:{C['border']} !important; }}
[data-testid="stAlert"] {{ background:{C['surface']} !important; border:1px solid {C['border']}; color:{C['text']} !important; }}
[data-testid="stPlotlyChart"], .js-plotly-plot {{ background:transparent !important; border-radius:12px; }}
::-webkit-scrollbar {{ width:10px; height:10px; }}
::-webkit-scrollbar-thumb {{ background:{C['border']}; border-radius:6px; }}
::-webkit-scrollbar-track {{ background:{C['bg']}; }}
</style>
"""


def plotly_layout(title: str | None = None, height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=C["text"])) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["surface"], height=height,
        font=dict(family=FONT, color=C["text"], size=12),
        margin=dict(l=50, r=20, t=50 if title else 20, b=45),
        xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], linecolor=C["border"]),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"], linecolor=C["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
        hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["border"], font_color=C["text"]),
    )

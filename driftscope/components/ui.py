"""Small reusable UI helpers (cards, notices, safe wrappers for Streamlit API changes)."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st


def card(label: str, value: str, note: str = "") -> str:
    return (f'<div class="ds-card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(str(value))}</div>'
            f'<div class="note">{html.escape(note)}</div></div>')


def cards(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (lab, val, note) in zip(cols, items):
        col.markdown(card(lab, val, note), unsafe_allow_html=True)


def notice(kind: str, title: str, body: str = "") -> None:
    """kind: info | warn | bad | ok."""
    st.markdown(f'<div class="ds-notice {kind}"><b>{html.escape(title)}</b> {body}</div>',
                unsafe_allow_html=True)


def chart(fig, key: str | None = None) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df: pd.DataFrame, **kw) -> None:
    """Show a frame; NaN rendered as 'undefined/missing' text, never as 0."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "undefined"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def show_df(df: pd.DataFrame, nd: int = 4) -> None:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "undefined" if pd.isna(v) else f"{v:.{nd}f}")
    table(out)


def header() -> None:
    st.markdown(
        '<div class="ds-header"><div><div class="ds-title">DRIFTSCOPE AI</div>'
        '<div class="ds-sub">Interactive Research Observatory</div></div>'
        '<div style="margin-left:auto"><span class="ds-tag">Concept Drift</span>'
        '<span class="ds-tag">Anomaly Detection</span></div></div>',
        unsafe_allow_html=True)


def footer() -> None:
    st.markdown('<div class="ds-footer">Research Implementation · CPU-Only Experimental Study</div>',
                unsafe_allow_html=True)

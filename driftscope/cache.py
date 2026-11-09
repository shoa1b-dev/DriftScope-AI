"""Cached, read-only access to artifacts for Streamlit."""
from __future__ import annotations

import streamlit as st

from driftscope import data_loader as L


@st.cache_data(show_spinner="Loading benchmark results…")
def benchmark():
    return L.read_json(L.P12_JSON)


@st.cache_data(show_spinner=False)
def benchmark_table():
    return L.read_csv(L.P12_CSV)


@st.cache_data(show_spinner=False)
def validation_tables():
    return L.load_validation_tables()


@st.cache_data(show_spinner=False)
def text(rel: str):
    return L.read_text(rel)


@st.cache_data(show_spinner="Loading time series…")
def series(path_str: str):
    from pathlib import Path
    return L.load_series(Path(path_str))

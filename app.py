"""DRIFTSCOPE AI — launch with:  streamlit run app.py   (from the project root)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="DRIFTSCOPE AI", page_icon="🛰️", layout="wide")

from driftscope import theme
from driftscope.components.ui import footer, header, notice
from driftscope.data_loader import ArtifactError
from driftscope.pages import (dataset_explorer, detector_lab, drift_monitor, overview, recovery_lab,
                       shap_explorer, statistical_center)

PAGES = {
    "01 · Research Overview": overview,
    "02 · Dataset Explorer": dataset_explorer,
    "03 · Drift Monitor": drift_monitor,
    "04 · Detector Lab": detector_lab,
    "05 · SHAP Explorer": shap_explorer,
    "06 · Retraining & Recovery": recovery_lab,
    "07 · Statistical Center": statistical_center,
}

st.markdown(theme.CSS, unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### DRIFTSCOPE AI")
    st.caption("Interactive Research Observatory")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.caption("Concept Drift | Anomaly Detection")
header()
try:
    PAGES[choice].render()
except ArtifactError as exc:
    notice("bad", "Data problem.", str(exc))
except FileNotFoundError as exc:
    notice("bad", "File not found.", str(exc))
footer()

import json

import pandas as pd
import streamlit as st

from driftscope import cache, data_loader as L
from driftscope.components.ui import notice, show_df, table


def render():
    d = cache.phase12()
    st.subheader("08 · Reproducibility & Audit Center")
    notice("info", "Read-only.", "This page never runs the experiment. Re-running "
           "<code>scripts/run_phase12_experiment.py</code> would overwrite the stored results.")
    req = cache.text("requirements-lock.txt")
    pins, py = L.parse_requirements(req or "")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Environment (from requirements-lock.txt)")
        st.write(f"Python: **{py or 'not recorded'}** · CPU-only: **True** (config)")
        keep = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "river", "shap", "PyYAML"]
        show_df(pd.DataFrame([(k, pins.get(k, "not recorded")) for k in keep], columns=["package", "version"]))
    with c2:
        st.markdown("##### Experiment configuration (config/experiment.yaml)")
        try:
            import yaml
            cfg = yaml.safe_load(cache.text("config/experiment.yaml") or "")
            st.write(f"Seeds: {cfg['reproducibility']['seeds']}")
            st.write(f"Series: `{cfg['dataset']['selected_series']}`")
            st.write(f"Drift offset: {cfg['drift_injection']['magnitude']['offset']} "
                     f"(sensitivity candidate {cfg['drift_injection']['sensitivity_magnitude']['offset']}, "
                     f"status: {cfg['drift_injection']['sensitivity_magnitude']['status']})")
            st.write(f"Retrain window: {cfg['retraining']['retrain_window_size']} · "
                     f"post-retrain window: {cfg['retraining']['post_retrain_window_size']}")
            with st.expander("Full YAML"):
                st.code(cache.text("config/experiment.yaml"), language="yaml")
        except Exception as exc:  # noqa: BLE001
            notice("warn", "Could not parse config.", str(exc))
    st.write(f"Checkpoints: {', '.join(L.CHECKPOINTS)}")
    st.markdown("##### Result file integrity (SHA-256 vs hashes recorded in scripts/verify_final_package.py)")
    exp = L.expected_hashes()
    rows = []
    for rel, want in exp.items():
        got = L.sha256_file(rel)
        rows.append(dict(file=rel, status="match" if got == want else ("missing" if got is None else "DIFFERS"),
                         sha256=(got or "")[:16] + "…"))
    show_df(pd.DataFrame(rows))
    st.markdown("##### Documented test runs (from docs/FINAL_AUDIT.md)")
    st.write("`python -m unittest discover -s tests -v` → 343 tests, OK (skipped=6) in the audit environment; "
             "the final-package verifier reported 42/42 in the audit. Different runners and environments "
             "(e.g. a later pytest run: 337 passed, 6 skipped) are not combined.")
    audit = cache.text("docs/FINAL_AUDIT.md")
    if audit:
        with st.expander("Final audit (full text)"):
            st.markdown(audit)
    st.markdown("##### Artifact browser (read-only)")
    files = L.list_artifacts()
    sel = st.selectbox("File", files)
    p = L.abspath(sel)
    if sel.endswith(".csv"):
        table(pd.read_csv(p, encoding="utf-8"))
    elif sel.endswith(".json"):
        txt = p.read_text(encoding="utf-8")
        st.code(txt[:20000] + ("\n… (truncated; download for full file)" if len(txt) > 20000 else ""), language="json")
    elif sel.endswith(".png"):
        st.image(str(p))
    else:
        st.code(p.read_text(encoding="utf-8"), language="yaml")
    st.download_button("Download this file", data=p.read_bytes(), file_name=p.name)
    rep = L.abspath("docs/PHASE13_FINAL_RESULTS.md")
    if rep.exists():
        st.download_button("Download Phase 13 final report", data=rep.read_bytes(), file_name=rep.name)

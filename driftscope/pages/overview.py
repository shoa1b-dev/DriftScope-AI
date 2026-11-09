import streamlit as st

from driftscope import cache, data_loader as L
from driftscope.components.ui import cards, notice

TITLE = ("Explanation Stability Under Concept Drift: An Empirical Evaluation of Detection-Performance "
         "and SHAP-Attribution Recovery in CPU-Only Time-Series Anomaly Detection")


def render():
    d = cache.benchmark()
    runs = L.runs_frame(d)
    stats = cache.validation_tables().get("G_statistical_tests")
    st.subheader("01 · Research Overview")
    st.markdown(f"**{TITLE}**")
    st.write("The study asks whether SHAP feature attributions of two anomaly detectors stay similar to "
             "their pre-drift explanations after injected concept drift and after sliding-window retraining.")
    failed = runs[runs.status == "failed"]
    fail_seeds = [int(s) for s in sorted(failed.seed.unique())]
    ok_drift = [int(s) for s in sorted(runs[(runs.condition == "drift") & (runs.status == "completed")].seed.unique())]
    cards([("Attempted runs", str(d["n_rows"]), "2 detectors × 5 seeds × 2 conditions"),
           ("Completed / failed", f"{d['n_completed']} / {d['n_failed']}",
            f"{len(fail_seeds)} seeds failed (same ADWIN issue): {fail_seeds}"),
           ("Successful drift seeds", str(len(ok_drift)), f"seeds {ok_drift}")])
    st.write("")
    p_dd = L.p_value(stats, "cosine_vs_post_retrain: drift vs no_drift") if stats is not None else None
    p_det = L.p_value(stats, "cosine_vs_post_retrain: detector_a vs detector_b") if stats is not None else None
    sr = L.stability_run_frame(d)
    pr = sr[sr.checkpoint == "post_retrain"].set_index("run_key").cosine
    pd_ = sr[sr.checkpoint == "post_drift_pre_retrain"].set_index("run_key").cosine
    lower = int((pr < pd_.reindex(pr.index)).sum())
    cards([("Drift vs no-drift (cosine)", f"p ≈ {p_dd:.4f}" if p_dd is not None else "n/a",
            "Mann–Whitney, 6 vs 10 · not significant at 0.05"),
           ("Explanation stability", f"{lower}/{len(pr)} lower", "post-retrain < post-drift/pre-retrain"),
           ("Detector comparison", f"p ≈ {p_det:.6f}" if p_det is not None else "n/a",
            "8 vs 8 · only these two implementations")])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Dataset")
        st.write("NAB `realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv` — 4,032 observations at 5-minute "
                 "sampling, 3 labeled anomaly windows. Primary injected drift offset: +3.85.")
        st.markdown("##### Detectors and monitors")
        st.write("A: Isolation Forest · B: MLPRegressor autoencoder (hidden layers 4-2-4) · primary monitor "
                 "River ADWIN, secondary Page-Hinkley · retrain window 100, post-retrain evaluation window 100.")
    with c2:
        st.markdown("##### Design")
        st.write("Checkpoints: pre_drift → post_drift_pre_retrain → post_retrain. SHAP background and "
                 "evaluation samples are fixed from the pre-drift period. Primary stability metric: cosine "
                 "similarity; secondary: top-k Jaccard. Seeds 13, 42, 101, 202, 303. CPU-only.")
    st.markdown("##### Executive summary")
    st.write("In every completed run, SHAP explanations at the post-retrain checkpoint were less similar to the "
             "pre-drift explanation than at the post-drift/pre-retrain checkpoint. The drift/no-drift "
             "difference in post-retrain cosine stability was not statistically distinguishable with the "
             "available sample sizes, and the two detectors differed strongly. These are observations from one "
             "dataset, one injected drift magnitude, and small samples; they do not show that retraining "
             "damages explainability in general.")
    notice("info", "Model note.",
           "The autoencoder's detection threshold is set to 0.0 while reconstruction error is non-negative, so "
           "every sample is flagged as anomalous. Its detection metrics should not be read as evidence of "
           "detection quality; explanation stability is analysed separately.")
    notice("warn", "Scope and limitations.",
           "Recall and F1 are undefined in post-retrain windows (no labeled anomalies) and are shown as undefined. "
           "Two drift seeds (101, 303) never triggered ADWIN and remain visible as such. Several comparisons use "
           "small samples and non-independent observations (both detectors share each seed).")

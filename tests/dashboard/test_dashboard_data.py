"""Tests for the DRIFTSCOPE data layer. Read-only: never touches experiment code or results."""
import hashlib
import unittest
from pathlib import Path

import numpy as np

from driftscope import analysis as A, data_loader as L

ROOT = L.ROOT
WATCHED = [L.P12_JSON, L.P12_CSV, "config/experiment.yaml"] + \
          [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / L.P13_TABLES).glob("*.csv"))]


def _hashes():
    return {r: hashlib.sha256((ROOT / r).read_bytes()).hexdigest() for r in WATCHED}


class TestLoadAndValidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = _hashes()
        cls.d = L.read_json(L.P12_JSON)
        cls.runs = L.runs_frame(cls.d)

    @classmethod
    def tearDownClass(cls):
        assert _hashes() == cls.before, "dashboard code modified a research artifact"

    def test_counts(self):
        self.assertEqual((self.d["n_rows"], self.d["n_completed"], self.d["n_failed"]), (20, 16, 4))
        self.assertEqual(L.validate_phase12(self.d), [])
        self.assertEqual(L.validate_csv_matches_json(L.read_csv(L.P12_CSV), self.d), [])

    def test_failures_visible(self):
        f = self.runs[self.runs.status == "failed"]
        self.assertEqual(len(f), 4)
        self.assertEqual(sorted(f.seed.unique()), [101, 303])
        self.assertTrue((f.error_type == "DriftNeverDetectedError").all())

    def test_undefined_not_zero(self):
        done = self.runs[self.runs.status == "completed"]
        rec = L.metric_series(done, "post_retrain", "recall")
        f1 = L.metric_series(done, "post_retrain", "f1")
        self.assertTrue(rec.isna().all() and f1.isna().all())
        self.assertEqual(int(done.post_retrain_recall_undefined.sum()), 16)
        prec = L.metric_series(done, "post_retrain", "precision")
        self.assertTrue(prec.notna().all())  # precision is defined (0.0) per stored output

    def test_failed_rows_have_no_metrics(self):
        f = self.runs[self.runs.status == "failed"]
        self.assertTrue(L.metric_series(f, "pre_drift", "f1").isna().all())

    def test_detector_b_flags_everything(self):
        det = L.detection_frame(self.d)
        b = det[det.detector == "detector_b"]
        self.assertEqual(len(b), 24)
        self.assertTrue(b.flags_all_samples.all())
        self.assertFalse(det[det.detector == "detector_a"].flags_all_samples.any())
        self.assertTrue((det.threshold == 0.0).all())

    def test_stability_all_lower_after_retrain(self):
        sr = L.stability_run_frame(self.d)
        pv = sr.pivot(index="run_key", columns="checkpoint", values="cosine")
        self.assertEqual(len(pv), 16)
        self.assertTrue((pv.post_retrain < pv.post_drift_pre_retrain).all())

    def test_attribution_shape_and_no_invention(self):
        k = "detector_a_drift_13"
        A_ = L.attribution_matrix(self.d, k, "pre_drift")
        self.assertEqual(A_.shape, (10, 5))
        raw = self.d["full_results"][k]["explanations"]["pre_drift"]["attributions"]
        self.assertTrue(np.array_equal(A_.values, np.asarray(raw)))

    def test_missing_artifact_message(self):
        with self.assertRaises(L.ArtifactError):
            L.read_json("artifacts/does_not_exist.json")

    def test_drift_events_keep_failures(self):
        ev = L.drift_events(self.d)
        self.assertEqual(int((ev.status == "failed").sum()), 4)
        self.assertTrue(ev[ev.status == "failed"].t_detect.isna().all())

    def test_phase13_stats_match_recomputation(self):
        stats = L.read_csv(f"{L.P13_TABLES}/G_statistical_tests.csv")
        rep = A.reproduce_tests(self.runs)
        for _, r in rep.iterrows():
            stored = L.p_value(stats, r.key)
            self.assertIsNotNone(stored, r.key)
            self.assertAlmostEqual(stored, r.p_value, places=6, msg=r.key)

    def test_headline_pvalues(self):
        stats = L.read_csv(f"{L.P13_TABLES}/G_statistical_tests.csv")
        self.assertAlmostEqual(L.p_value(stats, "cosine_vs_post_retrain: drift vs no_drift"), 0.3132, places=3)
        self.assertAlmostEqual(L.p_value(stats, "cosine_vs_post_retrain: detector_a vs detector_b"), 0.000155, places=6)

    def test_requirements_parse(self):
        pins, py = L.parse_requirements(L.read_text("requirements-lock.txt"))
        self.assertEqual(py, "3.11.9")
        self.assertIn("shap", pins)

    def test_find_dataset_absent_returns_none_or_path(self):
        p = L.find_dataset(self.d)
        self.assertTrue(p is None or Path(p).exists())


class TestAnalysis(unittest.TestCase):
    def test_min_wilcoxon_p(self):
        self.assertAlmostEqual(A.min_possible_wilcoxon_p(6), 0.03125)

    def test_cliffs_delta_extremes(self):
        self.assertEqual(A.cliffs_delta(np.array([2, 3]), np.array([0, 1])), 1.0)

    def test_bootstrap_requires_n3(self):
        self.assertIsNone(A.bootstrap_ci(np.array([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()

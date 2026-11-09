"""Presentation checks: public-facing text must not leak internal jargon; section 08 is gone."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UI_FILES = [ROOT / "app.py", ROOT / "driftscope" / "theme.py", ROOT / "driftscope" / "cache.py"] + \
    sorted((ROOT / "driftscope" / "pages").glob("*.py")) + sorted((ROOT / "driftscope" / "components").glob("*.py"))
FORBIDDEN = [r"explainable ai", r"phase\s*1[0-3]", r"final audit", r"sha-?256", r"DriftNeverDetectedError",
             r"verify_final_package", r"requirements-lock"]


class TestUiHygiene(unittest.TestCase):
    def test_no_forbidden_text(self):
        for f in UI_FILES:
            txt = f.read_text(encoding="utf-8")
            for pat in FORBIDDEN:
                self.assertIsNone(re.search(pat, txt, re.I), f"{pat!r} found in {f.name}")

    def test_section_08_removed(self):
        self.assertFalse((ROOT / "driftscope" / "pages" / "reproducibility.py").exists())
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("08 ·", app)
        self.assertEqual(len(re.findall(r'^\s+"0\d · ', app, re.M)), 7)

    def test_dark_theme_config_present(self):
        cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        self.assertIn('base = "dark"', cfg)

    def test_css_injected_at_app_root(self):
        self.assertIn("theme.CSS", (ROOT / "app.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

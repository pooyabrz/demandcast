"""Backups, repeatability, metadata-only cohort selection and overlap guards."""

from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
import io
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import apply_release_files as installer
import register_final_evaluation as registration


def catalogue():
    rows = []
    for item in ["FOODS_1_001", "FOODS_1_002", "FOODS_1_003", "FOODS_1_004"]:
        for store in ["CA_1", "TX_1"]:
            rows.append({"id": f"{item}_{store}_evaluation", "item_id": item,
                         "store_id": store, "unused_future_sales": len(rows)})
    return pd.DataFrame(rows)


class ReleaseToolsTests(unittest.TestCase):
    def test_installer_retains_original_documents_and_is_repeatable(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            root = Path(name)
            (root/"pyproject.toml").write_text('[project]\nname="fixture"\nversion="0"\n')
            (root/"README.md").write_text("Original core preparation command")
            (root/"README_GITHUB.md").write_text("New project presentation")
            (root/".gitignore").write_text("custom-rule\n")
            templates = root/"release_templates"
            templates.mkdir()
            (templates/"ci.yml").write_text("name: New CI\n")
            (templates/"gitignore.txt").write_text("/outputs/\n")
            workflows = root/".github/workflows"
            workflows.mkdir(parents=True)
            (workflows/"ci.yml").write_text("name: Original CI\n")
            with redirect_stdout(io.StringIO()):
                installer.apply(root)
                installer.apply(root)
            self.assertEqual((root/"README.md").read_text(), "New project presentation")
            self.assertEqual((root/"docs/release/CORE_SETUP_ORIGINAL.md").read_text(), "Original core preparation command")
            self.assertEqual((root/"docs/release/CI_BEFORE_RELEASE.yml.txt").read_text(), "name: Original CI\n")
            self.assertIn("custom-rule", (root/".gitignore").read_text())
            self.assertEqual((root/".gitignore").read_text().count(installer.START), 1)

    def test_cohort_excludes_whole_products_not_only_seen_series(self):
        result = registration.choose_cohort(catalogue(), ["FOODS_1_001_CA_1_validation"], n_items=2)
        self.assertNotIn("FOODS_1_001", result["selected_item_ids"])
        self.assertEqual(result["n_series"], 4)
        self.assertEqual(result["series_overlap_count"], 0)
        self.assertFalse(any("FOODS_1_001" in value for value in result["selected_raw_series_ids"]))

    def test_cohort_is_independent_of_row_order_and_unused_outcomes(self):
        frame = catalogue()
        known = ["FOODS_1_001_CA_1_validation"]
        expected = registration.choose_cohort(frame, known, n_items=2)
        shuffled = frame.sample(frac=1, random_state=17).copy()
        shuffled["unused_future_sales"] = 100000000
        self.assertEqual(registration.choose_cohort(shuffled, known, n_items=2), expected)

    def test_broken_catalogue_or_unknown_development_id_is_rejected(self):
        frame = catalogue()
        for invalid, known in [(pd.concat([frame, frame.iloc[:1]]), [frame.id.iloc[0]]),
                               (frame, ["unknown_series"])]:
            with self.assertRaises(ValueError):
                registration.choose_cohort(invalid, known, n_items=2)


if __name__ == "__main__":
    unittest.main()

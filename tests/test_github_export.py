"""Publication selection, immutable bytes and README checks; no GitHub writes."""

import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import build_github_export as publication
import repair_export_install_metadata as metadata_fix


def fixture_scores():
    points = {name: {"mae": value, "wape": value/4, "aggregate_volume_bias_fraction": -.1}
              for name, value in zip(publication.LABELS, [2, 1.5, 1.6, 1.2])}
    interval = {"coverage": .85, "mean_width": 3., "interval_score_80": 5., "lower_miss_rate": .02, "upper_miss_rate": .13}
    return points, interval


class PublicationChecks(unittest.TestCase):
    def test_completed_result_fixture_reconciles_and_altered_artifact_is_rejected(self):
        # Symbolic counts and tiny files test publication receipts only. They
        # are not a real M5 run, core preflight or independent model replication.
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            output = Path(name)
            for filename in publication.PUBLIC_FILES+["interval_predictions.csv", "forecasts.csv"]:
                (output/filename).write_text("explicit synthetic receipt fixture")
            cohort = {"n_items": 50, "n_series": 500, "synthetic_receipt_fixture": True}
            recipe = {"seed": 42, "horizon": 28}
            registration = {"cohort": cohort, "recipe_config": recipe}
            (output/"registration_snapshot.json").write_text(json.dumps(registration))
            bindings = {"registration_sha256": publication.hash_file(output/"registration_snapshot.json")}
            point_rows = []
            for i, model in enumerate(publication.LABELS):
                actual, predicted, count, mae = 56000, 56000-1400*i, 14000, 1+.2*i
                point_rows.append({"model": model, "n_observations": count, "mae": mae, "wape": mae*count/actual,
                                   "bias": (predicted-actual)/count, "aggregate_volume_bias_fraction": (predicted-actual)/actual,
                                   "actual_sales_total": actual, "predicted_sales_total": predicted})
            with (output/"point_overall_metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
                writer.writeheader()
                writer.writerows(point_rows)
            interval = {"method": "uncalibrated_envelope", "n_observations": 14000, "nominal_target_coverage": .8,
                        "coverage": .85, "mean_width": 3, "interval_score_80": 5, "lower_miss_rate": .02, "upper_miss_rate": .13}
            with (output/"interval_metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(interval))
                writer.writeheader()
                writer.writerow(interval)
            forecast = {"status": "forecasts_frozen", "bindings": bindings, "forecasts_sha256": publication.hash_file(output/"forecasts.csv")}
            (output/"forecast_run.json").write_text(json.dumps(forecast))
            run = {"status": "scored", "stage": "final_product_cohort", "n_series": 500, "n_items": 50,
                   "n_observations_per_model": 14000, "forecast_cutoff": "2016-04-24", "target_start": "2016-04-25",
                   "target_end": "2016-05-22", "target_sales_deserialized": True, "bindings": bindings, "cohort": cohort,
                   "recipe_config": recipe, "artifact_hashes": {filename: publication.hash_file(output/filename)
                       for filename in publication.PUBLIC_FILES+["interval_predictions.csv"]}}
            (output/"run.json").write_text(json.dumps(run))
            _, points, measured = publication.load_evidence(output)
            self.assertEqual(set(points), set(publication.LABELS))
            self.assertEqual(measured["coverage"], .85)
            (output/"interval_metrics.csv").write_text("altered summary")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                publication.load_evidence(output)

    def test_public_selection_omits_raw_sales_predictions_environments_and_archives(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            root = Path(name)
            keep = ["README.md", "pyproject.toml", "requirements-release.txt", "LICENSE",
                    "src/demandcast/models.py", "tests/fixtures/tiny.csv", "docs/protocol.md",
                    "reports/dashboard_reference/run.json", "reports/v01_diagnostics/summary.json"]
            omit = ["data/raw/m5/sales_train_evaluation.csv", "data/processed/m5.csv", "outputs/final-cohort/forecasts.csv",
                    ".env", ".git/config", ".venv/package.py", "src/demandcast/__pycache__/models.pyc", "old-addon.zip",
                    "scripts/local.env.pkl", "reports/release/project-preflight.json", "docs/.env", "reports/final_cohort_reference/stale.csv",
                    "src/demandcast.egg-info/PKG-INFO", "src/demandcast.egg-info/SOURCES.txt",
                    "src/demandcast-0.1.dist-info/METADATA", "src/legacy.egg-info"]
            for item in keep+omit:
                path = root/item
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("explicit file-selection fixture")
            found = {path.relative_to(root).as_posix() for path in publication.public_files(root)}
            self.assertEqual(found, set(keep))

    def test_snapshot_ignores_installer_metadata_but_still_checks_source_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            root = Path(name)
            model = root/"src/demandcast/models.py"
            metadata = root/"src/demandcast.egg-info/PKG-INFO"
            model.parent.mkdir(parents=True)
            metadata.parent.mkdir()
            model.write_text("frozen model source")
            metadata.write_text("old package description")
            files = publication.snapshot_hashes(root)
            self.assertEqual(set(files), {"src/demandcast/models.py"})
            manifest = root/metadata_fix.MANIFEST
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"files": files}))
            metadata.write_text("new description after editable installation")
            publication.verify_export(root)
            model.write_text("changed model source")
            with self.assertRaisesRegex(ValueError, "Export file changed"):
                publication.verify_export(root)

    def legacy_export_fixture(self, root):
        files = {}
        for name, mapping in metadata_fix.TOOL_UPDATES.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path)
            files[name] = mapping["before"]
        own = root/"scripts/repair_export_install_metadata.py"
        shutil.copyfile(ROOT/"scripts/repair_export_install_metadata.py", own)
        for name, text in [("src/demandcast/models.py", "frozen model source"),
                           ("reports/final_cohort_reference/interval_metrics.csv", "frozen result table"),
                           ("src/demandcast.egg-info/PKG-INFO", "original installation metadata")]:
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            files[name] = publication.hash_file(path)
        manifest = root/metadata_fix.MANIFEST
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"files": files, "final_run_sha256": "fixture-final-run"}))
        return manifest, files

    def test_legacy_repair_preserves_frozen_hashes_backup_and_installed_metadata(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            root = Path(name)
            manifest, old_files = self.legacy_export_fixture(root)
            original = manifest.read_bytes()
            installed = root/"src/demandcast.egg-info/PKG-INFO"
            installed.write_text("rewritten by editable installation")
            metadata_fix.repair(root)
            repaired = json.loads(manifest.read_bytes())
            self.assertNotIn("src/demandcast.egg-info/PKG-INFO", repaired["files"])
            self.assertEqual((root/metadata_fix.BACKUP).read_bytes(), original)
            self.assertEqual(installed.read_text(), "rewritten by editable installation")
            for name in ["src/demandcast/models.py", "reports/final_cohort_reference/interval_metrics.csv"]:
                self.assertEqual(repaired["files"][name], old_files[name])
            before_second = manifest.read_bytes()
            metadata_fix.repair(root)
            self.assertEqual(manifest.read_bytes(), before_second)

    def test_legacy_repair_rejects_changed_model_or_result_before_writing(self):
        for target in ["src/demandcast/models.py", "reports/final_cohort_reference/interval_metrics.csv"]:
            with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
                root = Path(name)
                manifest, _ = self.legacy_export_fixture(root)
                original = manifest.read_bytes()
                (root/target).write_text("unexpected frozen-file change")
                with self.assertRaisesRegex(ValueError, "Frozen export file changed"):
                    metadata_fix.repair(root)
                self.assertEqual(manifest.read_bytes(), original)
                self.assertFalse((root/metadata_fix.BACKUP).exists())

    def test_readme_separates_completed_final_results_from_development(self):
        points, interval = fixture_scores()
        text = ("# DemandCast\n\nThe current results are\ndevelopment validation; an additional outcome-uninspected product cohort is\n"
                "registered separately before its final evaluation.\n\n## Results\n\nPreserved development evidence.\n\n"
                "## Final evaluation\n\nPending final score.\n\n## Repository layout\n\nPreserved structure.\n")
        updated = publication.updated_readme(text, points, interval)
        self.assertIn("## Development validation", updated)
        self.assertIn("Preserved development evidence.", updated)
        self.assertIn("## Final product-cohort evaluation", updated)
        self.assertIn("85.00%", updated)
        self.assertNotIn("Pending final score.", updated)
        self.assertNotIn("registered separately before its final evaluation.", updated)
        self.assertTrue(updated.endswith("Preserved structure.\n"))
        with self.assertRaises(ValueError):
            publication.updated_readme("Customized README without expected sections", points, interval)
        original_ci = "name: Fixture\nsteps:\n  - run: "+publication.INSTALL+"\n"
        updated_ci = publication.publication_workflow(original_ci)
        self.assertIn(publication.PUBLIC_INSTALL, updated_ci)
        self.assertEqual(publication.publication_workflow(updated_ci), updated_ci)

    def test_changed_forecast_code_or_config_is_not_published_as_the_frozen_recipe(self):
        run = {"bindings": {"source_hashes": {"src/demandcast/models.py": "original-fixture-hash"}},
               "recipe_config": {"seed": 42, "horizon": 28, "models": ["seasonal_naive", "mean28", "lightgbm", "lightgbm_l1"]}}
        report = {"source_hashes": run["bindings"]["source_hashes"].copy()}
        config = {"seed": 42, "horizon": 28, "models": ["seasonal_naive", "mean28", "lightgbm"]}
        publication.check_checkout(report, run, config)
        changed = copy.deepcopy(report)
        changed["source_hashes"]["src/demandcast/models.py"] = "changed"
        for bad_report, bad_config in [(changed, config), (report, {**config, "seed": 7})]:
            with self.assertRaises(ValueError):
                publication.check_checkout(bad_report, run, bad_config)

    def test_git_round_trip_preserves_reference_bytes_with_autocrlf(self):
        if not shutil.which("git"):
            self.skipTest("Git is not installed")
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            base = Path(name)
            source, cloned = base/"source", base/"clone"
            source.mkdir()
            (source/".gitattributes").write_text(publication.ATTRIBUTES, encoding="utf-8")
            path = source/"reports/dashboard_reference/intervals/run.json"
            path.parent.mkdir(parents=True)
            real_reference = ROOT/"reports/dashboard_reference/intervals/run.json"
            path.write_bytes(real_reference.read_bytes())
            digest = publication.hash_file(path)
            for command in [["git", "init", "-b", "main"], ["git", "-c", "core.autocrlf=true", "add", "."],
                            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "byte-preservation fixture"]]:
                subprocess.run(command, cwd=source, capture_output=True, check=True)
            subprocess.run(["git", "-c", "core.autocrlf=true", "clone", str(source), str(cloned)], capture_output=True, check=True)
            self.assertEqual(publication.hash_file(cloned/"reports/dashboard_reference/intervals/run.json"), digest)

    def test_zero_sales_chart_and_regular_chart_render_without_claiming_test_results_as_m5(self):
        points, _ = fixture_scores()
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            root = Path(name)
            path = root/"synthetic-chart.png"
            publication.render_chart(points, path, preview=True)
            self.assertGreater(path.stat().st_size, 1000)
            for point in points.values():
                point["aggregate_volume_bias_fraction"] = None
            publication.render_chart(points, root/"synthetic-zero-sales-chart.png", preview=True)
            self.assertGreater((root/"synthetic-zero-sales-chart.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

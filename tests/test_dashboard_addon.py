"""Source reconciliation, weighted filters and Streamlit interaction checks."""

from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("dashboard_data", ROOT/"dashboard/dashboard_data.py")
DATA = module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def toy_forecasts():
    rows = []
    for cutoff in ["2016-01-31", "2016-02-28", "2016-03-27"]:
        for series in range(8):
            for horizon in range(1, 29):
                rows.append({"unique_id": f"FOODS_{series}", "cutoff": cutoff,
                             "ds": pd.Timestamp(cutoff)+pd.Timedelta(days=horizon), "horizon": horizon,
                             "y": 2*((series+horizon) % 2), "mean_estimate": 1.5,
                             "median_estimate": 1.0, "q10_raw": 0.0, "q90_raw": 2.0,
                             "lower": 0.0, "upper": 2.0})
    return pd.DataFrame(rows)


def toy_contract(payload):
    return {"cutoffs": ["2016-01-31", "2016-02-28", "2016-03-27"],
            "n_series": 8, "horizon": 28, "expected_sha256": hashlib.sha256(payload).hexdigest()}


def create_toy_project(root):
    shutil.copytree(ROOT/"dashboard", root/"dashboard", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT/"reports", root/"reports")
    ref = root/"reports/dashboard_reference"
    e02 = json.loads((ref/"intervals/run.json").read_text())
    e02["n_series"] = e02["config"]["n_series"] = 8
    e02_bytes = json.dumps(e02).encode()
    (ref/"intervals/run.json").write_bytes(e02_bytes)
    frame = toy_forecasts()
    payload = frame.to_csv(index=False).encode()
    output = root/"outputs/interval-validation"
    output.mkdir(parents=True)
    (output/"interval_predictions.csv").write_bytes(payload)
    e03 = json.loads((ref/"calibration/run.json").read_text())
    e03["e02_run_sha256"] = hashlib.sha256(e02_bytes).hexdigest()
    e03["interval_predictions_sha256"] = hashlib.sha256(payload).hexdigest()
    e03["config"] = e02["config"]
    e03["n_evaluation_observations_per_method"] = 448
    (ref/"calibration/run.json").write_text(json.dumps(e03))
    common = {"nominal_target_coverage": .8, "coverage": 1., "lower_miss_rate": 0., "upper_miss_rate": 0.,
              "mean_width": 2., "interval_score_80": 2., "mean_expansion": 0., "median_outside_rate": 0.}
    pd.DataFrame([{"method": m, "n_observations": 448, **common} for m in ["uncalibrated", "temporal_cqr"]]).to_csv(ref/"calibration/evaluation_metrics.csv", index=False)
    pd.DataFrame([{"method": m, "cutoff": c, "n_observations": 224, **common}
                  for m in ["uncalibrated", "temporal_cqr"] for c in e03["evaluation_cutoffs"]]).to_csv(ref/"calibration/evaluation_by_fold.csv", index=False)
    pd.DataFrame([{"method": m, "predicted_mean_bin": "1-2", "n_observations": 448, **common}
                  for m in ["uncalibrated", "temporal_cqr"]]).to_csv(ref/"calibration/evaluation_by_predicted_mean.csv", index=False)
    audit = pd.read_csv(ref/"calibration/calibration_table.csv")
    audit["n_calibration"] = audit.n_calibration//500*8
    audit["order_statistic_rank"] = np.ceil((audit.n_calibration+1)*.8).astype(int)
    audit.to_csv(ref/"calibration/calibration_table.csv", index=False)
    pd.DataFrame([{"model": name, "mae": 1., "wape": 1., "bias": bias, "n_observations": 672}
                  for name, bias in [("lightgbm", .5), ("lightgbm_l1", 0.), ("mean28", 0.), ("seasonal_naive", -1.)]]).to_csv(ref/"intervals/point_overall_metrics.csv", index=False)


class DashboardDataTests(unittest.TestCase):
    def test_filters_pool_with_observation_weights(self):
        frame = pd.DataFrame([{**dict.fromkeys(DATA.MEASURES, 1.), "coverage": .5, "n_observations": 10},
                              {**dict.fromkeys(DATA.MEASURES, 1.), "coverage": .9, "n_observations": 90}])
        result = DATA.weighted_metrics(frame)
        self.assertAlmostEqual(result["coverage"], .86)
        self.assertEqual(result["n_observations"], 100)
        with self.assertRaises(ValueError):
            DATA.weighted_metrics(frame.iloc[:0])

    def test_real_summary_receipts_and_populations(self):
        frames, receipts, provenance = DATA.load_bundle(ROOT)
        self.assertEqual(set(frames["overall"].n_observations), {28000})
        self.assertEqual(set(frames["points"].n_observations), {42000})
        self.assertEqual(len(provenance), 5)
        self.assertFalse(receipts["e03"]["holdout_evaluated"])
        l1 = frames["points"].set_index("model").loc["lightgbm_l1"]
        self.assertAlmostEqual(l1.volume_bias_pct, -34.50305261, places=6)

    def test_changed_source_receipt_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            folder = Path(name)
            shutil.copytree(ROOT/"reports", folder/"reports")
            path = folder/"reports/dashboard_reference/calibration/run.json"
            receipt = json.loads(path.read_text())
            receipt["e02_run_sha256"] = "changed"
            path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                DATA.load_bundle(folder)

    def test_row_source_hash_and_dates_are_checked(self):
        frame = toy_forecasts()
        payload = frame.to_csv(index=False).encode()
        loaded = DATA.load_forecasts(payload, toy_contract(payload))
        self.assertEqual(len(loaded), 672)
        with self.assertRaises(ValueError):
            DATA.load_forecasts(payload+b"\n", toy_contract(payload))
        frame.loc[0, "ds"] += pd.Timedelta(days=1)
        wrong = frame.to_csv(index=False).encode()
        with self.assertRaises(ValueError):
            DATA.load_forecasts(wrong, toy_contract(wrong))

    def test_zero_sales_wape_is_undefined(self):
        frame = pd.DataFrame({"y": [0, 0], "lower": [0, 0], "upper": [1, 1],
                              "median_estimate": [0, 0], "mean_estimate": [.5, .5]})
        result = DATA.series_metrics(frame)
        self.assertTrue(np.isnan(result["mean_wape"]))
        self.assertEqual(result["mean_mae"], .5)
        self.assertEqual(result["coverage"], 1.)


class DashboardInteractionTests(unittest.TestCase):
    def app(self, root=ROOT):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(root/"dashboard/app.py", default_timeout=20).run()
        # Assertions below use the shipped reference results, independently of
        # whether this checkout already contains generated project outputs.
        source = app.selectbox(key="result_source")
        if source.value != "Bundled validation results":
            source.set_value("Bundled validation results").run()
        return app

    def assert_clean(self, app):
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

    def test_default_view_and_single_window_filter(self):
        app = self.app()
        self.assert_clean(app)
        self.assertEqual(app.metric[0].value, "89.36%")
        self.assertEqual(app.metric[3].value, "28,000")
        app.multiselect(key="interval_cutoffs").set_value(["2016-03-27"]).run()
        self.assert_clean(app)
        self.assertEqual(app.metric[0].value, "89.52%")
        self.assertEqual(app.metric[3].value, "14,000")
        app.multiselect(key="interval_cutoffs").set_value([]).run()
        self.assert_clean(app)
        self.assertEqual(len(app.metric), 0)

    def test_predicted_mean_filter_and_empty_selection(self):
        app = self.app()
        app.radio(key="review_view").set_value("Demand segments").run()
        app.multiselect(key="mean_bins").set_value(["2+"]).run()
        self.assert_clean(app)
        self.assertEqual(app.metric[0].value, "85.98%")
        self.assertEqual(app.metric[1].value, "8.583")
        self.assertEqual(app.metric[2].value, "12.133")
        app.multiselect(key="mean_bins").set_value([]).run()
        self.assert_clean(app)
        self.assertEqual(len(app.metric), 0)

    def test_point_view_and_missing_row_file(self):
        app = self.app()
        app.radio(key="review_view").set_value("Point models").run()
        self.assert_clean(app)
        self.assertTrue(any("42,000" in c.value for c in app.caption))
        app.radio(key="review_view").set_value("Series explorer").run()
        self.assert_clean(app)
        # The real project may have a valid default forecast CSV. Explicitly
        # select a nonexistent path to exercise the missing-file branch.
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            missing = Path(name)/"interval_predictions.csv"
            self.assertFalse(missing.exists())
            app.text_input(key="forecast_path").set_value(str(missing)).run()
            self.assert_clean(app)
            self.assertTrue(any("aggregate views are ready" in i.value for i in app.info))
            self.assertEqual(len(app.get("plotly_chart")), 0)

    def test_series_view_filters_and_figure_with_synthetic_rows(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            folder = Path(name)
            create_toy_project(folder)
            reference = folder/"reports/dashboard_reference"
            shutil.copytree(reference/"calibration", folder/"outputs/calibration-validation")
            for path in (reference/"intervals").iterdir():
                shutil.copyfile(path, folder/"outputs/interval-validation"/path.name)
            app = self.app(folder)
            self.assert_clean(app)
            self.assertEqual(app.selectbox(key="result_source").value, "Bundled validation results")
            app.radio(key="review_view").set_value("Series explorer").run()
            self.assert_clean(app)
            self.assertEqual(app.metric[3].value, "28")
            app.slider(key="series_horizons").set_value((1, 7)).run()
            self.assert_clean(app)
            self.assertEqual(app.metric[3].value, "8")
            self.assertEqual(len(app.get("plotly_chart")), 1)
            # Regression: the missing-file check must also work when a valid
            # forecast CSV exists at the project's default path.
            app.text_input(key="forecast_path").set_value(str(folder/"missing.csv")).run()
            self.assert_clean(app)
            self.assertTrue(any("aggregate views are ready" in i.value for i in app.info))
            self.assertEqual(len(app.get("plotly_chart")), 0)


if __name__ == "__main__":
    unittest.main()

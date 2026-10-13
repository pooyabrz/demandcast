"""Small end-to-end tests using generated data only."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from demandcast.cli import main
from demandcast.config import Config
from demandcast.data import prepare_m5, synthetic_sales, to_panel
from demandcast.evaluation import forecast_latest, run_backtest


class PipelineIntegration(unittest.TestCase):
    def test_m5_import_samples_ids_without_using_target_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            days = [f"d_{day}" for day in range(1, 101)]
            sales = pd.DataFrame({
                "id": [f"ITEM_{i}_STORE_evaluation" for i in range(5)],
                "item_id": [f"ITEM_{i}" for i in range(5)],
                "dept_id": "DEPT", "cat_id": "CAT", "store_id": "STORE", "state_id": "STATE",
                **{day: np.arange(5) + index % 7 for index, day in enumerate(days)},
            })
            sales.to_csv(root / "sales_train_evaluation.csv", index=False)
            pd.DataFrame({"d": days, "date": pd.date_range("2020-01-01", periods=100)}).to_csv(
                root / "calendar.csv", index=False)
            config = Config(n_series=2, history_days=70, horizon=7, n_windows=1)
            first = prepare_m5(root, root / "first.csv", config)
            changed = sales.copy()
            changed[days] = 999
            changed.to_csv(root / "sales_train_evaluation.csv", index=False)
            second = prepare_m5(root, root / "second.csv", config)
            self.assertEqual(first["selected_ids"], second["selected_ids"])
            self.assertNotEqual(first["source_sales_sha256"], second["source_sales_sha256"])
            prepared = pd.read_csv(root / "first.csv")
            self.assertEqual(len(prepared), 2 * 70)
            self.assertEqual(len(to_panel(prepared).dates), 70)
            self.assertEqual(first["dataset_kind"], "real_m5_subset")

    def test_cli_marks_synthetic_results_and_writes_required_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "tiny.yaml"
            config.write_text("horizon: 7\nn_windows: 1\nn_series: 2\nhistory_days: 100\n"
                              "models: [seasonal_naive, mean28]\n", encoding="utf-8")
            main(["demo", "--config", str(config), "--output", str(root / "run")])
            manifest = json.loads((root / "run/run.json").read_text())
            self.assertEqual(manifest["dataset_kind"], "synthetic")
            self.assertTrue(manifest["synthetic_results_are_not_benchmark_evidence"])
            for filename in ["metrics.csv", "horizon_metrics.csv", "series_metrics.csv", "predictions.csv"]:
                self.assertTrue((root / "run" / filename).is_file())

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed")
    def test_histgbm_backtest_and_future_forecast(self):
        panel = to_panel(synthetic_sales(3, 120, 42))
        config = Config(horizon=7, n_windows=1, iterations=8, threads=2,
                        training_origin_days=56, models=("histgbm",))
        predictions = run_backtest(panel, config)
        self.assertEqual(len(predictions), 3 * 7)
        self.assertTrue(np.isfinite(predictions["prediction"]).all())
        future = forecast_latest(panel, config, "histgbm")
        self.assertTrue((future["ds"] > panel.dates[-1]).all())
        self.assertTrue((future["prediction"] >= 0).all())

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM not installed")
    def test_lightgbm_real_fit_and_forecast(self):
        panel = to_panel(synthetic_sales(3, 120, 42))
        config = Config(horizon=7, n_windows=1, iterations=8, threads=2,
                        training_origin_days=56, models=("lightgbm",))
        future = forecast_latest(panel, config, "lightgbm")
        self.assertEqual(len(future), 21)
        self.assertTrue(np.isfinite(future["prediction"]).all())


if __name__ == "__main__":
    unittest.main()

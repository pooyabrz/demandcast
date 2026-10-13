"""Prediction-bound, metric, frozen-control and quantile-component checks."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("interval_experiment", ROOT / "experiments/run_interval_benchmark.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ToyFeatures:
    def __init__(self, panel):
        self.panel = panel
        self.names = ["observed_level", "horizon"]
        self.categorical_indices = []

    def training_matrix(self, horizon, origin_days, stride):
        origins = np.arange(10, len(self.panel.dates)-horizon, stride)
        x = np.concatenate([self.at_origin(int(origin), horizon) for origin in origins])
        y = np.concatenate([self.panel.values[:, origin+1:origin+horizon+1].ravel() for origin in origins])
        return x, y, origins

    def at_origin(self, origin, horizon):
        return np.column_stack([np.repeat(self.panel.values[:, origin], horizon),
                                np.tile(np.arange(1, horizon+1), len(self.panel.ids))])


def paired_fixture():
    records = []
    values = {"seasonal_naive": [3, 4, 1], "mean28": [3, 4, 1],
              "lightgbm": [3, 4, 1], "lightgbm_l1": [3, 4, 1],
              "lightgbm_q10": [4, 1, 0], "lightgbm_q90": [2, 2, 1]}
    cutoff = pd.Timestamp("2024-01-01")
    for model, forecasts in values.items():
        for index, (actual, forecast) in enumerate(zip([3, 4, 0], forecasts)):
            records.append({"unique_id": "FOODS_1_001_CA_1_evaluation", "cutoff": cutoff,
                            "ds": cutoff+pd.Timedelta(days=index+1), "horizon": index+1,
                            "stage": "validation", "model": model, "y": actual, "prediction": forecast})
    return pd.DataFrame(records)


class IntervalChecks(unittest.TestCase):
    def test_declared_repair_preserves_median_and_reports_changed_coverage(self):
        frame = experiment.interval_frame(paired_fixture())
        np.testing.assert_array_equal(frame.lower, [2, 1, 0])
        np.testing.assert_array_equal(frame.upper, [4, 4, 1])
        np.testing.assert_array_equal(frame.median_estimate, [3, 4, 1])
        self.assertEqual(frame.raw_quantile_crossing.tolist(), [True, False, False])
        self.assertEqual(frame.covered_raw.tolist(), [False, False, True])
        self.assertTrue(frame.covered.all())
        self.assertTrue((frame.lower <= frame.median_estimate).all())

    def test_metrics_include_width_pinball_and_raw_vs_repaired_coverage(self):
        frame = experiment.interval_frame(paired_fixture())
        metrics = experiment.summarize_intervals(frame).iloc[0]
        self.assertAlmostEqual(metrics.coverage_raw, 1/3)
        self.assertAlmostEqual(metrics.coverage, 1)
        self.assertAlmostEqual(metrics.mean_width, 2)
        self.assertAlmostEqual(metrics.pinball_q10, (0.9+0.3+0)/3)
        self.assertEqual(metrics.n_observations, 3)

    def test_mismatched_actuals_or_missing_model_are_rejected(self):
        frame = paired_fixture()
        invalid = frame[frame.model != "lightgbm_q90"]
        with self.assertRaises(ValueError):
            experiment.interval_frame(invalid)
        frame.loc[frame.model == "lightgbm_q10", "y"] = 100
        with self.assertRaisesRegex(ValueError, "Different actual"):
            experiment.interval_frame(frame)

    def test_real_quantile_estimator_uses_requested_tail(self):
        config = SimpleNamespace(horizon=7, training_origin_days=56, origin_stride=7,
                                 iterations=8, threads=2, seed=42)
        rng = np.random.default_rng(42)
        panel = SimpleNamespace(ids=np.array(["A", "B"]),
                                dates=pd.date_range("2024-01-01", periods=90),
                                values=rng.poisson(np.array([2, 8])[:, None], (2, 90)).astype(float))
        for level in [0.1, 0.9]:
            model = experiment.QuantileDirectTree(config, ToyFeatures, level).fit(panel)
            result = model.predict(7)
            self.assertEqual(model.estimator.get_params()["objective"], "quantile")
            self.assertEqual(model.estimator.get_params()["alpha"], level)
            self.assertEqual(result.shape, (2, 7))
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue((result >= 0).all())

    def test_frozen_control_comparison_rejects_a_changed_result(self):
        expected = pd.DataFrame([{"model":"lightgbm_l1", "cutoff":"2016-01-31", "mae":1.0,
                                  "wape":0.5, "bias":-0.1, "n_observations":14000}])
        passed, _ = experiment.compare_controls(expected, expected)
        self.assertTrue(passed)
        altered = expected.copy()
        altered["mae"] = 1.2
        passed, _ = experiment.compare_controls(altered, expected)
        self.assertFalse(passed)

    def test_frozen_reference_accepts_original_config_and_rejects_new_data(self):
        base = SimpleNamespace(models=["seasonal_naive", "mean28", "lightgbm"], horizon=28, seed=42)
        panel = SimpleNamespace(ids=["A", "B"], dates=list(range(90)))
        reference = {"stage":"validation", "holdout_evaluated":False,
                     "control_reproduced_within_tolerance":True, "source_sha256":"original",
                     "config":{"models":experiment.POINT_MODELS, "horizon":28, "seed":42},
                     "n_series":2, "observed_days":90}
        experiment.check_frozen_reference(reference, base, panel, "original")
        with self.assertRaisesRegex(ValueError, "Prepared input differs"):
            experiment.check_frozen_reference(reference, base, panel, "changed")
        base.horizon = 14
        with self.assertRaisesRegex(ValueError, "Configuration changed"):
            experiment.check_frozen_reference(reference, base, panel, "original")


if __name__ == "__main__":
    unittest.main()

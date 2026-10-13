"""Focused model-component and frozen-reference checks, without real M5 history."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("l1_experiment", ROOT / "experiments/run_l1_ablation.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class TinyFeatures:
    """Explicit injected fixture; not a replacement for DemandCast's real feature code."""
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
        return np.column_stack([
            np.repeat(self.panel.values[:, origin], horizon),
            np.tile(np.arange(1, horizon+1), len(self.panel.ids)),
        ])


class L1ExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(horizon=7, training_origin_days=56, origin_stride=7,
                                      iterations=12, threads=2, seed=42)
        rng = np.random.default_rng(42)
        self.panel = SimpleNamespace(ids=np.array(["A", "B"]),
                                     dates=pd.date_range("2024-01-01", periods=90),
                                     values=rng.poisson(np.array([2, 8])[:, None], (2, 90)).astype(float))

    def test_l1_model_fits_real_estimator_and_predicts_nonnegative_horizons(self):
        model = experiment.L1DirectTree(self.config, TinyFeatures).fit(self.panel)
        predictions = model.predict(7)
        self.assertEqual(model.estimator.get_params()["objective"], "regression_l1")
        self.assertEqual(predictions.shape, (2, 7))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        with self.assertRaises(ValueError):
            model.predict(8)

    def test_all_zero_training_returns_zero_forecasts(self):
        self.panel.values[:] = 0
        model = experiment.L1DirectTree(self.config, TinyFeatures).fit(self.panel)
        np.testing.assert_array_equal(model.predict(7), np.zeros((2, 7)))

    def test_changed_data_or_settings_cannot_enter_frozen_ablation(self):
        reference = {"stage":"validation", "source_sha256":"original",
                     "config":{"horizon":7, "seed":42}, "n_series":2, "observed_days":90}
        experiment.check_reference(reference, self.config, self.panel, "original")
        with self.assertRaisesRegex(ValueError, "Data differs"):
            experiment.check_reference(reference, self.config, self.panel, "changed")
        self.config.horizon = 14
        with self.assertRaisesRegex(ValueError, "Configuration changed"):
            experiment.check_reference(reference, self.config, self.panel, "original")


if __name__ == "__main__":
    unittest.main()

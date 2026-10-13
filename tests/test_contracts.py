"""Tests for data boundaries, future-data isolation, and metric semantics."""

import unittest

import numpy as np
import pandas as pd

from demandcast.config import Config
from demandcast.data import synthetic_sales, to_panel
from demandcast.evaluation import cutoff_indices, metrics, run_backtest
from demandcast.features import OriginFeatures
from demandcast.models import Baseline


class ForecastContracts(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_sales(3, 150, 42)
        self.panel = to_panel(self.frame)
        self.config = Config(horizon=14, n_windows=2, models=("seasonal_naive",))

    def test_rejects_duplicate_observations(self):
        duplicated = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            to_panel(duplicated)

    def test_rejects_missing_series_day(self):
        with self.assertRaisesRegex(ValueError, "complete calendar"):
            to_panel(self.frame.drop(index=0))

    def test_rejects_invalid_sales_and_metadata(self):
        for value in [-1, np.inf, np.nan]:
            with self.subTest(value=value):
                invalid = self.frame.copy()
                invalid["y"] = invalid["y"].astype(float)
                invalid.loc[0, "y"] = value
                with self.assertRaises(ValueError):
                    to_panel(invalid)
        invalid = self.frame.copy()
        invalid.loc[0, "store_id"] = "CHANGED_STORE"
        with self.assertRaisesRegex(ValueError, "changed"):
            to_panel(invalid)

    def test_metrics_have_explicit_units_and_bias_direction(self):
        result = metrics([10, 0, 20], [8, 5, 25])
        self.assertAlmostEqual(result["mae"], 4.0)
        self.assertAlmostEqual(result["wape"], 0.4)
        self.assertAlmostEqual(result["bias"], 8 / 3)
        self.assertTrue(np.isnan(metrics([0, 0], [1, 2])["wape"]))
        with self.assertRaises(ValueError):
            metrics([1], [np.inf])

    def test_validation_and_holdout_dates_do_not_overlap(self):
        validation = cutoff_indices(self.panel, self.config, "validation")
        holdout = cutoff_indices(self.panel, self.config, "holdout")[0]
        covered = set()
        for cutoff in validation:
            window = set(self.panel.dates[cutoff + 1:cutoff + 15])
            self.assertFalse(window & covered)
            covered.update(window)
        self.assertFalse(covered & set(self.panel.dates[holdout + 1:]))
        self.assertEqual(len(self.panel.dates[holdout + 1:]), 14)

    def test_features_cannot_see_future_observed_sales(self):
        origin = 80
        modified = self.frame.copy()
        modified.loc[modified["ds"] > self.panel.dates[origin], "y"] = 999_999
        x_original = OriginFeatures(self.panel).at_origin(origin, 28)
        x_modified = OriginFeatures(to_panel(modified)).at_origin(origin, 28)
        np.testing.assert_array_equal(x_original, x_modified)

    def test_training_labels_stop_at_observed_prefix(self):
        prefix = self.panel.prefix(90)
        x, labels, origins = OriginFeatures(prefix).training_matrix(14, 60, 7)
        self.assertEqual(x.shape[0], len(labels))
        self.assertLessEqual(int(origins.max()) + 14, len(prefix.dates) - 1)
        expected = np.concatenate([
            self.panel.values[:, origin + 1:origin + 15].reshape(-1) for origin in origins
        ])
        np.testing.assert_array_equal(labels, expected)

    def test_weekly_baseline_repeats_known_week_over_long_horizon(self):
        one_series = pd.DataFrame({
            "unique_id": "A", "ds": pd.date_range("2024-01-01", periods=35),
            "y": np.tile(np.arange(1, 8), 5),
        })
        prediction = Baseline("seasonal_naive").fit(to_panel(one_series)).predict(28)[0]
        np.testing.assert_array_equal(prediction, np.tile(np.arange(1, 8), 4))

    def test_all_models_receive_only_history_at_shared_cutoff(self):
        seen = []

        class Spy:
            def fit(inner, history):
                seen.append(history.dates[-1])
                inner.count = len(history.ids)
                inner.last = history.values[:, -1].copy()
                return inner

            def predict(inner, horizon):
                return np.repeat(inner.last[:, None], horizon, axis=1)

        predictions = run_backtest(self.panel, self.config, factory=lambda name, cfg: Spy())
        self.assertEqual(seen, list(predictions["cutoff"].drop_duplicates()))
        self.assertTrue((predictions["ds"] > predictions["cutoff"]).all())
        self.assertEqual(predictions.groupby("cutoff").size().tolist(), [3 * 14, 3 * 14])

    def test_mutating_first_validation_future_cannot_change_its_forecast(self):
        cutoff = cutoff_indices(self.panel, self.config)[0]
        altered = self.frame.copy()
        altered.loc[altered["ds"] > self.panel.dates[cutoff], "y"] = 100_000
        before = run_backtest(self.panel, self.config)
        after = run_backtest(to_panel(altered), self.config)
        np.testing.assert_array_equal(
            before.loc[before["cutoff"] == self.panel.dates[cutoff], "prediction"],
            after.loc[after["cutoff"] == self.panel.dates[cutoff], "prediction"],
        )


if __name__ == "__main__":
    unittest.main()

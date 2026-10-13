"""Numerical, chronology and complete CSV-pipeline checks; no M5 training."""

from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("temporal_calibration", ROOT/"experiments/run_temporal_calibration.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    cuts = ["2016-01-31", "2016-02-28", "2016-03-27"]
    rows = []
    for cutoff in cuts:
        for series in range(8):
            for horizon in range(1, 29):
                rows.append({"unique_id": f"FOODS_{series}", "cutoff": cutoff,
                             "ds": pd.Timestamp(cutoff)+pd.Timedelta(days=horizon), "horizon": horizon,
                             "y": [0, 1, 2, 5][(series+horizon) % 4],
                             "mean_estimate": 1.5, "median_estimate": 1.0,
                             "mean28": 1.7, "seasonal_naive": 1.0,
                             "q10_raw": 0.0, "q90_raw": 2.0, "lower": 0.0, "upper": 2.0})
    contract = {"n_series": 8, "horizon": 28, "validation_cutoffs": cuts}
    return pd.DataFrame(rows), contract


def receipt():
    return {"stage": "validation", "source_sha256": "toy-source", "reference_run_sha256": "toy-ref",
            "config": {"horizon": 28, "n_series": 8, "n_windows": 3},
            "quantile_levels": {"lightgbm_q10": .1, "lightgbm_q90": .9},
            "nominal_target_coverage": .8, "interval_policy": "toy envelope", "calibration": "none",
            "inclusive_bounds": True, "n_series": 8, "observed_days": 730,
            "control_reproduced_within_tolerance": True, "holdout_evaluated": False}


class TemporalCalibrationTests(unittest.TestCase):
    def setUp(self):
        raw, self.contract = fixture()
        self.frame = MODULE.validate_frame(raw, self.contract)

    def test_interval_score_tails_and_inclusive_bounds(self):
        frame = pd.DataFrame({"y": [0, 1, 5], "lower": [1, 1, 1], "upper": [3, 3, 3],
                              "median_estimate": [2, 2, 2]})
        detail = MODULE.add_diagnostics(frame)
        np.testing.assert_array_equal(detail.interval_score_80, [12, 2, 22])
        summary = MODULE.summarize_diagnostics(detail).iloc[0]
        self.assertAlmostEqual(summary.coverage, 1/3)
        self.assertAlmostEqual(summary.lower_miss_rate, 1/3)
        self.assertAlmostEqual(summary.upper_miss_rate, 1/3)
        self.assertAlmostEqual(summary.interval_score_80, 12)

    def test_order_statistic_and_insufficient_calibration(self):
        self.assertEqual(MODULE.finite_sample_quantile([-1, 2, 0, 1]), (2, 4))
        self.assertEqual(MODULE.finite_sample_quantile([-3, -2, -1, -4]), (-1, 4))
        with self.assertRaises(ValueError):
            MODULE.finite_sample_quantile([1, 2])
        with self.assertRaises(ValueError):
            MODULE.finite_sample_quantile([1, np.nan, 2, 3])

    def test_future_labels_do_not_change_any_correction(self):
        _, before = MODULE.temporal_calibrate(self.frame)
        poisoned = self.frame.copy()
        poisoned.loc[poisoned.cutoff == poisoned.cutoff.max(), "y"] = 1000000
        _, after = MODULE.temporal_calibrate(poisoned)
        pd.testing.assert_frame_equal(before, after)

    def test_current_labels_do_not_change_current_correction(self):
        cutoff = pd.Timestamp("2016-02-28")
        _, before = MODULE.temporal_calibrate(self.frame)
        poisoned = self.frame.copy()
        poisoned.loc[poisoned.cutoff == cutoff, "y"] = 1000000
        _, after = MODULE.temporal_calibrate(poisoned)
        pd.testing.assert_frame_equal(before[before.evaluation_cutoff == cutoff],
                                      after[after.evaluation_cutoff == cutoff])
        self.assertTrue((after[after.evaluation_cutoff > cutoff].applied_expansion > 1000).all())

    def test_incomplete_earlier_window_not_available(self):
        frame = self.frame.copy()
        selected = frame.cutoff == pd.Timestamp("2016-02-28")
        frame.loc[selected, "cutoff"] = pd.Timestamp("2016-02-15")
        frame.loc[selected, "ds"] = frame.loc[selected, "cutoff"]+pd.to_timedelta(frame.loc[selected, "horizon"], unit="D")
        _, audit = MODULE.temporal_calibrate(frame)
        self.assertEqual(set(audit.evaluation_cutoff), {pd.Timestamp("2016-03-27")})
        self.assertTrue(audit.past_only_check.all())

    def test_bounds_expand_and_preserve_central_predictions(self):
        evaluated, audit = MODULE.temporal_calibrate(self.frame)
        self.assertEqual(set(audit.applied_expansion), {3.0})
        base = evaluated[evaluated.method == "uncalibrated"].sort_values(MODULE.KEY).reset_index(drop=True)
        adjusted = evaluated[evaluated.method == "temporal_cqr"].sort_values(MODULE.KEY).reset_index(drop=True)
        self.assertTrue((adjusted.lower <= base.lower).all())
        self.assertTrue((adjusted.upper >= base.upper).all())
        pd.testing.assert_frame_equal(adjusted[list(MODULE.FORECASTS.values())], base[list(MODULE.FORECASTS.values())])
        self.assertFalse(MODULE.add_diagnostics(adjusted).median_outside.any())
        self.assertEqual(len(base), 8*28*2)

    def test_conservative_intervals_have_zero_expansion(self):
        frame = self.frame.copy()
        frame["upper"] = frame["q90_raw"] = 10.0
        _, audit = MODULE.temporal_calibrate(frame)
        self.assertTrue(audit.applied_expansion.eq(0).all())

    def test_missing_duplicate_and_misaligned_rows_rejected(self):
        raw, contract = fixture()
        invalid = [pd.concat([raw, raw.iloc[:1]], ignore_index=True), raw.iloc[1:].copy()]
        wrong_date = raw.copy()
        wrong_date.loc[0, "ds"] += pd.Timedelta(days=1)
        invalid.append(wrong_date)
        wrong_bounds = raw.copy()
        wrong_bounds.loc[0, "upper"] = 0.5
        invalid.append(wrong_bounds)
        for frame in invalid:
            with self.subTest(size=len(frame)):
                with self.assertRaises(ValueError):
                    MODULE.validate_frame(frame, contract)

    def test_manifest_changes_rejected(self):
        original = receipt()
        MODULE.check_manifest(original, original)
        for field, value in [("source_sha256", "changed"), ("holdout_evaluated", True),
                             ("config", {"horizon": 7}), ("control_reproduced_within_tolerance", False)]:
            changed = {**original, field: value}
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    MODULE.check_manifest(changed, original)

    def test_complete_csv_pipeline_and_frozen_summary_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as directory:
            root = Path(directory)
            inp, ref, out = root/"input", root/"reference", root/"output"
            inp.mkdir()
            ref.mkdir()
            for location in [inp, ref]:
                (location/"run.json").write_text(json.dumps(receipt()), encoding="utf-8")
            (ref/"split_contract.json").write_text(json.dumps(self.contract), encoding="utf-8")
            self.frame.to_csv(inp/"interval_predictions.csv", index=False)
            for name, groups in [("interval_metrics.csv", []), ("interval_by_band.csv", ["horizon_band"]),
                                 ("interval_by_sales_state.csv", ["sales_state"])]:
                MODULE.summarize_frozen(self.frame, groups).to_csv(ref/name, index=False)
            MODULE.summarize_points(self.frame).to_csv(ref/"point_overall_metrics.csv", index=False)
            with redirect_stdout(io.StringIO()):
                result = MODULE.run(inp, out, ref)
            self.assertFalse(result["model_refitted"])
            self.assertFalse(result["holdout_evaluated"])
            self.assertTrue(result["past_only_check"])
            summary = pd.read_csv(out/"evaluation_metrics.csv")
            self.assertEqual(set(summary.n_observations), {448})
            self.assertEqual(summary.loc[summary.method == "temporal_cqr", "coverage"].iloc[0], 1.0)
            self.assertTrue(pd.read_csv(out/"summary_reproduction.csv").passed.all())
            frozen = pd.read_csv(ref/"interval_metrics.csv")
            frozen.loc[0, "coverage"] = 0.0
            frozen.to_csv(ref/"interval_metrics.csv", index=False)
            with self.assertRaises(ValueError):
                MODULE.run(inp, root/"rejected", ref)


if __name__ == "__main__":
    unittest.main()

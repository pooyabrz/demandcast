"""New-cohort cutoff, identity, freeze and scoring checks with explicit fixtures.

No original core checkout, real M5 sales or substitute core installation is
used here. Existing addon tests cover the real L1/quantile estimator classes.
"""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
sys.path.insert(0, str(ROOT/"scripts"))
import run_final_evaluation as final


def identities():
    return pd.DataFrame([
        {"id": f"FOODS_1_00{i}_{store}_evaluation", "item_id": f"FOODS_1_00{i}",
         "dept_id": "FOODS_1", "cat_id": "FOODS", "store_id": store, "state_id": store[:2]}
        for i, store in [(1, "CA_1"), (2, "TX_1")]])


def small_registration():
    return {"cohort": {"selected_raw_series_ids": identities().id.tolist()}, "horizon": 2,
            "forecast_cutoff": "2011-01-30", "target_start": "2011-01-31", "target_end": "2011-02-01"}


def scored_fixture():
    registration = small_registration()
    forecasts, targets = [], []
    values = {"seasonal_naive": 4, "mean28": 3, "lightgbm": 5, "lightgbm_l1": 2,
              "lightgbm_q10": 1, "lightgbm_q90": 4}
    for i, row in identities().iterrows():
        for horizon, date in enumerate(pd.date_range("2011-01-31", periods=2), 1):
            targets.append({**row.drop("id").to_dict(), "unique_id": final.normal_id(row.id), "ds": date, "y": 10*i})
            for model, prediction in values.items():
                forecasts.append({"unique_id": final.normal_id(row.id), "ds": date, "cutoff": pd.Timestamp("2011-01-30"),
                                  "horizon": horizon, "model": model, "prediction": prediction})
    return pd.DataFrame(forecasts), pd.DataFrame(targets), registration


class FinalEvaluationChecks(unittest.TestCase):
    def test_chunked_history_reader_does_not_deserialize_future_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            path = Path(name)/"sales.csv"
            raw = identities().assign(d_1=[1, 2], d_2=[3, 4], d_3="UNPARSED_FUTURE")
            raw.to_csv(path, index=False)
            history = final.read_sales(path, raw.id.tolist(), {"d_1": pd.Timestamp("2011-01-29"), "d_2": pd.Timestamp("2011-01-30")}, chunksize=1)
            self.assertEqual(len(history), 4)
            self.assertEqual(history.y.sum(), 10)
            raw["d_3"] = "A_DIFFERENT_FUTURE"
            raw.to_csv(path, index=False)
            pd.testing.assert_frame_equal(history, final.read_sales(path, raw.id.tolist(), {"d_1": pd.Timestamp("2011-01-29"), "d_2": pd.Timestamp("2011-01-30")}, chunksize=1))

    def test_calendar_enforces_m5_numbering_and_702_plus_28_split(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            path = Path(name)/"calendar.csv"
            dates = pd.date_range(end="2016-05-22", periods=730)
            numbers = (dates-pd.Timestamp("2011-01-29")).days+1
            calendar = pd.DataFrame({"date": dates, "d": [f"d_{n}" for n in numbers]})
            calendar.to_csv(path, index=False)
            protocol = {"target_end": "2016-05-22", "target_start": "2016-04-25", "forecast_cutoff": "2016-04-24", "history_days_in_prepared_panel": 730}
            split = final.calendar_split(path, protocol)
            self.assertEqual((len(split["history"]), len(split["targets"])), (702, 28))
            self.assertEqual(list(split["history"])[-1], "d_1913")
            self.assertEqual(list(split["targets"])[-1], "d_1941")
            calendar["date"] += pd.Timedelta(days=1)
            calendar.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "day numbering"):
                final.calendar_split(path, protocol)

    def test_forecasting_passes_only_complete_observed_history_to_models(self):
        history = pd.DataFrame([{ "unique_id": identifier, "ds": day, "y": value}
            for identifier, value in [("A", 2), ("B", 7)] for day in pd.date_range("2011-01-29", periods=2)])
        seen = []

        def panel_builder(frame):
            pivot = frame.pivot(index="unique_id", columns="ds", values="y")
            return SimpleNamespace(ids=pivot.index.to_numpy(), dates=pivot.columns, values=pivot.to_numpy())

        class ObservedOnlyModel:
            def fit(self, panel):
                seen.append(pd.DatetimeIndex(panel.dates))
                self.prediction = panel.values[:, -1]
            def predict(self, horizon):
                return np.repeat(self.prediction[:, None], horizon, axis=1)

        config = SimpleNamespace(horizon=2, models=final.MODELS)
        with redirect_stdout(io.StringIO()):
            forecasts = final.forecast_models(history, config, "2011-01-30", lambda *_: ObservedOnlyModel(), panel_builder)
        self.assertEqual(len(forecasts), 24)
        self.assertNotIn("y", forecasts)
        self.assertTrue(all(index.max() == pd.Timestamp("2011-01-30") for index in seen))
        self.assertEqual(forecasts.ds.min(), pd.Timestamp("2011-01-31"))
        future = history.iloc[:1].assign(ds=pd.Timestamp("2011-01-31"), y=999999)
        with self.assertRaises(ValueError):
            final.forecast_models(pd.concat([history, future]), config, "2011-01-30", lambda *_: ObservedOnlyModel(), panel_builder)

    def test_point_and_interval_scores_match_hand_calculations(self):
        forecasts, targets, registration = scored_fixture()
        frame = final.score_forecasts(forecasts, targets, registration)
        points = final.point_metrics(frame).set_index("model")
        median = points.loc["lightgbm_l1"]
        self.assertEqual(median.mae, 5)
        self.assertEqual(median.wape, 1)
        self.assertEqual(median.bias, -3)
        self.assertEqual(median.aggregate_volume_bias_fraction, -.6)
        self.assertEqual(points.loc["lightgbm", "aggregate_volume_bias_fraction"], 0)
        interval = final.interval_metrics(frame).iloc[0]
        self.assertEqual(interval.n_observations, 4)
        self.assertEqual(interval.coverage, 0)
        self.assertEqual(interval.mean_width, 3)
        self.assertEqual(interval.lower_miss_rate, .5)
        self.assertEqual(interval.upper_miss_rate, .5)
        self.assertEqual(interval.interval_score_80, 38)
        self.assertAlmostEqual(interval.pinball_q10, .9)
        self.assertAlmostEqual(interval.pinball_q90, 2.9)

    def test_crossed_tails_keep_registered_envelope_and_median(self):
        forecasts, targets, registration = scored_fixture()
        forecasts.loc[forecasts.model == "lightgbm_q10", "prediction"] = 4
        forecasts.loc[forecasts.model == "lightgbm_q90", "prediction"] = 1
        frame = final.score_forecasts(forecasts, targets, registration)
        self.assertTrue(frame.raw_quantile_crossing.all())
        self.assertTrue(frame.bounds_adjusted.all())
        self.assertTrue(frame.median_estimate.eq(2).all())
        self.assertTrue(frame.lower.eq(1).all())
        self.assertTrue(frame.upper.eq(4).all())

    def test_zero_sales_have_undefined_wape_and_volume_bias(self):
        forecasts, targets, registration = scored_fixture()
        targets["y"] = 0
        points = final.point_metrics(final.score_forecasts(forecasts, targets, registration))
        self.assertTrue(points.wape.isna().all())
        self.assertTrue(points.aggregate_volume_bias_fraction.isna().all())

    def test_incomplete_or_altered_forecast_and_target_grids_rejected(self):
        forecasts, targets, registration = scored_fixture()
        for bad in [forecasts.iloc[1:], pd.concat([forecasts, forecasts.iloc[:1]]),
                    forecasts.assign(y=0), forecasts.assign(horizon=1), forecasts.assign(prediction=np.inf)]:
            with self.assertRaises(ValueError):
                final.score_forecasts(bad, targets, registration)
        for bad in [targets.iloc[1:], pd.concat([targets, targets.iloc[:1]]), targets.assign(y=-1)]:
            with self.assertRaises(ValueError):
                final.score_forecasts(forecasts, bad, registration)

    def test_registration_rejects_recipe_environment_and_source_changes(self):
        recipe = {"seed": 42, "horizon": 28, "n_windows": 3, "n_series": 500, "history_days": 730,
                  "training_origin_days": 365, "origin_stride": 7, "iterations": 200, "threads": 4, "models": final.MODELS}
        registration = {"status": "registered_not_evaluated", "forecast_cutoff": "2016-04-24", "target_start": "2016-04-25",
                        "target_end": "2016-05-22", "horizon": 28, "history_days_in_prepared_panel": 730, "recipe_config": recipe,
                        "versions": {"numpy": "fixture-version"}, "development_panel_sha256": "fixture-data-hash",
                        "source_hashes": {"src/demandcast/models.py": "core-hash", "experiments/run_l1_ablation.py": "l1-hash",
                                          "experiments/run_interval_benchmark.py": "q-hash"},
                        "cohort": {"n_items": 50, "n_series": 500, "stores": list(range(10)), "selection_seed": 20261013,
                                   "item_overlap_count": 0, "series_overlap_count": 0, "selected_item_ids": list(range(50)),
                                   "selected_raw_series_ids": [f"fixture-{i}" for i in range(500)]}}
        report = {"ready_for_local_review": True, "versions": registration["versions"].copy(),
                  "prepared_data": {"sha256": "fixture-data-hash"}, "source_hashes": {**registration["source_hashes"], final.RUNNER: "new-runner"}}
        base = {**recipe, "models": final.MODELS[:3]}
        final.check_locked_state(registration, report, base)
        altered = copy.deepcopy(report)
        altered["source_hashes"]["src/demandcast/models.py"] = "changed"
        newer = copy.deepcopy(report)
        newer["source_hashes"]["experiments/unregistered_tuning.py"] = "new"
        version = copy.deepcopy(report)
        version["versions"]["numpy"] = "changed"
        for bad_report, bad_config in [(altered, base), (newer, base), (version, base), (report, {**base, "seed": 7})]:
            with self.assertRaises(ValueError):
                final.check_locked_state(registration, bad_report, bad_config)

    def test_frozen_predictions_or_binding_changes_cannot_be_scored(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            output = Path(name)
            path = output/"forecasts.csv"
            path.write_text("original forecasts")
            binding = {"registration_sha256": "fixture-registration"}
            receipt = {"status": "forecasts_frozen", "bindings": binding, "forecasts_sha256": final.hash_file(path)}
            final.verify_seal(output, receipt, binding)
            with self.assertRaises(ValueError):
                final.verify_seal(output, receipt, {"registration_sha256": "altered"})
            path.write_text("altered predictions")
            with self.assertRaisesRegex(ValueError, "Frozen forecasts changed"):
                final.verify_seal(output, receipt, binding)

    def test_registration_attempt_cannot_be_restarted_in_another_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            root = Path(name)
            registration = root/"reports/registration.json"
            registration.parent.mkdir()
            registration.write_text("fixture registration")
            final.claim_attempt(root, registration, root/"outputs/first", {"registration_sha256": "fixture-hash"})
            with self.assertRaises(FileExistsError):
                final.claim_attempt(root, registration, root/"outputs/different", {"registration_sha256": "fixture-hash"})

    def test_score_phase_writes_audited_results_and_does_not_reload_scored_targets(self):
        # Explicitly inject the already-verified context. This is an orchestration
        # fixture, not a successful preflight for a substitute DemandCast package.
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as name:
            root = Path(name)
            forecasts, targets, registration = scored_fixture()
            registration.update({"recipe_config": {"models": final.MODELS, "horizon": 2},
                                 "limitations": ["synthetic orchestration fixture"],
                                 "fit_policy": "fixture observed-only fit", "comparison_policy": "same fixture targets"})
            path = root/"reports/registration.json"
            path.parent.mkdir()
            path.write_text(json.dumps(registration))
            output = root/"outputs/test-run"
            bindings = {"registration_sha256": final.hash_file(path), "context": "explicit test fixture"}
            marker, attempt = final.claim_attempt(root, path, output, bindings)
            forecasts.to_csv(output/"forecasts.csv", index=False)
            (output/"registration_snapshot.json").write_bytes(path.read_bytes())
            receipt = {"status": "forecasts_frozen", "bindings": bindings,
                       "forecasts_sha256": final.hash_file(output/"forecasts.csv"), "observed_training_days": 2}
            final.write_json(output/"forecast_run.json", receipt)
            attempt.update({"status": "forecasts_frozen", "forecasts_sha256": receipt["forecasts_sha256"],
                            "forecast_run_sha256": final.hash_file(output/"forecast_run.json")})
            final.write_json(marker, attempt)
            sales = root/"sales.csv"
            identities().assign(d_3=[0, 10], d_4=[0, 10]).to_csv(sales, index=False)
            split = {"targets": {"d_3": pd.Timestamp("2011-01-31"), "d_4": pd.Timestamp("2011-02-01")}}
            with patch.object(final, "verified_context", return_value=(registration, split, bindings)), \
                    patch.object(final, "read_sales", wraps=final.read_sales) as reader, redirect_stdout(io.StringIO()):
                final.score_phase(root, path, sales, root/"calendar-not-needed-in-this-fixture.csv", output)
                original = (output/"run.json").read_bytes()
                final.score_phase(root, path, sales, root/"calendar-not-needed-in-this-fixture.csv", output)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual((output/"run.json").read_bytes(), original)
            run = json.loads(original)
            self.assertEqual(run["status"], "scored")
            self.assertTrue(run["target_sales_deserialized"])
            self.assertEqual(run["n_observations_per_model"], 4)
            self.assertEqual(run["n_series"], 2)
            self.assertEqual(run["n_items"], 2)
            for artifact, digest in run["artifact_hashes"].items():
                self.assertEqual(final.hash_file(output/artifact), digest)


if __name__ == "__main__":
    unittest.main()

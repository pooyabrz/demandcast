"""Validate quantile prediction intervals while preserving frozen point controls."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from run_l1_ablation import L1DirectTree, hash_file
except ModuleNotFoundError as exc:
    raise SystemExit("Copy this script into the existing project's experiments folder, "
                     "beside run_l1_ablation.py from the previous addon.") from exc

KEY = ["unique_id", "cutoff", "ds", "horizon"]
QUANTILES = {"lightgbm_q10": 0.1, "lightgbm_q90": 0.9}
POINT_MODELS = ["seasonal_naive", "mean28", "lightgbm", "lightgbm_l1"]


class QuantileDirectTree(L1DirectTree):
    def __init__(self, config, feature_class, level):
        super().__init__(config, feature_class)
        if not 0 < level < 1:
            raise ValueError("Quantile level must be between zero and one")
        self.level = float(level)

    def fit(self, panel):
        from lightgbm import LGBMRegressor
        self.features = self.feature_class(panel)
        x, y, self.training_origins = self.features.training_matrix(
            self.config.horizon, self.config.training_origin_days, self.config.origin_stride)
        self.constant = float(y[0]) if np.all(y == y[0]) else None
        if self.constant is not None:
            return self
        self.estimator = LGBMRegressor(
            objective="quantile", alpha=self.level, learning_rate=0.05,
            n_estimators=self.config.iterations, num_leaves=31,
            n_jobs=self.config.threads, random_state=self.config.seed,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
        self.estimator.fit(pd.DataFrame(x, columns=self.features.names), y,
                           categorical_feature=self.features.categorical_indices,
                           feature_name=self.features.names)
        return self


def interval_frame(predictions):
    needed = set(POINT_MODELS + list(QUANTILES))
    if predictions.empty or set(predictions.stage) != {"validation"}:
        raise ValueError("Expected nonempty validation predictions")
    if set(predictions.model) != needed:
        raise ValueError("Missing or unexpected point/quantile models")
    if predictions.duplicated(KEY + ["model"]).any():
        raise ValueError("Duplicate prediction keys")
    if not np.isfinite(predictions[["y", "prediction"]]).all().all():
        raise ValueError("Nonfinite actuals or predictions")
    if (predictions[["y", "prediction"]] < 0).any().any():
        raise ValueError("Expected nonnegative unit sales and predictions")
    if not predictions.groupby(KEY).y.nunique().eq(1).all():
        raise ValueError("Different actual values across models")
    wide = predictions.pivot(index=KEY, columns="model", values="prediction").reset_index()
    actual = predictions.drop_duplicates(KEY)[KEY + ["y"]]
    wide = wide.merge(actual, on=KEY, validate="one_to_one")
    if wide[list(needed)].isna().any().any():
        raise ValueError("Models do not cover the same observations")
    wide = wide.rename(columns={"lightgbm": "mean_estimate", "lightgbm_l1": "median_estimate",
                                "lightgbm_q10": "q10_raw", "lightgbm_q90": "q90_raw"})
    wide["raw_quantile_crossing"] = wide.q10_raw > wide.q90_raw
    wide["median_outside_raw_bounds"] = ((wide.median_estimate < wide.q10_raw) |
                                          (wide.median_estimate > wide.q90_raw))
    # Declared coherence policy: widen to the envelope of the two tails and
    # the unchanged L1 median. This is not conformal calibration or a guarantee.
    wide["lower"] = wide[["q10_raw", "median_estimate", "q90_raw"]].min(axis=1)
    wide["upper"] = wide[["q10_raw", "median_estimate", "q90_raw"]].max(axis=1)
    wide["bounds_adjusted"] = ((wide.lower != wide.q10_raw) | (wide.upper != wide.q90_raw))
    wide["covered_raw"] = (wide.y >= wide.q10_raw) & (wide.y <= wide.q90_raw)
    wide["covered"] = (wide.y >= wide.lower) & (wide.y <= wide.upper)
    wide["width"] = wide.upper - wide.lower
    for level, name in [(0.1, "q10"), (0.9, "q90")]:
        error = wide.y - wide[f"{name}_raw"]
        wide[f"pinball_{name}"] = np.maximum(level * error, (level-1) * error)
    wide["sales_state"] = np.where(wide.y == 0, "zero", "positive")
    wide["category"] = wide.unique_id.str.split("_").str[0]
    wide["horizon_band"] = pd.cut(wide.horizon, [0, 7, 14, 28],
                                   labels=["1-7", "8-14", "15-28"])
    if wide.horizon_band.isna().any():
        raise ValueError("This frozen experiment expects a horizon no longer than 28")
    return wide


def summarize_intervals(frame, groups=None):
    groups = groups or []
    partitions = frame.groupby(groups, observed=True, sort=True) if groups else [((), frame)]
    records = []
    for key, part in partitions:
        key = key if isinstance(key, tuple) else (key,)
        records.append({**dict(zip(groups, key)), "n_observations": len(part),
                        "nominal_target_coverage": 0.8,
                        "coverage_raw": float(part.covered_raw.mean()),
                        "coverage": float(part.covered.mean()),
                        "mean_width": float(part.width.mean()),
                        "mean_lower": float(part.lower.mean()), "mean_upper": float(part.upper.mean()),
                        "pinball_q10": float(part.pinball_q10.mean()),
                        "pinball_q90": float(part.pinball_q90.mean()),
                        "raw_crossing_rate": float(part.raw_quantile_crossing.mean()),
                        "median_outside_raw_bounds_rate": float(part.median_outside_raw_bounds.mean()),
                        "bounds_adjustment_rate": float(part.bounds_adjusted.mean())})
    return pd.DataFrame(records)


def check_frozen_reference(reference, base, panel, source_hash):
    if reference.get("stage") != "validation" or reference.get("holdout_evaluated") is not False:
        raise ValueError("Reference must be the frozen L1 validation experiment")
    if reference.get("control_reproduced_within_tolerance") is not True:
        raise ValueError("The L1 reference has no reproduced control")
    if reference.get("source_sha256") != source_hash:
        raise ValueError("Prepared input differs from the frozen reference")
    for name, expected in reference["config"].items():
        if name == "models":
            if list(base.models) + ["lightgbm_l1"] != expected:
                raise ValueError("Original model configuration changed")
        elif getattr(base, name) != expected:
            raise ValueError(f"Configuration changed: {name}")
    if len(panel.ids) != reference["n_series"] or len(panel.dates) != reference["observed_days"]:
        raise ValueError("Panel differs from the reference")


def compare_controls(metrics, expected):
    actual = metrics[metrics.model.isin(POINT_MODELS)].copy()
    expected = expected.copy()
    actual["cutoff"] = pd.to_datetime(actual.cutoff)
    expected["cutoff"] = pd.to_datetime(expected.cutoff)
    compared = expected.merge(actual, on=["model", "cutoff"], how="outer", indicator=True,
                              suffixes=("_expected", "_actual"), validate="one_to_one")
    passed = compared._merge.eq("both").all() and all(
        np.allclose(compared[f"{metric}_expected"], compared[f"{metric}_actual"], rtol=1e-6, atol=1e-7)
        for metric in ["mae", "wape", "bias", "n_observations"])
    return bool(passed), compared


def main(argv=None):
    parser = argparse.ArgumentParser(description="DemandCast: validate 80% target prediction intervals")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/m5.yaml")
    parser.add_argument("--output", default="outputs/interval-validation")
    parser.add_argument("--reference-dir", default="reports/l1_reference")
    args = parser.parse_args(argv)
    try:
        from demandcast.config import load_config
        from demandcast.data import to_panel
        from demandcast.evaluation import run_backtest, summarize
        from demandcast.features import OriginFeatures
        from demandcast.models import make_model
    except ImportError as exc:
        parser.exit(2, f"Activate the original DemandCast environment. Import error: {exc}\n")
    base = load_config(args.config)
    reference_dir = Path(args.reference_dir)
    reference = json.loads((reference_dir / "run.json").read_text(encoding="utf-8"))
    source_hash = hash_file(args.data)
    panel = to_panel(pd.read_csv(args.data, dtype={"unique_id": str}))
    check_frozen_reference(reference, base, panel, source_hash)
    payload = base.to_dict()
    payload["models"] = POINT_MODELS + list(QUANTILES)
    config = SimpleNamespace(**payload)

    def factory(name, settings):
        if name in QUANTILES:
            return QuantileDirectTree(settings, OriginFeatures, QUANTILES[name])
        if name == "lightgbm_l1":
            return L1DirectTree(settings, OriginFeatures)
        return make_model(name, settings)

    started = time.perf_counter()
    predictions = run_backtest(panel, config, stage="validation", factory=factory)
    point_predictions = predictions[predictions.model.isin(POINT_MODELS)]
    point_metrics = summarize(point_predictions, ["model", "cutoff"])
    expected = pd.read_csv(reference_dir / "metrics.csv", parse_dates=["cutoff"])
    reproduced, compared = compare_controls(point_metrics, expected)
    intervals = interval_frame(predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    point_metrics.to_csv(output / "point_metrics.csv", index=False)
    summarize(point_predictions, ["model"]).to_csv(output / "point_overall_metrics.csv", index=False)
    intervals.to_csv(output / "interval_predictions.csv", index=False)
    summaries = {"interval_metrics.csv": [], "interval_by_fold.csv": ["cutoff"],
                 "interval_by_horizon.csv": ["horizon"], "interval_by_band.csv": ["horizon_band"],
                 "interval_by_sales_state.csv": ["sales_state"], "interval_by_category.csv": ["category"]}
    for filename, groups in summaries.items():
        summarize_intervals(intervals, groups).to_csv(output / filename, index=False)
    compared.to_csv(output / "control_reproduction.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "validation",
        "experiment": "E02: quantile tails with frozen Poisson and L1 central forecasts",
        "source_sha256": source_hash, "reference_run_sha256": hash_file(reference_dir / "run.json"),
        "config": payload, "quantile_levels": QUANTILES, "nominal_target_coverage": 0.8,
        "interval_policy": "lower/upper envelope of q10, unchanged L1 median and q90; report raw and adjusted coverage",
        "calibration": "none; nominal 80% is a target, not verified coverage or a distribution-free guarantee",
        "inclusive_bounds": True, "n_series": len(panel.ids), "observed_days": len(panel.dates),
        "control_reproduced_within_tolerance": reproduced, "holdout_evaluated": False,
        "lightgbm_version": importlib.metadata.version("lightgbm"),
        "elapsed_seconds": round(time.perf_counter()-started, 3),
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summarize_intervals(intervals).to_string(index=False))
    print(f"Results: {output}")
    if not reproduced:
        parser.exit(3, "Point controls did not reproduce. Investigate before interpreting intervals.\n")
    print("Both central models preserved; holdout not evaluated; coverage remains empirical.")


if __name__ == "__main__":
    main()

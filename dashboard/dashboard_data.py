"""Validated, source-backed data helpers for the local DemandCast dashboard."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


MEASURES = ["coverage", "lower_miss_rate", "upper_miss_rate", "mean_width",
            "interval_score_80", "mean_expansion", "median_outside_rate"]
KEY = ["unique_id", "cutoff", "ds", "horizon"]
BIN_ORDER = ["0-0.5", "0.5-1", "1-2", "2+"]
MODEL_LABELS = {"lightgbm_l1": "LightGBM L1 (median)", "lightgbm": "LightGBM Poisson (mean)",
                "mean28": "28-day mean", "seasonal_naive": "Seasonal naive"}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def weighted_metrics(rows):
    if rows.empty:
        raise ValueError("Select at least one observation group")
    weights = pd.to_numeric(rows.n_observations).to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any() or not np.equal(weights, np.floor(weights)).all():
        raise ValueError("Observation counts must be positive integers")
    values = rows[MEASURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite summary metrics")
    return {"n_observations": int(weights.sum()),
            **dict(zip(MEASURES, np.average(values, axis=0, weights=weights)))}


def check_interval_rows(rows):
    required = set(["method", "n_observations", "nominal_target_coverage", *MEASURES])
    if rows.empty or required - set(rows.columns):
        raise ValueError("Incomplete interval summary schema")
    numeric = rows[["n_observations", "nominal_target_coverage", *MEASURES]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite interval summary values")
    if not rows.nominal_target_coverage.eq(.8).all():
        raise ValueError("This dashboard expects the E03 nominal 80% protocol")
    rates = rows[["coverage", "lower_miss_rate", "upper_miss_rate", "median_outside_rate"]]
    if ((rates < 0) | (rates > 1)).any().any():
        raise ValueError("Invalid probability values")
    if not np.allclose(rows.coverage+rows.lower_miss_rate+rows.upper_miss_rate, 1, atol=1e-10):
        raise ValueError("Coverage and tail misses do not reconcile")
    if (rows.mean_width < 0).any() or (rows.interval_score_80 < rows.mean_width-1e-10).any():
        raise ValueError("Invalid width or interval score")
    if (rows.mean_expansion < 0).any():
        raise ValueError("Negative expansion")
    for _, part in rows.groupby("method"):
        weighted_metrics(part)


def load_bundle(project_root, source="bundled"):
    root = Path(project_root)
    if source == "bundled":
        interval_dir = root/"reports/dashboard_reference/intervals"
        calibration_dir = root/"reports/dashboard_reference/calibration"
    elif source == "project":
        interval_dir = root/"outputs/interval-validation"
        calibration_dir = root/"outputs/calibration-validation"
    else:
        raise ValueError("Unknown result source")
    e02_bytes = (interval_dir/"run.json").read_bytes()
    e02 = json.loads(e02_bytes)
    e03 = json.loads((calibration_dir/"run.json").read_text(encoding="utf-8"))
    if e03["e02_run_sha256"] != sha256_bytes(e02_bytes) or e02["source_sha256"] != e03["source_sha256"]:
        raise ValueError("E02 and E03 receipts do not refer to the same run/data")
    if (e03["stage"] != "validation_replay" or e03["holdout_evaluated"] is not False
            or e03["summary_reproduced_within_tolerance"] is not True or e03["past_only_check"] is not True):
        raise ValueError("Expected the verified E03 validation replay")
    files = {"overall": calibration_dir/"evaluation_metrics.csv",
             "folds": calibration_dir/"evaluation_by_fold.csv",
             "segments": calibration_dir/"evaluation_by_predicted_mean.csv",
             "calibration": calibration_dir/"calibration_table.csv",
             "points": interval_dir/"point_overall_metrics.csv"}
    frames = {name: pd.read_csv(path) for name, path in files.items()}
    for name in ["overall", "folds", "segments"]:
        check_interval_rows(frames[name])
        if set(frames[name].method) != {"uncalibrated", "temporal_cqr"}:
            raise ValueError("Unexpected interval methods")
    if frames["overall"].duplicated("method").any():
        raise ValueError("Duplicate pooled interval summaries")
    for name, dimension in [("folds", "cutoff"), ("segments", "predicted_mean_bin")]:
        if frames[name].duplicated(["method", dimension]).any():
            raise ValueError(f"Duplicate {dimension} summaries")
        for method, expected in frames["overall"].set_index("method").iterrows():
            observed = weighted_metrics(frames[name][frames[name].method == method])
            for metric in ["n_observations", *MEASURES]:
                if not np.isclose(observed[metric], expected[metric], rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Weighted summaries differ: {name}/{metric}")
    if set(frames["folds"].cutoff.astype(str)) != set(e03["evaluation_cutoffs"]):
        raise ValueError("Evaluation dates differ from the receipt")
    if not frames["overall"].n_observations.eq(e03["n_evaluation_observations_per_method"]).all():
        raise ValueError("Evaluation population differs from receipt")
    audit = frames["calibration"]
    if audit.duplicated(["evaluation_cutoff", "horizon_band"]).any() or len(audit) != 3*len(e03["evaluation_cutoffs"]):
        raise ValueError("Incomplete or duplicate calibration audit")
    if not audit.past_only_check.eq(True).all():
        raise ValueError("Calibration receipt contains an unavailable target")
    if not (pd.to_datetime(audit.calibration_target_end) <= pd.to_datetime(audit.evaluation_cutoff)).all():
        raise ValueError("Calibration targets extend past the forecast cutoff")
    if not np.allclose(audit.applied_expansion, np.maximum(0, audit.signed_score_quantile)):
        raise ValueError("Applied calibration correction differs from declared policy")
    if not np.equal(audit.order_statistic_rank, np.ceil((audit.n_calibration+1)*.8)).all():
        raise ValueError("Calibration order-statistic ranks differ from the declared 80% protocol")
    for row in audit.itertuples(index=False):
        if not all(str(cutoff) < str(row.evaluation_cutoff) for cutoff in json.loads(row.calibration_cutoffs)):
            raise ValueError("Calibration contains a current or future forecast window")
    points = frames["points"]
    if set(points.model) != set(MODEL_LABELS) or points.duplicated("model").any():
        raise ValueError("Point model population changed")
    if not np.isfinite(points[["mae", "wape", "bias", "n_observations"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite point metrics")
    expected_count = e02["n_series"]*e02["config"]["horizon"]*e02["config"]["n_windows"]
    if not points.n_observations.eq(expected_count).all() or (points[["mae", "wape"]] < 0).any().any():
        raise ValueError("Invalid point model metrics or population")
    totals = points.mae*points.n_observations/points.wape
    if not np.isfinite(totals).all() or not np.allclose(totals, totals.iloc[0], rtol=1e-10):
        raise ValueError("Point metrics do not imply a common actual-sales total")
    frames["points"] = points.assign(model_label=points.model.map(MODEL_LABELS),
                                    volume_bias_pct=100*points.bias*points.n_observations/totals)
    provenance = [{"file": str(path.relative_to(root)), "sha256": sha256_bytes(path.read_bytes()),
                   "rows": len(frames[name])} for name, path in files.items()]
    return frames, {"e02": e02, "e03": e03}, provenance


def forecast_contract(receipts, audit):
    cuts = set(receipts["e03"]["evaluation_cutoffs"])
    for value in audit.calibration_cutoffs:
        cuts.update(json.loads(value))
    return {"cutoffs": sorted(cuts), "n_series": receipts["e02"]["n_series"],
            "horizon": receipts["e02"]["config"]["horizon"],
            "expected_sha256": receipts["e03"]["interval_predictions_sha256"]}


def load_forecasts(data, contract):
    if sha256_bytes(data) != contract["expected_sha256"]:
        raise ValueError("Forecast file hash differs from the file used in E03")
    required = [*KEY, "y", "mean_estimate", "median_estimate", "q10_raw", "q90_raw", "lower", "upper"]
    frame = pd.read_csv(io.BytesIO(data), dtype={"unique_id": str})
    if set(required)-set(frame.columns):
        raise ValueError("Use E02 interval_predictions.csv, not the long predictions.csv")
    frame = frame[required].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Empty or missing forecast values")
    for column in ["cutoff", "ds"]:
        dates = pd.DatetimeIndex(pd.to_datetime(frame[column], errors="raise"))
        if dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("Expected timezone-naive daily forecast dates")
        frame[column] = dates
    numbers = frame[["horizon", "y", "mean_estimate", "median_estimate", "q10_raw", "q90_raw", "lower", "upper"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError("Invalid nonnegative forecast values")
    if frame.duplicated(KEY).any() or not ((frame.ds-frame.cutoff).dt.days == frame.horizon).all():
        raise ValueError("Duplicate or misaligned forecast rows")
    if not np.equal(frame.horizon, np.floor(frame.horizon)).all() or not frame.horizon.between(1, contract["horizon"]).all():
        raise ValueError("Unexpected forecast horizon")
    if sorted(frame.cutoff.dt.strftime("%Y-%m-%d").unique()) != contract["cutoffs"]:
        raise ValueError("Unexpected validation forecast windows")
    ids = set(frame.unique_id)
    if len(ids) != contract["n_series"]:
        raise ValueError("Forecast series population changed")
    for _, part in frame.groupby("cutoff"):
        if set(part.unique_id) != ids or len(part) != len(ids)*contract["horizon"]:
            raise ValueError("Incomplete forecast grid")
    bounds = frame[["q10_raw", "median_estimate", "q90_raw"]].to_numpy()
    if not (np.allclose(frame.lower, bounds.min(axis=1), atol=1e-10, rtol=0)
            and np.allclose(frame.upper, bounds.max(axis=1), atol=1e-10, rtol=0)):
        raise ValueError("Forecast envelope does not match the declared E02 policy")
    frame["category"] = frame.unique_id.str.split("_").str[0]
    return frame.sort_values(KEY).reset_index(drop=True)


def series_metrics(frame):
    if frame.empty:
        raise ValueError("No observations in the selected forecast")
    total = float(frame.y.sum())
    result = {"n": len(frame), "actual_total": total,
              "coverage": float(((frame.y >= frame.lower) & (frame.y <= frame.upper)).mean()),
              "mean_width": float((frame.upper-frame.lower).mean())}
    for name, column in [("median", "median_estimate"), ("mean", "mean_estimate")]:
        errors = np.abs(frame[column]-frame.y)
        result[f"{name}_mae"] = float(errors.mean())
        result[f"{name}_wape"] = float(errors.sum()/total) if total else np.nan
    return result

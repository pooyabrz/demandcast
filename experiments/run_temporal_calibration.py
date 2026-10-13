"""Replay past-only, expansion-only interval calibration on frozen E02 forecasts.

This is an empirical rolling CQR-style diagnostic, not split CQR or a
distribution-free coverage guarantee. No model is trained by this script.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


KEY = ["unique_id", "cutoff", "ds", "horizon"]
ALPHA = 0.2
FORECASTS = {
    "lightgbm": "mean_estimate", "lightgbm_l1": "median_estimate",
    "mean28": "mean28", "seasonal_naive": "seasonal_naive",
}
VALUES = ["y", *FORECASTS.values(), "q10_raw", "q90_raw", "lower", "upper"]
BANDS = ["1-7", "8-14", "15-28"]


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(actual, reference):
    fields = ["stage", "source_sha256", "reference_run_sha256", "config",
              "quantile_levels", "nominal_target_coverage", "interval_policy",
              "calibration", "inclusive_bounds", "n_series", "observed_days",
              "control_reproduced_within_tolerance", "holdout_evaluated"]
    for field in fields:
        if field not in actual or actual[field] != reference[field]:
            raise ValueError(f"E02 receipt changed: {field}")
    if (actual["stage"] != "validation" or actual["holdout_evaluated"] is not False
            or actual["control_reproduced_within_tolerance"] is not True):
        raise ValueError("Expected the reproduced E02 validation experiment")


def validate_frame(frame, contract):
    missing = set(KEY + VALUES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing interval_predictions.csv columns: {sorted(missing)}")
    frame = frame[KEY + VALUES].copy()
    if frame.empty or frame[KEY + VALUES].isna().any().any():
        raise ValueError("Empty predictions or missing values")
    if frame.unique_id.astype(str).str.strip().eq("").any():
        raise ValueError("Empty series identifiers")
    frame["unique_id"] = frame.unique_id.astype(str)
    for name in ["ds", "cutoff"]:
        dates = pd.DatetimeIndex(pd.to_datetime(frame[name], errors="raise"))
        if dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError(f"Expected timezone-naive daily dates: {name}")
        frame[name] = dates
    for name in VALUES + ["horizon"]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame[VALUES + ["horizon"]].to_numpy()).all():
        raise ValueError("Nonfinite actuals, bounds or forecasts")
    if (frame[VALUES] < 0).any().any():
        raise ValueError("Expected nonnegative unit sales and forecasts")
    if not np.allclose(frame.y, np.round(frame.y), rtol=0, atol=1e-9):
        raise ValueError("This M5 protocol expects integer unit sales")
    if not np.equal(frame.horizon, np.floor(frame.horizon)).all():
        raise ValueError("Noninteger forecast horizons")
    frame["horizon"] = frame.horizon.astype(int)
    if not frame.horizon.between(1, contract["horizon"]).all():
        raise ValueError("Unexpected forecast horizons")
    if not ((frame.ds - frame.cutoff).dt.days == frame.horizon).all():
        raise ValueError("Target date does not match cutoff + horizon")
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate series/date/cutoff/horizon keys")
    cutoffs = sorted(frame.cutoff.dt.strftime("%Y-%m-%d").unique())
    if cutoffs != sorted(contract["validation_cutoffs"]):
        raise ValueError("Unexpected validation cutoffs; holdout is not allowed")
    ids = set(frame.unique_id)
    if len(ids) != contract["n_series"]:
        raise ValueError("Series population changed")
    for cutoff, part in frame.groupby("cutoff", sort=True):
        if set(part.unique_id) != ids or len(part) != len(ids) * contract["horizon"]:
            raise ValueError(f"Incomplete series/horizon grid at {cutoff}")
    tails = frame[["q10_raw", "median_estimate", "q90_raw"]].to_numpy()
    if not (np.allclose(frame.lower, tails.min(axis=1), rtol=0, atol=1e-10)
            and np.allclose(frame.upper, tails.max(axis=1), rtol=0, atol=1e-10)):
        raise ValueError("Bounds differ from E02's declared envelope policy")
    frame["horizon_band"] = np.select(
        [frame.horizon <= 7, frame.horizon <= 14], BANDS[:2], default=BANDS[2])
    frame["sales_state"] = np.where(frame.y == 0, "zero", "positive")
    frame["predicted_mean_bin"] = pd.cut(
        frame.mean_estimate, [0, .5, 1, 2, np.inf], right=False,
        labels=["0-0.5", "0.5-1", "1-2", "2+"]).astype(str)
    return frame.sort_values(KEY).reset_index(drop=True)


def summarize_frozen(frame, groups=None):
    groups = groups or []
    partitions = frame.groupby(groups, observed=True, sort=True) if groups else [((), frame)]
    rows = []
    for key, part in partitions:
        key = key if isinstance(key, tuple) else (key,)
        y, lo, hi = part.y.to_numpy(), part.lower.to_numpy(), part.upper.to_numpy()
        qlo, qhi, mid = part.q10_raw.to_numpy(), part.q90_raw.to_numpy(), part.median_estimate.to_numpy()
        rows.append({**dict(zip(groups, key)), "n_observations": len(part),
                     "nominal_target_coverage": 1 - ALPHA,
                     "coverage_raw": np.mean((qlo <= y) & (y <= qhi)),
                     "coverage": np.mean((lo <= y) & (y <= hi)),
                     "mean_width": np.mean(hi-lo), "mean_lower": np.mean(lo), "mean_upper": np.mean(hi),
                     "pinball_q10": np.mean(np.maximum(.1*(y-qlo), -.9*(y-qlo))),
                     "pinball_q90": np.mean(np.maximum(.9*(y-qhi), -.1*(y-qhi))),
                     "raw_crossing_rate": np.mean(qlo > qhi),
                     "median_outside_raw_bounds_rate": np.mean((mid < qlo) | (mid > qhi)),
                     "bounds_adjustment_rate": np.mean((lo != qlo) | (hi != qhi))})
    return pd.DataFrame(rows)


def summarize_points(frame):
    rows = []
    total = float(frame.y.sum())
    for name, column in FORECASTS.items():
        error = frame[column].to_numpy() - frame.y.to_numpy()
        rows.append({"model": name, "mae": np.abs(error).mean(),
                     "wape": np.abs(error).sum()/total if total else np.nan,
                     "bias": error.mean(), "n_observations": len(frame)})
    return pd.DataFrame(rows)


def compare_summary(actual, expected, dimensions, label):
    actual, expected = actual.copy(), expected.copy()
    keys = dimensions or ["all"]
    if not dimensions:
        actual["all"] = expected["all"] = "all"
    metrics = [column for column in expected if column not in keys]
    compared = expected.merge(actual, on=keys, how="outer", suffixes=("_expected", "_actual"),
                              indicator=True, validate="one_to_one")
    if not compared._merge.eq("both").all():
        raise ValueError(f"Summary groups changed: {label}")
    rows = []
    for _, row in compared.iterrows():
        for metric in metrics:
            wanted, observed = float(row[f"{metric}_expected"]), float(row[f"{metric}_actual"])
            passed = (wanted == observed if metric == "n_observations" else
                      np.isclose(wanted, observed, rtol=1e-6, atol=1e-7, equal_nan=True))
            rows.append({"source": label, "group": json.dumps({k: str(row[k]) for k in keys}),
                         "metric": metric, "expected": wanted, "actual": observed, "passed": bool(passed)})
    result = pd.DataFrame(rows)
    if not result.passed.all():
        raise ValueError(f"Frozen summary did not reproduce: {label}")
    return result


def finite_sample_quantile(scores, alpha=ALPHA):
    scores = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected finite calibration scores and 0 < alpha < 1")
    rank = int(((len(scores)+1) * (Decimal(1)-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    if rank > len(scores):
        raise ValueError("Too few calibration scores for a finite correction at this alpha")
    return float(np.partition(scores, rank-1)[rank-1]), rank


def temporal_calibrate(frame):
    """Pool only complete earlier forecast windows; never use current-window labels."""
    last_targets = frame.groupby("cutoff").ds.max()
    outputs, audit = [], []
    for cutoff in sorted(frame.cutoff.unique()):
        cutoff = pd.Timestamp(cutoff)
        eligible = last_targets.index[(last_targets.index < cutoff) & (last_targets <= cutoff)]
        if len(eligible) == 0:
            continue  # Initial window supplies calibration data only.
        history = frame[frame.cutoff.isin(eligible)]
        current = frame[frame.cutoff == cutoff]
        baseline = current.copy()
        baseline["method"], baseline["applied_expansion"] = "uncalibrated", 0.0
        outputs.append(baseline)
        adjusted = current.copy()
        adjusted["method"], adjusted["applied_expansion"] = "temporal_cqr", np.nan
        for band, part in current.groupby("horizon_band", observed=True, sort=True):
            past = history[history.horizon_band == band]
            signed_scores = np.maximum(past.lower-past.y, past.y-past.upper)
            signed_quantile, rank = finite_sample_quantile(signed_scores)
            expansion = max(0.0, signed_quantile)
            selected = adjusted.horizon_band == band
            adjusted.loc[selected, "lower"] = np.maximum(0.0, part.lower-expansion)
            adjusted.loc[selected, "upper"] = part.upper+expansion
            adjusted.loc[selected, "applied_expansion"] = expansion
            audit.append({"evaluation_cutoff": cutoff, "horizon_band": band,
                          "calibration_cutoffs": json.dumps([str(x.date()) for x in eligible]),
                          "n_calibration": len(past), "order_statistic_rank": rank,
                          "signed_score_quantile": signed_quantile, "applied_expansion": expansion,
                          "calibration_target_start": past.ds.min(), "calibration_target_end": past.ds.max(),
                          "past_only_check": bool(past.ds.max() <= cutoff)})
        if adjusted.applied_expansion.isna().any():
            raise ValueError("Missing correction for a forecast horizon band")
        outputs.append(adjusted)
    if not outputs:
        raise ValueError("No evaluation window has a complete earlier calibration window")
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audit)


def add_diagnostics(frame):
    frame = frame.copy()
    y, lo, hi = frame.y.to_numpy(), frame.lower.to_numpy(), frame.upper.to_numpy()
    frame["lower_miss"] = y < lo
    frame["upper_miss"] = y > hi
    frame["covered"] = (lo <= y) & (y <= hi)
    frame["width"] = hi-lo
    frame["interval_score_80"] = (hi-lo) + (2/ALPHA)*(np.maximum(lo-y, 0)+np.maximum(y-hi, 0))
    frame["median_outside"] = (frame.median_estimate < frame.lower) | (frame.median_estimate > frame.upper)
    return frame


def summarize_diagnostics(frame, groups=None):
    groups = groups or []
    partitions = frame.groupby(groups, observed=True, sort=True) if groups else [((), frame)]
    rows = []
    for key, part in partitions:
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(groups, key)), "n_observations": len(part),
                     "nominal_target_coverage": 1-ALPHA, "coverage": part.covered.mean(),
                     "lower_miss_rate": part.lower_miss.mean(), "upper_miss_rate": part.upper_miss.mean(),
                     "mean_width": part.width.mean(), "interval_score_80": part.interval_score_80.mean(),
                     "mean_expansion": part.applied_expansion.mean() if "applied_expansion" in part else 0.0,
                     "median_outside_rate": part.median_outside.mean()})
    return pd.DataFrame(rows)


def run(input_dir, output_dir, reference_dir):
    started = time.perf_counter()
    input_dir, output_dir, reference_dir = map(Path, [input_dir, output_dir, reference_dir])
    if output_dir.resolve() in {input_dir.resolve(), reference_dir.resolve()}:
        raise ValueError("Output directory must differ from input and reference directories")
    receipt = json.loads((input_dir/"run.json").read_text(encoding="utf-8"))
    reference = json.loads((reference_dir/"run.json").read_text(encoding="utf-8"))
    check_manifest(receipt, reference)
    contract = json.loads((reference_dir/"split_contract.json").read_text(encoding="utf-8"))
    frame = validate_frame(pd.read_csv(input_dir/"interval_predictions.csv", dtype={"unique_id": str}), contract)
    comparisons = []
    sources = [("interval_metrics.csv", []), ("interval_by_band.csv", ["horizon_band"]),
               ("interval_by_sales_state.csv", ["sales_state"])]
    for filename, groups in sources:
        comparisons.append(compare_summary(summarize_frozen(frame, groups),
                           pd.read_csv(reference_dir/filename), groups, filename))
    comparisons.append(compare_summary(summarize_points(frame),
                       pd.read_csv(reference_dir/"point_overall_metrics.csv"), ["model"], "point_overall_metrics.csv"))
    evaluated, audit = temporal_calibrate(frame)
    expected_eval = len(frame) - contract["n_series"]*contract["horizon"]
    if len(evaluated[evaluated.method == "temporal_cqr"]) != expected_eval:
        raise ValueError("Evaluation population differs from the frozen two-window protocol")
    if not audit.past_only_check.all():
        raise ValueError("Calibration contains labels unavailable at the forecast cutoff")
    evaluated = add_diagnostics(evaluated)
    if evaluated.median_outside.any() or (evaluated.width < 0).any():
        raise ValueError("Invalid calibrated bounds")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output_dir/"evaluation_predictions.csv", index=False)
    audit.to_csv(output_dir/"calibration_table.csv", index=False)
    pd.concat(comparisons, ignore_index=True).to_csv(output_dir/"summary_reproduction.csv", index=False)
    all_windows = add_diagnostics(frame)
    summarize_diagnostics(all_windows).to_csv(output_dir/"baseline_metrics_all_windows.csv", index=False)
    summaries = {"evaluation_metrics.csv": ["method"], "evaluation_by_fold.csv": ["method", "cutoff"],
                 "evaluation_by_band.csv": ["method", "horizon_band"],
                 "evaluation_by_sales_state.csv": ["method", "sales_state"],
                 "evaluation_by_predicted_mean.csv": ["method", "predicted_mean_bin"]}
    for filename, groups in summaries.items():
        summarize_diagnostics(evaluated, groups).to_csv(output_dir/filename, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "validation_replay",
        "experiment": "E03: past-only horizon-band interval calibration",
        "source_sha256": receipt["source_sha256"],
        "interval_predictions_sha256": hash_file(input_dir/"interval_predictions.csv"),
        "e02_run_sha256": hash_file(input_dir/"run.json"), "config": receipt["config"],
        "nominal_target_coverage": 1-ALPHA,
        "score": "max(lower-y, y-upper); signed score retained",
        "correction": "max(0, k-th smallest score), k=ceil((n+1)*(1-alpha)); separately by horizon band",
        "history_policy": "all complete earlier E02 windows with last target date <= evaluation cutoff",
        "bounds_policy": "max(0, lower-correction), upper+correction; central forecasts unchanged",
        "calibration_routing": "forecast horizon only; future zero/positive actuals never used for routing",
        "n_evaluation_observations_per_method": expected_eval,
        "evaluation_cutoffs": [str(x.date()) for x in sorted(evaluated.cutoff.unique())],
        "summary_reproduced_within_tolerance": True, "past_only_check": True,
        "model_refitted": False, "holdout_evaluated": False,
        "coverage_guarantee": "none; dependent panel/time-series scores and changing fitted base models",
        "selection_scope": "exploratory retrospective replay; no automatic promotion or new blind test",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "elapsed_seconds": round(time.perf_counter()-started, 3),
    }
    (output_dir/"run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summarize_diagnostics(evaluated, ["method"]).to_string(index=False))
    print(f"Results: {output_dir}; past-only checks passed; no holdout or model fitting.")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="outputs/interval-validation")
    parser.add_argument("--output", default="outputs/calibration-validation")
    parser.add_argument("--reference-dir", default="reports/interval_reference")
    args = parser.parse_args(argv)
    try:
        run(args.input_dir, args.output, args.reference_dir)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"Calibration stopped: {exc}\n")


if __name__ == "__main__":
    main()

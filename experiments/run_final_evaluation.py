"""Freeze new-cohort forecasts before loading final outcomes, then score them.

The original DemandCast checkout and previous addons are required. This script
does not replace core features, search models or resample the registered cohort.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from project_preflight import hash_file, inspect_project
from register_final_evaluation import choose_cohort, normal_id

MODELS = ["seasonal_naive", "mean28", "lightgbm", "lightgbm_l1", "lightgbm_q10", "lightgbm_q90"]
POINTS = {"seasonal_naive": "seasonal_naive", "mean28": "mean28",
          "lightgbm": "mean_estimate", "lightgbm_l1": "median_estimate"}
META = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
KEY = ["unique_id", "cutoff", "ds", "horizon"]
FORECAST_COLUMNS = KEY+["model", "prediction"]
RUNNER = "experiments/run_final_evaluation.py"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def check_locked_state(registration, report, base_config):
    if registration["status"] != "registered_not_evaluated" or not report["ready_for_local_review"]:
        raise ValueError("A valid registration and passing local preflight are required")
    if (registration["forecast_cutoff"], registration["target_start"], registration["target_end"],
        registration["horizon"], registration["history_days_in_prepared_panel"]) != (
            "2016-04-24", "2016-04-25", "2016-05-22", 28, 730):
        raise ValueError("The registered dates/history/horizon differ from this protocol")
    cohort = registration["cohort"]
    if (cohort["n_items"], cohort["n_series"], len(cohort["stores"]), cohort["selection_seed"],
        cohort["item_overlap_count"], cohort["series_overlap_count"]) != (50, 500, 10, 20261013, 0, 0):
        raise ValueError("The registered cohort differs from the 50-product protocol")
    if (len(set(cohort["selected_item_ids"])) != 50 or len(cohort["selected_raw_series_ids"]) != 500
            or len({normal_id(value) for value in cohort["selected_raw_series_ids"]}) != 500):
        raise ValueError("The registered cohort has missing or duplicate identities")
    recipe = registration["recipe_config"]
    fixed = {"seed": 42, "horizon": 28, "n_windows": 3, "n_series": 500, "history_days": 730,
             "training_origin_days": 365, "origin_stride": 7, "iterations": 200, "threads": 4, "models": MODELS}
    if recipe != fixed:
        raise ValueError("The registered forecast recipe changed")
    if list(base_config["models"]) != MODELS[:3] or any(
            base_config.get(key) != value for key, value in recipe.items() if key != "models"):
        raise ValueError("Current configs/m5.yaml differs from the registered development recipe")
    if report["versions"] != registration["versions"]:
        raise ValueError("Installed dependency versions changed since registration")
    if report["prepared_data"]["sha256"] != registration["development_panel_sha256"]:
        raise ValueError("Development input changed since registration")
    old, current = registration["source_hashes"], report["source_hashes"]
    required = {"experiments/run_l1_ablation.py", "experiments/run_interval_benchmark.py"}
    if not required.issubset(old) or not any(key.startswith(("src/demandcast/", "demandcast/")) for key in old):
        raise ValueError("Registration does not bind the original core and objective implementations")
    changed = [key for key, value in old.items() if current.get(key) != value]
    added = set(current)-set(old)-{RUNNER}
    if changed or added:
        raise ValueError(f"Registered source code changed: {changed}; unexpected new code: {sorted(added)}")


def calendar_split(path, registration):
    calendar = pd.read_csv(path, usecols=["d", "date"], dtype={"d": str})
    calendar["date"] = pd.to_datetime(calendar.date, errors="raise")
    dates = pd.DatetimeIndex(calendar.date)
    if (dates.tz is not None or not dates.equals(dates.normalize()) or dates.hasnans
            or calendar.d.duplicated().any() or calendar.date.duplicated().any()
            or not calendar.d.str.fullmatch(r"d_[1-9]\d*").all()):
        raise ValueError("Malformed M5 day/date mapping")
    numbers = calendar.d.str[2:].astype(int)
    canonical = pd.Timestamp("2011-01-29")+pd.to_timedelta(numbers-1, unit="D")
    if not np.array_equal(calendar.date.to_numpy(dtype="datetime64[ns]"), canonical.to_numpy(dtype="datetime64[ns]")):
        raise ValueError("Calendar dates do not match M5 day numbering")
    wanted = pd.date_range(end=registration["target_end"], periods=registration["history_days_in_prepared_panel"])
    mapping = calendar.set_index("date").d
    if not wanted.isin(mapping.index).all():
        raise ValueError("Calendar does not contain the registered 730-day interval")
    cutoff = pd.Timestamp(registration["forecast_cutoff"])
    history_dates, target_dates = wanted[wanted <= cutoff], wanted[wanted > cutoff]
    if len(history_dates) != 702 or len(target_dates) != 28 or str(target_dates[0].date()) != registration["target_start"]:
        raise ValueError("Calendar split differs from the registered forecast dates")
    return {"history": dict(zip(mapping.loc[history_dates], history_dates)),
            "targets": dict(zip(mapping.loc[target_dates], target_dates))}


def read_sales(path, raw_ids, day_mapping, chunksize=256):
    """Deserialize only selected day columns; filter series before collecting rows."""
    days = list(day_mapping)
    selected_ids = set(raw_ids)
    chunks = []
    types = {**{name: str for name in META}, **{name: np.float64 for name in days}}
    for chunk in pd.read_csv(path, usecols=META+days, dtype=types, chunksize=chunksize):
        part = chunk[chunk.id.isin(selected_ids)]
        if not part.empty:
            chunks.append(part.copy())
    if not chunks:
        raise ValueError("No registered sales series found")
    wide = pd.concat(chunks, ignore_index=True)
    if wide.id.duplicated().any() or set(wide.id) != selected_ids or wide[META].isna().any().any():
        raise ValueError("Sales identities/metadata do not match the registered cohort")
    if not (wide.id.map(normal_id) == wide.item_id+"_"+wide.store_id).all():
        raise ValueError("M5 item/store metadata disagrees with series identifiers")
    values = wide[days].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Expected finite nonnegative integer unit sales")
    frame = wide.melt(id_vars=META, value_vars=days, var_name="d", value_name="y")
    frame["unique_id"] = frame.id.map(normal_id)
    frame["ds"] = frame.d.map(day_mapping)
    frame = frame[["unique_id", "ds", "y", *META[1:]]].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    validate_grid(frame, {normal_id(value) for value in raw_ids}, list(day_mapping.values()))
    return frame


def validate_grid(frame, ids, dates):
    required = {"unique_id", "ds", "y"}
    if not required.issubset(frame) or frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Missing actual/history fields")
    dates = pd.DatetimeIndex(dates)
    observed = pd.DatetimeIndex(pd.to_datetime(frame.ds, errors="raise"))
    if (observed.tz is not None or not observed.equals(observed.normalize())
            or set(frame.unique_id) != set(ids) or set(observed) != set(dates)
            or frame.duplicated(["unique_id", "ds"]).any() or len(frame) != len(ids)*len(dates)):
        raise ValueError("Incomplete or unexpected series/date grid")
    values = frame.y.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Invalid observed unit sales")


def forecast_models(history, config, cutoff, factory, panel_builder):
    """The injected model factory receives a panel containing observed history only."""
    cutoff = pd.Timestamp(cutoff)
    history = history.copy()
    history["ds"] = pd.to_datetime(history.ds)
    observed_dates = pd.DatetimeIndex(sorted(history.ds.unique())).as_unit("ns")
    if history.empty or observed_dates[-1] != cutoff or (history.ds > cutoff).any():
        raise ValueError("Training data must end exactly at the forecast cutoff")
    validate_grid(history, set(history.unique_id), observed_dates)
    panel = panel_builder(history)
    if (not pd.DatetimeIndex(panel.dates).as_unit("ns").equals(observed_dates)
            or set(panel.ids) != set(history.unique_id) or panel.values.shape != (len(panel.ids), len(observed_dates))):
        raise ValueError("Core panel conversion changed the observed training grid")
    target_dates = pd.date_range(cutoff+pd.Timedelta(days=1), periods=config.horizon)
    frames = []
    for name in config.models:
        started = time.perf_counter()
        model = factory(name, config)
        model.fit(panel)
        values = np.asarray(model.predict(config.horizon), dtype=float)
        if values.shape != (len(panel.ids), config.horizon) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid forecast output: {name}")
        frames.append(pd.DataFrame({"unique_id": np.repeat(panel.ids, config.horizon), "cutoff": cutoff,
                                    "ds": np.tile(target_dates, len(panel.ids)),
                                    "horizon": np.tile(np.arange(1, config.horizon+1), len(panel.ids)),
                                    "model": name, "prediction": values.ravel()}))
        print(f"Fitted {name}: {time.perf_counter()-started:.1f}s", flush=True)
        del model
    return pd.concat(frames, ignore_index=True)


def validate_forecasts(frame, registration):
    if set(frame.columns) != set(FORECAST_COLUMNS) or frame.empty:
        raise ValueError("Frozen forecast file must contain forecasts only, without target sales")
    frame = frame.copy()
    for name in ["horizon", "prediction"]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    for name in ["cutoff", "ds"]:
        frame[name] = pd.to_datetime(frame[name], errors="raise")
        dates = pd.DatetimeIndex(frame[name])
        if dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("Forecast dates must be timezone-naive daily dates")
    numeric = frame[["horizon", "prediction"]].to_numpy(dtype=float)
    if (frame.isna().any().any() or not np.isfinite(numeric).all() or (frame.prediction < 0).any()
            or not np.equal(frame.horizon, np.floor(frame.horizon)).all()):
        raise ValueError("Invalid frozen forecast values")
    ids = {normal_id(value) for value in registration["cohort"]["selected_raw_series_ids"]}
    horizon = registration["horizon"]
    expected_dates = pd.date_range(registration["target_start"], periods=horizon)
    if (set(frame.model) != set(MODELS) or set(frame.unique_id) != ids
            or set(frame.cutoff) != {pd.Timestamp(registration["forecast_cutoff"])}
            or not frame.horizon.between(1, horizon).all()
            or not ((frame.ds-frame.cutoff).dt.days == frame.horizon).all()
            or set(frame.ds) != set(expected_dates) or frame.duplicated(KEY+["model"]).any()
            or len(frame) != len(ids)*horizon*len(MODELS)):
        raise ValueError("Frozen forecasts do not cover the exact registered model/series/horizon grid")
    return frame.sort_values(KEY+["model"]).reset_index(drop=True)


def score_forecasts(forecasts, targets, registration):
    forecasts = validate_forecasts(forecasts, registration)
    targets = targets.copy()
    targets["ds"] = pd.to_datetime(targets.ds)
    ids = {normal_id(value) for value in registration["cohort"]["selected_raw_series_ids"]}
    dates = pd.date_range(registration["target_start"], periods=registration["horizon"])
    validate_grid(targets, ids, dates)
    wide = forecasts.pivot(index=KEY, columns="model", values="prediction").reset_index()
    wide = wide.merge(targets, on=["unique_id", "ds"], validate="one_to_one")
    wide = wide.rename(columns={"lightgbm": "mean_estimate", "lightgbm_l1": "median_estimate",
                                "lightgbm_q10": "q10_raw", "lightgbm_q90": "q90_raw"})
    tails = wide[["q10_raw", "median_estimate", "q90_raw"]]
    wide["lower"], wide["upper"] = tails.min(axis=1), tails.max(axis=1)
    wide["width"] = wide.upper-wide.lower
    wide["lower_miss"], wide["upper_miss"] = wide.y < wide.lower, wide.y > wide.upper
    wide["covered"] = (wide.y >= wide.lower) & (wide.y <= wide.upper)
    wide["covered_raw"] = (wide.y >= wide.q10_raw) & (wide.y <= wide.q90_raw)
    wide["raw_quantile_crossing"] = wide.q10_raw > wide.q90_raw
    wide["median_outside_raw_bounds"] = (wide.median_estimate < wide.q10_raw) | (wide.median_estimate > wide.q90_raw)
    wide["bounds_adjusted"] = (wide.lower != wide.q10_raw) | (wide.upper != wide.q90_raw)
    wide["interval_score_80"] = wide.width+10*(np.maximum(wide.lower-wide.y, 0)+np.maximum(wide.y-wide.upper, 0))
    wide["pinball_q10"] = np.maximum(.1*(wide.y-wide.q10_raw), -.9*(wide.y-wide.q10_raw))
    wide["pinball_q90"] = np.maximum(.9*(wide.y-wide.q90_raw), -.1*(wide.y-wide.q90_raw))
    wide["horizon_band"] = np.select([wide.horizon <= 7, wide.horizon <= 14], ["1-7", "8-14"], default="15-28")
    wide["sales_state"] = np.where(wide.y == 0, "zero", "positive")
    wide["predicted_mean_bin"] = pd.cut(wide.mean_estimate, [0, .5, 1, 2, np.inf], right=False,
                                         labels=["0-0.5", "0.5-1", "1-2", "2+"]).astype(str)
    return wide.sort_values(KEY).reset_index(drop=True)


def partitions(frame, groups):
    return frame.groupby(groups, observed=True, sort=True) if groups else [((), frame)]


def point_metrics(frame, groups=None):
    groups, rows = groups or [], []
    for key, part in partitions(frame, groups):
        key = key if isinstance(key, tuple) else (key,)
        actual_total = float(part.y.sum())
        for model, column in POINTS.items():
            error = part[column].to_numpy()-part.y.to_numpy()
            rows.append({**dict(zip(groups, key)), "model": model, "n_observations": len(part),
                         "mae": float(np.abs(error).mean()), "wape": float(np.abs(error).sum()/actual_total) if actual_total else np.nan,
                         "bias": float(error.mean()), "aggregate_volume_bias_fraction": float(error.sum()/actual_total) if actual_total else np.nan,
                         "actual_sales_total": actual_total, "predicted_sales_total": float(part[column].sum())})
    return pd.DataFrame(rows)


def interval_metrics(frame, groups=None):
    groups, rows = groups or [], []
    for key, part in partitions(frame, groups):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(groups, key)), "method": "uncalibrated_envelope", "n_observations": len(part),
                     "nominal_target_coverage": .8, "coverage": float(part.covered.mean()), "coverage_raw": float(part.covered_raw.mean()),
                     "lower_miss_rate": float(part.lower_miss.mean()), "upper_miss_rate": float(part.upper_miss.mean()),
                     "mean_width": float(part.width.mean()), "interval_score_80": float(part.interval_score_80.mean()),
                     "pinball_q10": float(part.pinball_q10.mean()), "pinball_q90": float(part.pinball_q90.mean()),
                     "raw_crossing_rate": float(part.raw_quantile_crossing.mean()),
                     "median_outside_raw_bounds_rate": float(part.median_outside_raw_bounds.mean()),
                     "bounds_adjustment_rate": float(part.bounds_adjusted.mean())})
    return pd.DataFrame(rows)


def comparisons(metrics):
    table, rows = metrics.set_index("model"), []
    pairs = [("lightgbm_l1", base, "mae") for base in ["mean28", "lightgbm", "seasonal_naive"]]
    pairs += [("lightgbm", base, "wape") for base in ["mean28", "seasonal_naive"]]
    for candidate, baseline, metric in pairs:
        wanted, observed = float(table.loc[baseline, metric]), float(table.loc[candidate, metric])
        rows.append({"candidate": candidate, "baseline": baseline, "metric": metric,
                     "candidate_value": observed, "baseline_value": wanted,
                     "relative_improvement_fraction": (wanted-observed)/wanted if wanted else np.nan})
    return pd.DataFrame(rows)


def verified_context(root, registration_path, sales_path, calendar_path):
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    report = inspect_project(root, "local")
    if not report["ready_for_local_review"]:
        raise ValueError("Project preflight failed: "+"; ".join(report["errors"]))
    from demandcast.config import load_config
    check_locked_state(registration, report, load_config(root/"configs/m5.yaml").to_dict())
    if hash_file(sales_path) != registration["raw_catalogue_file_sha256"]:
        raise ValueError("Raw sales file changed since cohort registration")
    catalogue = pd.read_csv(sales_path, usecols=["id", "item_id", "store_id"], dtype=str)
    known = pd.read_csv(root/"data/processed/m5.csv", usecols=["unique_id"], dtype=str).unique_id.unique()
    chosen = choose_cohort(catalogue, known)
    if chosen != registration["cohort"]:
        raise ValueError("Registered cohort does not reproduce from identity metadata")
    split = calendar_split(calendar_path, registration)
    bindings = {"registration_sha256": hash_file(registration_path), "raw_sales_sha256": registration["raw_catalogue_file_sha256"],
                "calendar_sha256": hash_file(calendar_path), "development_panel_sha256": registration["development_panel_sha256"],
                "source_hashes": report["source_hashes"], "versions": report["versions"],
                "python": report["python"], "platform": report["platform"],
                "release_helper_hashes": {name: hash_file(root/name) for name in
                    ["scripts/project_preflight.py", "scripts/register_final_evaluation.py"]}}
    return registration, split, bindings


def claim_attempt(root, registration_path, output, bindings):
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root/"outputs") or output == root/"outputs":
        raise ValueError("Final output must be a new directory below this project's outputs/")
    if output.exists():
        raise ValueError("Output already exists; final forecasts/results are never overwritten")
    marker = registration_path.parent/"final_evaluation_attempt.json"
    payload = {"created_at_utc": now(), "status": "forecast_started", "output_directory": output.relative_to(root).as_posix(),
               "registration_sha256": bindings["registration_sha256"], "target_sales_deserialized": False}
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    output.mkdir(parents=True, exist_ok=False)
    return marker, payload


def verify_seal(output, receipt, bindings):
    if receipt["status"] != "forecasts_frozen" or receipt["bindings"] != bindings:
        raise ValueError("Forecast receipt or registered inputs/code/environment changed")
    if hash_file(output/"forecasts.csv") != receipt["forecasts_sha256"]:
        raise ValueError("Frozen forecasts changed; do not refit or score altered predictions")


def make_factory():
    from inspect import getfile
    from demandcast.features import OriginFeatures
    from demandcast.models import make_model
    from run_l1_ablation import L1DirectTree
    from run_interval_benchmark import QuantileDirectTree
    for implementation, name in [(L1DirectTree, "run_l1_ablation.py"), (QuantileDirectTree, "run_interval_benchmark.py")]:
        if Path(getfile(implementation)).resolve() != (ROOT/"experiments"/name).resolve():
            raise ValueError("Objective implementation was imported from a different checkout")

    def factory(name, config):
        if name == "lightgbm_l1":
            return L1DirectTree(config, OriginFeatures)
        if name in ["lightgbm_q10", "lightgbm_q90"]:
            return QuantileDirectTree(config, OriginFeatures, .1 if name.endswith("q10") else .9)
        return make_model(name, config)
    return factory


def forecast_phase(root, registration_path, sales_path, calendar_path, output):
    registration, split, bindings = verified_context(root, registration_path, sales_path, calendar_path)
    marker, attempt = claim_attempt(root, registration_path, output, bindings)
    started = time.perf_counter()
    receipt = {"created_at_utc": now(), "status": "forecast_started", "bindings": bindings,
               "recipe_config": registration["recipe_config"], "cohort": registration["cohort"],
               "forecast_cutoff": registration["forecast_cutoff"], "target_start": registration["target_start"],
               "target_end": registration["target_end"], "target_sales_deserialized": False}
    write_json(output/"forecast_run.json", receipt)
    try:
        history = read_sales(sales_path, registration["cohort"]["selected_raw_series_ids"], split["history"])
        from demandcast.data import to_panel
        forecasts = forecast_models(history, SimpleNamespace(**registration["recipe_config"]),
                                    registration["forecast_cutoff"], make_factory(), to_panel)
        forecasts = validate_forecasts(forecasts, registration)
        _, _, after = verified_context(root, registration_path, sales_path, calendar_path)
        if after != bindings:
            raise ValueError("Input or code changed during training")
        forecasts.to_csv(output/"forecasts.csv", index=False)
        (output/"registration_snapshot.json").write_bytes(registration_path.read_bytes())
        receipt.update({"status": "forecasts_frozen", "forecasts_sha256": hash_file(output/"forecasts.csv"),
                        "history_start": str(history.ds.min().date()), "history_end": str(history.ds.max().date()),
                        "observed_training_days": len(split["history"]), "n_forecast_rows": len(forecasts),
                        "n_observations_per_model": len(forecasts)//len(MODELS), "elapsed_seconds": round(time.perf_counter()-started, 3)})
        write_json(output/"forecast_run.json", receipt)
        attempt.update({"status": "forecasts_frozen", "forecasts_sha256": receipt["forecasts_sha256"],
                        "forecast_run_sha256": hash_file(output/"forecast_run.json")})
        write_json(marker, attempt)
    except Exception as exc:
        receipt.update({"status": "forecast_failed", "error": str(exc)})
        write_json(output/"forecast_run.json", receipt)
        attempt.update({"status": "forecast_failed", "error": str(exc)})
        write_json(marker, attempt)
        raise
    print(f"Frozen forecasts: {output/'forecasts.csv'}. No final outcome column was deserialized.")


def summary_markdown(points, intervals):
    def formatted(value, percentage=False):
        return "undefined (zero actual sales)" if not np.isfinite(value) else (f"{100*value:.2f}%" if percentage else f"{value:.4f}")
    rows = ["# Final product-cohort evaluation", "", "Frozen recipe retrained on 50 fresh products across 10 stores.",
            "Cutoff: 2016-04-24. Targets: 2016-04-25 to 2016-05-22.", "", "| Model | MAE | WAPE | Aggregate volume bias |", "| --- | ---: | ---: | ---: |"]
    for _, row in points.iterrows():
        rows.append(f"| {row.model} | {formatted(row.mae)} | {formatted(row.wape, True)} | {formatted(row.aggregate_volume_bias_fraction, True)} |")
    interval = intervals.iloc[0]
    rows += ["", f"Observations per model/interval: {int(interval.n_observations):,}.",
             f"Envelope coverage: {formatted(interval.coverage, True)}; mean width: {formatted(interval.mean_width)}; IS80: {formatted(interval.interval_score_80)}.",
             f"Lower/upper miss rates: {formatted(interval.lower_miss_rate, True)} / {formatted(interval.upper_miss_rate, True)}.", "",
             "L1 is median-oriented; Poisson is mean-oriented. Inspect expected volume separately from MAE.",
             "Comparisons use the same new-cohort targets. Prior validation scores are not a paired before/after comparison.",
             "The cohort is product-disjoint, while dates/store conditions are shared. This is not an independent dataset,",
             "a statistical-significance result or an inventory service-level guarantee. This cohort is now inspected:",
             "do not tune or resample and continue calling it untouched.", ""]
    return "\n".join(rows)


def score_phase(root, registration_path, sales_path, calendar_path, output):
    registration, split, bindings = verified_context(root, registration_path, sales_path, calendar_path)
    forecast_receipt = json.loads((output/"forecast_run.json").read_text(encoding="utf-8"))
    verify_seal(output, forecast_receipt, bindings)
    marker = registration_path.parent/"final_evaluation_attempt.json"
    attempt = json.loads(marker.read_text(encoding="utf-8"))
    if (attempt["registration_sha256"] != bindings["registration_sha256"] or
            (root/attempt["output_directory"]).resolve() != output.resolve() or
            attempt.get("forecasts_sha256") != forecast_receipt["forecasts_sha256"] or
            attempt.get("forecast_run_sha256") != hash_file(output/"forecast_run.json")):
        raise ValueError("Attempt record does not match this frozen run")
    final_receipt = output/"run.json"
    if final_receipt.exists():
        existing = json.loads(final_receipt.read_text(encoding="utf-8"))
        if existing["status"] == "scored" and existing["bindings"] == bindings:
            if attempt["status"] != "scored" or attempt.get("final_run_sha256") != hash_file(final_receipt):
                raise ValueError("The retained final receipt changed")
            for name, value in existing["artifact_hashes"].items():
                if hash_file(output/name) != value:
                    raise ValueError("An existing final result file changed")
            print(f"Already scored; retained existing results: {output/'SUMMARY.md'}")
            return
        raise ValueError("An incomplete final scoring attempt exists; preserve it and send its error for review")
    forecasts = validate_forecasts(pd.read_csv(output/"forecasts.csv", dtype={"unique_id": str}), registration)
    receipt = {"created_at_utc": now(), "status": "scoring_started", "stage": "final_product_cohort",
               "bindings": bindings, "recipe_config": registration["recipe_config"], "cohort": registration["cohort"],
               "forecast_cutoff": registration["forecast_cutoff"], "target_start": registration["target_start"],
               "target_end": registration["target_end"], "target_outcome_access_started": True,
               "target_sales_deserialized": None,
               "calibration": "none; frozen q10/L1/q90 envelope", "nominal_target_coverage": .8,
               "observed_training_days": forecast_receipt["observed_training_days"],
               "limitations": registration["limitations"]}
    write_json(final_receipt, receipt)
    attempt.update({"status": "scoring_started", "target_outcome_access_started": True})
    write_json(marker, attempt)
    try:
        targets = read_sales(sales_path, registration["cohort"]["selected_raw_series_ids"], split["targets"])
        receipt["target_sales_deserialized"] = True
        frame = score_forecasts(forecasts, targets, registration)
        _, _, after = verified_context(root, registration_path, sales_path, calendar_path)
        verify_seal(output, forecast_receipt, after)
        tables = {"interval_predictions.csv": frame, "point_overall_metrics.csv": point_metrics(frame),
                  "interval_metrics.csv": interval_metrics(frame)}
        tables["comparisons.csv"] = comparisons(tables["point_overall_metrics.csv"])
        for name, group in [("horizon", "horizon"), ("store", "store_id"), ("sales_state", "sales_state")]:
            tables[f"point_by_{name}.csv"] = point_metrics(frame, [group])
            tables[f"interval_by_{name}.csv"] = interval_metrics(frame, [group])
        tables["interval_by_band.csv"] = interval_metrics(frame, ["horizon_band"])
        tables["interval_by_predicted_mean.csv"] = interval_metrics(frame, ["predicted_mean_bin"])
        for name, table in tables.items():
            table.to_csv(output/name, index=False)
        (output/"SUMMARY.md").write_text(summary_markdown(tables["point_overall_metrics.csv"], tables["interval_metrics.csv"]), encoding="utf-8")
        receipt.update({"status": "scored", "n_series": frame.unique_id.nunique(), "n_items": frame.item_id.nunique(),
                        "n_observations_per_model": len(frame), "fit_policy": registration["fit_policy"],
                        "comparison_policy": registration["comparison_policy"],
                        "artifact_hashes": {name: hash_file(output/name) for name in [*tables, "SUMMARY.md", "registration_snapshot.json"]}})
        write_json(final_receipt, receipt)
        attempt.update({"status": "scored", "target_sales_deserialized": True,
                        "final_run_sha256": hash_file(final_receipt)})
        write_json(marker, attempt)
    except Exception as exc:
        receipt.update({"status": "scoring_failed", "error": str(exc)})
        write_json(final_receipt, receipt)
        attempt.update({"status": "scoring_failed", "error": str(exc)})
        write_json(marker, attempt)
        raise
    print(tables["point_overall_metrics.csv"].to_string(index=False))
    print(tables["interval_metrics.csv"].to_string(index=False))
    print(f"Final results: {output/'SUMMARY.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["forecast", "score"], required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path, default=Path("reports/final_evaluation/registration.json"))
    parser.add_argument("--sales", type=Path, default=Path("data/raw/m5/sales_train_evaluation.csv"))
    parser.add_argument("--calendar", type=Path, default=Path("data/raw/m5/calendar.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/final-cohort"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    resolve = lambda path: path.resolve() if path.is_absolute() else (root/path).resolve()
    try:
        if root != ROOT.resolve():
            raise ValueError("Run this script from its own assembled checkout; --root must match that checkout")
        operation = forecast_phase if args.phase == "forecast" else score_phase
        operation(root, resolve(args.registration), resolve(args.sales), resolve(args.calendar), resolve(args.output))
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Final evaluation stopped: {exc}\nPreserve the registration and attempt records; send the error for review.\n")


if __name__ == "__main__":
    main()

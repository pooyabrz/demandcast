"""Pair forecasts at the same series/date/origin and decompose absolute error."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("seasonal_naive", "mean28", "lightgbm")
KEY = ["unique_id", "cutoff", "ds", "horizon"]


def load_paired(path):
    frame = pd.read_csv(path, parse_dates=["ds", "cutoff"])
    required = set(KEY + ["stage", "model", "y", "prediction"])
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Empty data or missing required fields")
    if set(frame.stage) != {"validation"}:
        raise ValueError("This diagnostic requires validation predictions only")
    if set(frame.model) != set(MODELS):
        raise ValueError(f"Expected the v0.1 models: {MODELS}")
    if frame.duplicated(KEY + ["model"]).any():
        raise ValueError("Duplicate series/date/model forecast")
    if not np.isfinite(frame[["y", "prediction"]]).all().all():
        raise ValueError("Nonfinite sales or forecasts")
    if (frame[["y", "prediction"]] < 0).any().any():
        raise ValueError("Negative unit sales or forecasts")
    if not ((frame.ds - frame.cutoff).dt.days == frame.horizon).all():
        raise ValueError("Horizon does not match the date boundary")
    if not frame.groupby(KEY).y.nunique().eq(1).all():
        raise ValueError("Different actual sales for the same evaluation row")
    wide = frame.pivot(index=KEY, columns="model", values="prediction").reset_index()
    truth = frame.drop_duplicates(KEY)[KEY + ["y"]]
    wide = wide.merge(truth, on=KEY, validate="one_to_one")
    if wide[list(MODELS)].isna().any().any():
        raise ValueError("Models are not evaluated on the same observations")
    horizon = int(wide.horizon.max())
    for cutoff, part in wide.groupby("cutoff"):
        coverage = part.groupby("unique_id").horizon.agg(["min", "max", "nunique"])
        if not ((coverage["min"] == 1) & (coverage["max"] == horizon) &
                (coverage["nunique"] == horizon)).all():
            raise ValueError(f"Incomplete horizon coverage at {cutoff}")
    for model in MODELS:
        wide[f"ae_{model}"] = (wide[model] - wide.y).abs()
    wide["delta_ae"] = wide.ae_lightgbm - wide.ae_mean28
    wide["horizon_band"] = pd.cut(wide.horizon, [0, 7, 14, 28], labels=["1-7", "8-14", "15-28"])
    if wide.horizon_band.isna().any():
        raise ValueError("This notebook's bands require horizons of at most 28 days")
    wide["category"] = wide.unique_id.str.split("_").str[0]
    wide["store"] = wide.unique_id.str.extract(r"_(CA_\d|TX_\d|WI_\d)_evaluation$")[0]
    if not set(wide.category) <= {"FOODS", "HOBBIES", "HOUSEHOLD"} or wide.store.isna().any():
        raise ValueError("Expected original M5 evaluation-series identifiers")
    wide["sales_state"] = np.where(wide.y == 0, "Zero sales", "Positive sales")
    return frame, wide


def metric_table(frame, groups):
    records = []
    for key, part in frame.groupby(groups, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        context = dict(zip(groups, key))
        volume = float(part.y.sum())
        for model in MODELS:
            error = part[model] - part.y
            records.append({**context, "model": model, "n_observations": len(part),
                            "actual_sales": volume, "mae": float(error.abs().mean()),
                            "wape": float(error.abs().sum() / volume) if volume else np.nan,
                            "bias": float(error.mean()),
                            "zero_fraction": float((part.y == 0).mean())})
    return pd.DataFrame(records)


def decomposition(frame, dimension):
    return frame.groupby(dimension, observed=True).agg(
        n_observations=("y", "size"), actual_sales=("y", "sum"),
        zero_fraction=("y", lambda values: (values == 0).mean()),
        mae_lightgbm=("ae_lightgbm", "mean"), mae_mean28=("ae_mean28", "mean"),
        net_extra_absolute_error=("delta_ae", "sum"),
    ).reset_index()


def analyze(path):
    frame, paired = load_paired(path)
    series = paired.groupby("unique_id").agg(
        n_observations=("y", "size"), actual_sales=("y", "sum"),
        zero_fraction=("y", lambda values: (values == 0).mean()),
        mae_lightgbm=("ae_lightgbm", "mean"), mae_mean28=("ae_mean28", "mean"),
        absolute_error_lightgbm=("ae_lightgbm", "sum"),
        net_extra_absolute_error=("delta_ae", "sum"),
    )
    series["mae_difference"] = series.mae_lightgbm - series.mae_mean28
    delta = float(paired.delta_ae.sum())
    zeros = decomposition(paired, "sales_state")
    assert np.isclose(zeros.net_extra_absolute_error.sum(), delta)
    summary = {
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "input_rows": len(frame), "paired_observations": len(paired),
        "n_series": len(series), "cutoffs": sorted(paired.cutoff.dt.strftime("%Y-%m-%d").unique()),
        "zero_sales_fraction": float((paired.y == 0).mean()),
        "lgb_series_wins": int((series.mae_difference < -1e-8).sum()),
        "lgb_series_losses": int((series.mae_difference > 1e-8).sum()),
        "lgb_series_ties": int((series.mae_difference.abs() <= 1e-8).sum()),
        "net_extra_absolute_error": delta,
        "top50_absolute_error_share": float(series.absolute_error_lightgbm.nlargest(50).sum() /
                                             series.absolute_error_lightgbm.sum()),
        "scope": "500 sampled M5 product-store series; validation only; no causal claim",
        "retrospective_grouping_warning": "Actual future zero-sales labels are diagnostic groups, not forecasting inputs or model-routing rules",
    }
    tables = {
        "fold_metrics": metric_table(paired, ["cutoff"]),
        "horizon_metrics": metric_table(paired, ["horizon"]),
        "horizon_band_metrics": metric_table(paired, ["horizon_band"]),
        "category_metrics": metric_table(paired, ["category"]),
        "store_metrics": metric_table(paired, ["store"]),
        "sales_state_metrics": metric_table(paired, ["sales_state"]),
        "sales_state_decomposition": zeros,
        "category_decomposition": decomposition(paired, "category"),
        "series_metrics": series.reset_index(),
        "largest_regressions": series.nlargest(10, "net_extra_absolute_error").reset_index(),
        "largest_improvements": series.nsmallest(10, "net_extra_absolute_error").reset_index(),
    }
    return summary, tables, paired


def save_results(summary, tables, output):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)


def plot_diagnostics(tables):
    import matplotlib.pyplot as plt
    colors = {"lightgbm": "#2563A6", "mean28": "#B7791F", "seasonal_naive": "#718096"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    figures = {}
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    for model in ["mean28", "lightgbm", "seasonal_naive"]:
        rows = tables["horizon_metrics"].query("model == @model").sort_values("horizon")
        ax.plot(rows.horizon, rows.mae, label=model, color=colors[model], linewidth=1.8,
                linestyle="--" if model == "mean28" else "-")
    ax.set(xlabel="Forecast horizon (days)", ylabel="MAE (units)",
           title="Validation error by forecast horizon", xlim=(1, 28), ylim=(0, None))
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.16)
    figures["horizon_mae"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    states = ["Zero sales", "Positive sales"]
    for ax, state in zip(axes, states):
        rows = tables["sales_state_metrics"].query("sales_state == @state").set_index("model")
        names = ["mean28", "lightgbm"]
        values = rows.loc[names, "mae"]
        bars = ax.bar(names, values, color=[colors[n] for n in names], width=0.56)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set(title=f"{state} (n={int(rows.iloc[0].n_observations):,})",
               ylabel="MAE (units)", ylim=(0, 1.85))
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    figures["sales_state_mae"] = fig

    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    bridge = tables["sales_state_decomposition"].set_index("sales_state")
    names = ["Zero sales", "Positive sales", "Net difference"]
    values = [float(bridge.loc[name, "net_extra_absolute_error"]) for name in names[:2]]
    values.append(sum(values))
    bars = ax.barh(names, values, color=["#B7791F", "#2563A6", "#718096"], height=0.55)
    for bar, value in zip(bars, values):
        ax.text(value + (18 if value >= 0 else -18), bar.get_y() + bar.get_height()/2,
                f"{value:+,.1f}", va="center", ha="left" if value >= 0 else "right")
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set(xlabel="Extra absolute error: LightGBM minus mean28 (units)",
           title="Signed error contributions on the same validation rows", xlim=(-1060, 1220))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    figures["error_contributions"] = fig
    return figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="reports/v01_diagnostics")
    args = parser.parse_args()
    summary, tables, _ = analyze(args.predictions)
    save_results(summary, tables, args.output)
    for name, figure in plot_diagnostics(tables).items():
        figure.savefig(Path(args.output) / f"{name}.png", dpi=170)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

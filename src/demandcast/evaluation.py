"""Shared date cutoffs, an explicit final holdout, and transparent metrics."""

import numpy as np
import pandas as pd

from demandcast.models import make_model


def cutoff_indices(panel, config, stage="validation"):
    if stage not in {"validation", "holdout"}:
        raise ValueError("Stage must be validation or holdout")
    final_cutoff = len(panel.dates) - config.horizon - 1
    cutoffs = [final_cutoff] if stage == "holdout" else [
        final_cutoff - index * config.horizon
        for index in range(config.n_windows, 0, -1)
    ]
    if min(cutoffs) < 28 + config.horizon:
        raise ValueError("Not enough history for these evaluation windows and training horizon")
    return cutoffs


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Metrics require equally shaped, nonempty observations")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics do not accept nonfinite values")
    error = predicted - actual
    denominator = np.abs(actual).sum()
    return {"mae": float(np.abs(error).mean()),
            "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
            "bias": float(error.mean()), "n_observations": int(actual.size)}


def summarize(predictions, group_columns):
    records = []
    for key, group in predictions.groupby(group_columns, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        records.append({**dict(zip(group_columns, key)),
                        **metrics(group["y"], group["prediction"])})
    return pd.DataFrame(records)


def run_backtest(panel, config, stage="validation", factory=make_model):
    results = []
    for cutoff in cutoff_indices(panel, config, stage):
        history = panel.prefix(cutoff)
        target_dates = panel.dates[cutoff + 1:cutoff + config.horizon + 1]
        actual = panel.values[:, cutoff + 1:cutoff + config.horizon + 1].reshape(-1)
        for name in config.models:
            model = factory(name, config).fit(history)
            predicted = model.predict(config.horizon)
            if predicted.shape != (len(panel.ids), config.horizon):
                raise ValueError("Forecast shape does not match the evaluation panel")
            results.append(pd.DataFrame({
                "unique_id": np.repeat(panel.ids, config.horizon),
                "ds": np.tile(target_dates.to_numpy(), len(panel.ids)),
                "cutoff": history.dates[-1], "stage": stage, "model": name,
                "horizon": np.tile(np.arange(1, config.horizon + 1), len(panel.ids)),
                "y": actual, "prediction": predicted.reshape(-1),
            }))
    return pd.concat(results, ignore_index=True)


def forecast_latest(panel, config, name):
    model = make_model(name, config).fit(panel)
    dates = pd.date_range(panel.dates[-1] + pd.Timedelta(days=1),
                          periods=config.horizon, freq="D")
    predictions = model.predict(config.horizon)
    return pd.DataFrame({
        "unique_id": np.repeat(panel.ids, config.horizon),
        "ds": np.tile(dates.to_numpy(), len(panel.ids)), "model": name,
        "cutoff": panel.dates[-1],
        "horizon": np.tile(np.arange(1, config.horizon + 1), len(panel.ids)),
        "prediction": predictions.reshape(-1),
    })

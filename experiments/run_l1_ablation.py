"""A single-objective ablation using DemandCast v0.1's feature and split code."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd


class L1DirectTree:
    def __init__(self, config, feature_class):
        self.config = config
        self.feature_class = feature_class

    def fit(self, panel):
        from lightgbm import LGBMRegressor
        self.features = self.feature_class(panel)
        x, y, self.training_origins = self.features.training_matrix(
            self.config.horizon, self.config.training_origin_days, self.config.origin_stride)
        self.constant = float(y[0]) if np.all(y == y[0]) else None
        if self.constant is not None:
            return self
        # Identical settings to the shipped v0.1 LightGBM except objective.
        self.estimator = LGBMRegressor(
            objective="regression_l1", learning_rate=0.05,
            n_estimators=self.config.iterations, num_leaves=31,
            n_jobs=self.config.threads, random_state=self.config.seed,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
        self.estimator.fit(pd.DataFrame(x, columns=self.features.names), y,
                           categorical_feature=self.features.categorical_indices,
                           feature_name=self.features.names)
        return self

    def predict(self, horizon):
        if horizon != self.config.horizon:
            raise ValueError("Forecast horizon must equal the trained horizon")
        count = len(self.features.panel.ids)
        if self.constant is not None:
            return np.full((count, horizon), self.constant)
        x = self.features.at_origin(len(self.features.panel.dates)-1, horizon)
        values = self.estimator.predict(pd.DataFrame(x, columns=self.features.names))
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite L1 forecast")
        return np.maximum(values, 0).reshape(count, horizon)


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_reference(reference, base_config, panel, source_hash):
    if reference.get("stage") != "validation":
        raise ValueError("Reference must be a validation run")
    if reference.get("source_sha256") != source_hash:
        raise ValueError("Data differs from the frozen v0.1 input; stop this ablation")
    for name, value in reference["config"].items():
        actual = getattr(base_config, name)
        if name == "models":
            actual = list(actual)
        if actual != value:
            raise ValueError(f"Configuration changed: {name}. Keep all settings fixed")
    if len(panel.ids) != reference["n_series"] or len(panel.dates) != reference["observed_days"]:
        raise ValueError("Observed panel differs from the frozen reference")


def main(argv=None):
    parser = argparse.ArgumentParser(description="DemandCast: Poisson vs L1, validation only")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/m5.yaml")
    parser.add_argument("--output", default="outputs/l1-validation")
    parser.add_argument("--reference-run", default="reports/v01_diagnostics/reference_run.json")
    parser.add_argument("--reference-metrics", default="reports/v01_diagnostics/reference_metrics.csv")
    args = parser.parse_args(argv)
    try:
        from demandcast.config import load_config
        from demandcast.data import to_panel
        from demandcast.evaluation import run_backtest, summarize
        from demandcast.features import OriginFeatures
        from demandcast.models import make_model
    except ImportError as exc:
        parser.exit(2, 'Activate the existing DemandCast v0.1 environment and install -e ".[ml]".\n'
                    f"Import error: {exc}\n")
    base = load_config(args.config)
    source_hash = hash_file(args.data)
    panel = to_panel(pd.read_csv(args.data, dtype={"unique_id": str}))
    reference = json.loads(Path(args.reference_run).read_text(encoding="utf-8"))
    check_reference(reference, base, panel, source_hash)
    config_payload = base.to_dict()
    config_payload["models"] = list(base.models) + ["lightgbm_l1"]
    config = SimpleNamespace(**config_payload)

    def factory(name, settings):
        return L1DirectTree(settings, OriginFeatures) if name == "lightgbm_l1" else make_model(name, settings)

    started = time.perf_counter()
    predictions = run_backtest(panel, config, stage="validation", factory=factory)
    fold_metrics = summarize(predictions, ["model", "cutoff"])
    expected = pd.read_csv(args.reference_metrics, parse_dates=["cutoff"])
    expected["cutoff"] = pd.to_datetime(expected.cutoff)
    actual = fold_metrics[fold_metrics.model != "lightgbm_l1"].copy()
    actual["cutoff"] = pd.to_datetime(actual.cutoff)
    compared = expected.merge(actual, on=["model", "cutoff"], suffixes=("_expected", "_actual"),
                              how="outer", indicator=True, validate="one_to_one")
    reproduced = compared._merge.eq("both").all() and all(
        np.allclose(compared[f"{metric}_expected"], compared[f"{metric}_actual"], rtol=1e-6, atol=1e-7)
        for metric in ["mae", "wape", "bias", "n_observations"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    fold_metrics.to_csv(output / "metrics.csv", index=False)
    summarize(predictions, ["model"]).to_csv(output / "overall_metrics.csv", index=False)
    summarize(predictions, ["model", "horizon"]).to_csv(output / "horizon_metrics.csv", index=False)
    summarize(predictions, ["model", "unique_id"]).to_csv(output / "series_metrics.csv", index=False)
    segments = predictions.assign(sales_state=np.where(predictions.y == 0, "zero", "positive"))
    summarize(segments, ["model", "sales_state"]).to_csv(output / "sales_state_metrics.csv", index=False)
    compared.to_csv(output / "control_reproduction.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "validation",
        "experiment": "only LightGBM objective changes: poisson -> regression_l1",
        "source_sha256": source_hash, "reference_run_sha256": hash_file(args.reference_run),
        "config": config_payload, "n_series": len(panel.ids), "observed_days": len(panel.dates),
        "control_reproduced_within_tolerance": bool(reproduced),
        "reproduction_tolerance": {"relative": 1e-6, "absolute": 1e-7},
        "lightgbm_version": importlib.metadata.version("lightgbm"),
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "holdout_evaluated": False,
        "guardrails": "Inspect signed bias and positive-sales MAE alongside pooled MAE/WAPE; no business bias threshold has been established",
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summarize(predictions, ["model"]).to_string(index=False))
    print(f"Output: {output}")
    if not reproduced:
        parser.exit(3, "Control results differ from v0.1. Investigate before accepting any improvement.\n")
    print("Frozen control reproduced. Holdout was not evaluated.")


if __name__ == "__main__":
    main()

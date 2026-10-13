"""Explicit commands; validation never automatically evaluates the final holdout."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import pandas as pd

from demandcast import __version__
from demandcast.config import MODEL_NAMES, load_config
from demandcast.data import prepare_m5, sha256_file, synthetic_sales, to_panel
from demandcast.evaluation import forecast_latest, run_backtest, summarize


def environment_versions():
    versions = {"python": platform.python_version(), "demandcast": __version__}
    for name in ["numpy", "pandas", "PyYAML", "scikit-learn", "lightgbm", "threadpoolctl"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def save_run(panel, config, output, stage, dataset_kind, source_sha256=None, provenance=None):
    started = time.perf_counter()
    predictions = run_backtest(panel, config, stage)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination / "predictions.csv", index=False)
    summaries = {
        "metrics.csv": ["model", "cutoff"],
        "horizon_metrics.csv": ["model", "horizon"],
        "series_metrics.csv": ["model", "unique_id"],
    }
    for filename, groups in summaries.items():
        summarize(predictions, groups).to_csv(destination / filename, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_kind": dataset_kind, "stage": stage,
        "source_sha256": source_sha256, "dataset_provenance": provenance,
        "config": config.to_dict(), "environment": environment_versions(),
        "n_series": len(panel.ids), "observed_days": len(panel.dates),
        "cutoffs": sorted(str(value.date()) for value in predictions["cutoff"].unique()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "wape_definition": "sum(abs(prediction-y))/sum(abs(y)); fraction, undefined at zero denominator",
        "metric_scope": "sampled bottom-level series; not official M5 WRMSSE",
        "synthetic_results_are_not_benchmark_evidence": dataset_kind == "synthetic",
    }
    (destination / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summarize(predictions, ["model"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="DemandCast v0.1 forecasting foundation")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a clearly labeled synthetic smoke test")
    demo.add_argument("--config", default="configs/demo.yaml")
    demo.add_argument("--output", default="outputs/demo")
    for name in ["backtest", "forecast"]:
        command = commands.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--config", default="configs/m5.yaml")
        command.add_argument("--output", default=f"outputs/{name}")
        if name == "backtest":
            command.add_argument("--stage", choices=["validation", "holdout"], default="validation")
        else:
            command.add_argument("--model", choices=sorted(MODEL_NAMES), default="lightgbm")
    prepare = commands.add_parser("prepare-m5", help="Sample before expanding M5 to long format")
    prepare.add_argument("--input-dir", required=True)
    prepare.add_argument("--config", default="configs/m5.yaml")
    prepare.add_argument("--output", default="data/processed/m5.csv")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "prepare-m5":
            manifest = prepare_m5(args.input_dir, args.output, config)
            print(f"Prepared {manifest['n_series']} real M5 series, {manifest['history_days']} days")
            print(f"Dataset: {args.output}")
        elif args.command == "demo":
            panel = to_panel(synthetic_sales(config.n_series, config.history_days, config.seed))
            print("SYNTHETIC DEMO — these values are not M5 benchmark evidence")
            print(save_run(panel, config, args.output, "validation", "synthetic").to_string(index=False))
            print(f"Results: {args.output}")
        else:
            panel = to_panel(pd.read_csv(args.data, dtype={"unique_id": str}))
            if args.command == "forecast":
                destination = Path(args.output)
                destination.mkdir(parents=True, exist_ok=True)
                forecast_latest(panel, config, args.model).to_csv(destination / "forecast.csv", index=False)
                print(f"Future forecasts: {destination / 'forecast.csv'}")
            else:
                source_hash = sha256_file(args.data)
                sidecar = Path(args.data).with_suffix(".manifest.json")
                provenance = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
                if provenance and provenance.get("prepared_sha256") != source_hash:
                    raise ValueError("Prepared dataset no longer matches its provenance manifest")
                dataset_kind = provenance.get("dataset_kind", "user_supplied") if provenance else "user_supplied"
                print(save_run(panel, config, args.output, args.stage, dataset_kind,
                               source_hash, provenance).to_string(index=False))
                print(f"Results: {args.output}")
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

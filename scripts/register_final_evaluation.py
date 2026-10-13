"""Register an outcome-uninspected product cohort before its forecasts are run."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from project_preflight import hash_file, inspect_project

ROOT = Path(__file__).resolve().parents[1]
SELECTION_SEED = 20261013
N_ITEMS = 50


def normal_id(value):
    value = str(value)
    for suffix in ["_evaluation", "_validation"]:
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value


def choose_cohort(catalogue, known_series, n_items=N_ITEMS, seed=SELECTION_SEED):
    columns = ["id", "item_id", "store_id"]
    if set(columns)-set(catalogue.columns) or catalogue[columns].isna().any().any():
        raise ValueError("Catalogue must contain complete id, item_id and store_id metadata")
    metadata = catalogue[columns].copy()
    metadata["normal_id"] = metadata.id.map(normal_id)
    if metadata.normal_id.duplicated().any() or metadata.duplicated(["item_id", "store_id"]).any():
        raise ValueError("Duplicate catalogue series or item/store combinations")
    known = {normal_id(value) for value in known_series}
    if not known or not known.issubset(set(metadata.normal_id)):
        raise ValueError("Development series cannot be matched to the original M5 catalogue")
    excluded_items = set(metadata.loc[metadata.normal_id.isin(known), "item_id"])
    candidates = set(metadata.item_id)-excluded_items
    if len(candidates) < n_items or n_items < 1:
        raise ValueError("Insufficient products outside the development cohort")
    rank = lambda item: (hashlib.sha256(f"{seed}|{item}".encode()).hexdigest(), str(item))
    selected_items = sorted(sorted(candidates, key=rank)[:n_items])
    selected = metadata[metadata.item_id.isin(selected_items)].sort_values(["item_id", "store_id"])
    stores = set(metadata.store_id)
    if any(set(part.store_id) != stores for _, part in selected.groupby("item_id")):
        raise ValueError("Selected products do not each contain every catalogue store")
    if excluded_items.intersection(selected_items) or known.intersection(selected.normal_id):
        raise ValueError("Development and evaluation cohorts overlap")
    return {"selection_seed": seed, "selected_item_ids": selected_items,
            "selected_raw_series_ids": selected.id.tolist(), "stores": sorted(stores),
            "n_items": len(selected_items), "n_series": len(selected),
            "excluded_development_items": len(excluded_items),
            "n_development_series": len(known), "item_overlap_count": 0, "series_overlap_count": 0}


def register(root, catalogue_path, output):
    root, catalogue_path, output = map(Path, [root, catalogue_path, output])
    if output.exists():
        raise ValueError("A registration already exists. Preserve it and review its status before any new registration")
    report = inspect_project(root, "local")
    if not report["ready_for_local_review"]:
        raise ValueError("Project preflight has unresolved errors; run project_preflight.py first")
    if catalogue_path.name != "sales_train_evaluation.csv":
        raise ValueError("This dated protocol requires the original sales_train_evaluation.csv catalogue")
    panel = root/"data/processed/m5.csv"
    known = pd.read_csv(panel, usecols=["unique_id"], dtype=str).unique_id.unique()
    # Target sales columns are deliberately not deserialized for registration.
    catalogue = pd.read_csv(catalogue_path, usecols=["id", "item_id", "store_id"], dtype=str)
    cohort = choose_cohort(catalogue, known)
    if len(cohort["stores"]) != 10 or cohort["n_series"] != 500:
        raise ValueError("The planned M5 test expects 50 fresh products across all 10 stores")
    reference = json.loads((root/"reports/dashboard_reference/intervals/run.json").read_text())
    from demandcast.config import load_config
    config = load_config(root/"configs/m5.yaml").to_dict()
    for name, value in reference["config"].items():
        if name != "models" and config.get(name) != value:
            raise ValueError(f"Development configuration changed: {name}")
    if list(config["models"]) != ["seasonal_naive", "mean28", "lightgbm"]:
        raise ValueError("The original point-control model list changed")
    payload = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "registered_not_evaluated", "protocol": "selection-disjoint product cohort, one final time window",
               "previous_v01_holdout_already_inspected": True,
               "target_outcomes_deserialized_during_registration": False,
               "development_panel_sha256": report["prepared_data"]["sha256"],
               "raw_catalogue_file_sha256": hash_file(catalogue_path),
               "recipe_config": reference["config"], "source_hashes": report["source_hashes"],
               "versions": report["versions"], "cohort": cohort,
               "forecast_cutoff": "2016-04-24", "target_start": "2016-04-25", "target_end": "2016-05-22",
               "history_days_in_prepared_panel": 730, "horizon": 28,
               "mean_model": "lightgbm", "median_model": "lightgbm_l1",
               "interval": "q10-q90 envelope including unchanged L1 median; no rolling expansion",
               "fit_policy": "retrain the frozen recipe on only the new cohort's history through cutoff; not zero-shot transfer",
               "primary_metrics": ["median MAE", "mean WAPE and aggregate volume bias", "interval_score_80"],
               "secondary_metrics": ["inclusive interval coverage", "mean interval width", "both tail miss rates"],
               "comparison_policy": "Compare controls on this same cohort; old/new cohort absolute metrics are not directly comparable",
               "limitations": ["Shared calendar/store shocks remain; not an independent dataset or iid test",
                               "No statistical-significance or inventory service-level guarantee",
                               "No resampling or tuning after evaluation outcomes are inspected"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(f"Registered {cohort['n_items']} fresh products and {cohort['n_series']} series. No forecast evaluated.")
    print(f"Registration: {output}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--catalogue", type=Path, default=Path("data/raw/m5/sales_train_evaluation.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/final_evaluation/registration.json"))
    args = parser.parse_args(argv)
    catalogue = args.catalogue if args.catalogue.is_absolute() else args.root/args.catalogue
    output = args.output if args.output.is_absolute() else args.root/args.output
    try:
        register(args.root, catalogue, output)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"Registration stopped: {exc}\n")


if __name__ == "__main__":
    main()

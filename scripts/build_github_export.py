"""Build a new, byte-preserving publication snapshot from the actual checkout.

Reads the completed final run; does not train, resample, change the source tree,
initialize Git or send anything to GitHub.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

from project_preflight import hash_file, inspect_project

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "demandcast", "experiments", "dashboard", "scripts", "tests", "configs",
               "analysis", "notebooks", "docs", "examples", "assets", ".github", "release_templates"}
ROOT_FILES = {"README.md", "README_GITHUB.md", "pyproject.toml", ".gitignore", "setup.py", "setup.cfg", "MANIFEST.in",
              "Makefile", "Dockerfile", "docker-compose.yml", "uv.lock", "poetry.lock",
              "environment.yml", "CONTRIBUTING.md", "NOTICE", ".env.example"}
SKIP_PARTS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip", ".pkl", ".pickle", ".joblib"}
PUBLIC_FILES = ["point_overall_metrics.csv", "interval_metrics.csv", "comparisons.csv",
                "point_by_horizon.csv", "interval_by_horizon.csv", "point_by_store.csv", "interval_by_store.csv",
                "point_by_sales_state.csv", "interval_by_sales_state.csv", "interval_by_band.csv",
                "interval_by_predicted_mean.csv", "SUMMARY.md", "registration_snapshot.json"]
LABELS = {"seasonal_naive": "Seasonal naive", "mean28": "28-day mean",
          "lightgbm": "LightGBM Poisson (mean)", "lightgbm_l1": "LightGBM L1 (median)"}
ATTRIBUTES = "# Preserve bytes bound by experiment receipts and source checksums.\n* -text\n"
INSTALL = "python -m pip install -r requirements-release.txt -e ."
PUBLIC_INSTALL = "python -m pip install -r requirements-release.txt -r requirements-publication.txt -e ."


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_file(directory, name):
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid receipt filename: {name}")
    path = directory/name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or linked result file: {name}")
    return path


def is_install_metadata(path):
    return any(part.lower().endswith((".egg-info", ".dist-info")) for part in Path(path).parts)


def snapshot_hashes(root):
    root = Path(root)
    return {path.relative_to(root).as_posix(): hash_file(path)
            for path in sorted(root.rglob("*")) if path.is_file()
            and not is_install_metadata(path.relative_to(root))
            and not any(part in SKIP_PARTS for part in path.relative_to(root).parts)
            and path.suffix.lower() not in SKIP_SUFFIXES}


def load_evidence(output):
    output = Path(output)
    run = read_json(output/"run.json")
    forecast = read_json(output/"forecast_run.json")
    if (run["status"], run["stage"], run["n_series"], run["n_items"], run["n_observations_per_model"]) != (
            "scored", "final_product_cohort", 500, 50, 14000):
        raise ValueError("A completed 50-product/500-series final run is required")
    if (run["forecast_cutoff"], run["target_start"], run["target_end"]) != (
            "2016-04-24", "2016-04-25", "2016-05-22"):
        raise ValueError("Final dates differ from the registered protocol")
    if forecast["status"] != "forecasts_frozen" or forecast["bindings"] != run["bindings"]:
        raise ValueError("Final/forecast receipts do not reconcile")
    if hash_file(output/"forecasts.csv") != forecast["forecasts_sha256"]:
        raise ValueError("Frozen forecast bytes changed")
    hashes = run["artifact_hashes"]
    if not set(PUBLIC_FILES+["interval_predictions.csv"]).issubset(hashes):
        raise ValueError("Final receipt is missing expected result files")
    for name, expected in hashes.items():
        if hash_file(safe_file(output, name)) != expected:
            raise ValueError(f"Final artifact changed: {name}")
    registration = read_json(output/"registration_snapshot.json")
    if (hash_file(output/"registration_snapshot.json") != run["bindings"]["registration_sha256"]
            or registration["cohort"] != run["cohort"] or registration["recipe_config"] != run["recipe_config"]
            or not run["target_sales_deserialized"]):
        raise ValueError("Final run does not match its original cohort registration")
    with (output/"point_overall_metrics.csv").open(encoding="utf-8", newline="") as handle:
        raw_points = list(csv.DictReader(handle))
    points = {}
    for row in raw_points:
        model = row.pop("model")
        if model in points:
            raise ValueError("Duplicate point models")
        points[model] = {key: float(value) if value else None for key, value in row.items()}
    if set(points) != set(LABELS):
        raise ValueError("Final point models differ from the frozen controls")
    totals = set()
    for row in points.values():
        count, actual, predicted = row["n_observations"], row["actual_sales_total"], row["predicted_sales_total"]
        if count != 14000 or actual < 0 or predicted < 0 or row["mae"] < 0 or not all(
                math.isfinite(row[key]) for key in ["mae", "bias", "actual_sales_total", "predicted_sales_total"]):
            raise ValueError("Invalid point counts/totals")
        totals.add(actual)
        expected = {"bias": (predicted-actual)/count,
                    "wape": row["mae"]*count/actual if actual else None,
                    "aggregate_volume_bias_fraction": (predicted-actual)/actual if actual else None}
        for key, value in expected.items():
            if ((value is None and row[key] is not None) or
                    (value is not None and (row[key] is None or not math.isclose(row[key], value, rel_tol=1e-8, abs_tol=1e-8)))):
                raise ValueError(f"Point summary does not reconcile: {key}")
    if len(totals) != 1:
        raise ValueError("Point models used different actual-sales totals")
    with (output/"interval_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or rows[0]["method"] != "uncalibrated_envelope":
        raise ValueError("Expected a single frozen-envelope interval summary")
    interval = {key: float(value) for key, value in rows[0].items() if key != "method"}
    if (not all(math.isfinite(value) for value in interval.values()) or interval["n_observations"] != 14000
            or interval["nominal_target_coverage"] != .8 or interval["mean_width"] < 0
            or interval["interval_score_80"] < interval["mean_width"]
            or not all(0 <= interval[key] <= 1 for key in ["coverage", "lower_miss_rate", "upper_miss_rate"])
            or not math.isclose(sum(interval[key] for key in ["coverage", "lower_miss_rate", "upper_miss_rate"]), 1, abs_tol=1e-10)):
        raise ValueError("Interval population/rates/score do not reconcile")
    return run, points, interval


def public_files(root):
    root = Path(root)
    selected = []
    candidates = [path for path in root.iterdir() if path.is_file()]
    trees = [root/name for name in DIRECTORIES]
    if (root/"reports").is_dir():
        trees += [path for path in (root/"reports").iterdir() if path.is_dir() and
                  path.name != "final_cohort_reference" and (path.name.endswith("_reference") or path.name == "v01_diagnostics")]
    trees += [root/"data/examples", root/"data/synthetic"]
    for tree in trees:
        if tree.is_symlink():
            raise ValueError(f"Publication directory is a link: {tree.name}")
        if tree.is_dir():
            candidates.extend(tree.rglob("*"))
    for path in sorted(set(candidates)):
        relative = path.relative_to(root)
        parts = relative.parts
        if (not path.is_file() or is_install_metadata(relative)
                or any(part in SKIP_PARTS for part in parts) or path.suffix.lower() in SKIP_SUFFIXES):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        allowed = (len(parts) == 1 and (path.name in ROOT_FILES or path.name.startswith(("requirements", "LICENSE", "LICENCE"))))
        allowed |= parts[0] in DIRECTORIES and not any(part.startswith(".") and part != ".github" for part in parts)
        allowed |= (len(parts) > 2 and parts[0] == "reports" and parts[1] != "final_cohort_reference"
                    and (parts[1].endswith("_reference") or parts[1] == "v01_diagnostics"))
        allowed |= len(parts) > 2 and parts[:2] in [("data", "examples"), ("data", "synthetic")]
        if allowed:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Publication input is a link or outside the checkout: {relative}")
            if path.stat().st_size > 50*1024*1024:
                raise ValueError(f"Publication file exceeds this snapshot's 50 MiB budget: {relative}")
            selected.append(path)
    return selected


def check_checkout(report, run, config):
    if report["source_hashes"] != run["bindings"]["source_hashes"]:
        raise ValueError("Forecast source code changed after the final run")
    if list(config["models"]) != ["seasonal_naive", "mean28", "lightgbm"] or any(
            config.get(key) != value for key, value in run["recipe_config"].items() if key != "models"):
        raise ValueError("Source configuration changed after the final run")


def formatted(value, percentage=False):
    if value is None:
        return "undefined (zero sales)"
    return f"{100*value:.2f}%" if percentage else f"{value:.4f}"


def final_section(points, interval):
    lines = ["## Final product-cohort evaluation", "", "The frozen recipe was retrained on 50 products outside the development sample,",
             "across all 10 stores. It uses 702 observed days through 2016-04-24 and",
             "14,000 target observations over 2016-04-25 to 2016-05-22.", "",
             "| Model | MAE | WAPE | Aggregate predicted-sales bias |", "| --- | ---: | ---: | ---: |"]
    for name in LABELS:
        row = points[name]
        lines.append(f"| {LABELS[name]} | {formatted(row['mae'])} | {formatted(row['wape'], True)} | {formatted(row['aggregate_volume_bias_fraction'], True)} |")
    lines += ["", "![Final-cohort point models](assets/final-cohort-point-models.png)", "",
              "| Interval measure | Value |", "| --- | ---: |",
              f"| Inclusive envelope coverage | {formatted(interval['coverage'], True)} |",
              f"| Mean width | {formatted(interval['mean_width'])} sales units |",
              f"| 80% interval score | {formatted(interval['interval_score_80'])} |",
              f"| Below lower / above upper | {formatted(interval['lower_miss_rate'], True)} / {formatted(interval['upper_miss_rate'], True)} |", "",
              "The interval is the frozen q10/L1/q90 envelope without temporal expansion.",
              "Compare models on these same targets. Development and final-cohort absolute",
              "scores use different products and are not a paired before/after improvement.",
              "L1 remains median-oriented and Poisson remains mean-oriented; volume bias is",
              "shown separately from absolute error.", "",
              "This is a product-disjoint recipe check with shared calendar/store conditions,",
              "not an independent dataset, zero-shot transfer or a significance claim.",
              "The final outcomes have now been inspected; further tuning needs a new plan.", "",
              "Sources and original receipts: [final reference](reports/final_cohort_reference).",
              "The [registered protocol](docs/release/FINAL_EVALUATION_PROTOCOL.md) and",
              "[runner contract](docs/release/FINAL_EVALUATION_RUNNER.md) explain the dates and guards.", "",
              "To reproduce on separately obtained M5 inputs, register before scoring:", "", "```bash",
              "python scripts/register_final_evaluation.py",
              "python experiments/run_final_evaluation.py --phase forecast",
              "python experiments/run_final_evaluation.py --phase score", "```", ""]
    return "\n".join(lines)


def updated_readme(text, points, interval):
    start, end = "## Final evaluation", "## Repository layout"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError("README sections changed; preserve your edits and review the publication template")
    before = text[:text.index(start)].replace("## Results", "## Development validation", 1)
    old = ("The current results are\ndevelopment validation; an additional outcome-uninspected product cohort is\n"
           "registered separately before its final evaluation.")
    before = before.replace(old, "Development validation and a separately preregistered product-cohort evaluation\nare reported separately below.")
    return before+final_section(points, interval)+"\n"+text[text.index(end):]


def publication_workflow(text):
    if PUBLIC_INSTALL in text:
        return text
    if text.count(INSTALL) != 1:
        raise ValueError("CI install command was customized; review the publication dependencies before exporting")
    return text.replace(INSTALL, PUBLIC_INSTALL)


def render_chart(points, path, *, preview=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(LABELS.values())
    colors = ["#a4adb9", "#768497", "#2877a8", "#9156a6"]
    maes = [points[key]["mae"] for key in LABELS]
    biases = [points[key]["aggregate_volume_bias_fraction"] for key in LABELS]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), layout="constrained")
        title = ("SYNTHETIC LAYOUT CHECK\nIllustrative values; no M5 performance claim" if preview else
                 "Final product cohort: frozen recipe\n50 products · 10 stores · 14,000 target observations")
        fig.suptitle(title, fontsize=13)
        axes[0].barh(labels, maes, color=colors)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("MAE (daily sales units)")
        axes[0].set_title("Absolute-error accuracy")
        axes[0].set_xlim(0, max(max(maes)*1.3, .1))
        for i, value in enumerate(maes):
            axes[0].annotate(f"{value:.4f}", (value, i), xytext=(4, 0), textcoords="offset points", va="center")
        if all(value is not None for value in biases):
            percentages = [100*value for value in biases]
            axes[1].barh(labels, percentages, color=colors)
            axes[1].invert_yaxis()
            axes[1].axvline(0, color="#263445", linewidth=.8)
            limit = max(max(abs(value) for value in percentages)*1.9, 1)
            axes[1].set_xlim(-limit, limit)
            for i, value in enumerate(percentages):
                label = "0.00%" if abs(value) < .005 else f"{value:+.2f}%"
                axes[1].annotate(label, (value, i), xytext=(4 if value >= 0 else -4, 0),
                                 textcoords="offset points", va="center", ha="left" if value >= 0 else "right")
        else:
            axes[1].text(.5, .5, "Volume bias undefined:\nactual sales total is zero", transform=axes[1].transAxes, ha="center")
        axes[1].set_title("Aggregate sales bias (descriptive)")
        axes[1].set_xlabel("Predicted-volume bias (%)")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170, facecolor="white")
        plt.close(fig)


def verify_export(root):
    root = Path(root)
    manifest = read_json(root/"docs/release/GITHUB_EXPORT_MANIFEST.json")
    for name, wanted in manifest["files"].items():
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or hash_file(path) != wanted:
            raise ValueError(f"Export file changed: {name}")
    print(f"Export verified: {len(manifest['files'])} files; frozen bytes preserved.")


def build(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("Choose a new destination outside the source checkout; nothing is overwritten")
    report = inspect_project(root, "local")
    if not report["ready_for_local_review"]:
        raise ValueError("Source preflight failed: "+"; ".join(report["errors"]))
    output = root/"outputs/final-cohort"
    run, points, interval = load_evidence(output)
    from demandcast.config import load_config
    check_checkout(report, run, load_config(root/"configs/m5.yaml").to_dict())
    attempt = read_json(root/"reports/final_evaluation/final_evaluation_attempt.json")
    if (attempt["status"] != "scored" or attempt["final_run_sha256"] != hash_file(output/"run.json")
            or (root/attempt["output_directory"]).resolve() != output):
        raise ValueError("The source attempt record does not match the completed final run")
    readme = updated_readme((root/"README.md").read_text(encoding="utf-8"), points, interval)
    workflow = publication_workflow((root/".github/workflows/ci.yml").read_text(encoding="utf-8"))
    template = publication_workflow((root/"release_templates/ci.yml").read_text(encoding="utf-8"))
    files = public_files(root)
    # Check the renderer dependency before creating any destination files.
    import matplotlib
    destination.mkdir(parents=True, exist_ok=False)
    for path in files:
        target = destination/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    reference = destination/"reports/final_cohort_reference"
    reference.mkdir(parents=True, exist_ok=False)
    for name in PUBLIC_FILES+["run.json", "forecast_run.json"]:
        shutil.copyfile(output/name, reference/name)
    (reference/"README.md").write_text(
        "# Final product-cohort reference\n\nOriginal completed-run receipts and aggregate results.\n"
        "The original run receipt also records hashes for local row-level files, which\n"
        "are deliberately omitted here. Raw M5 sales, interval_predictions.csv and\n"
        "forecasts.csv are not included in this publication snapshot. The exporter\n"
        "verified their original hashes before copying these summaries. This is an\n"
        "integrity check, not an independent replication of the experiment.\n", encoding="utf-8")
    (destination/"README.md").write_text(readme, encoding="utf-8", newline="\n")
    (destination/"README_GITHUB.md").write_text(readme, encoding="utf-8", newline="\n")
    (destination/".github/workflows/ci.yml").write_text(workflow, encoding="utf-8", newline="\n")
    (destination/"release_templates/ci.yml").write_text(template, encoding="utf-8", newline="\n")
    (destination/".gitattributes").write_text(ATTRIBUTES, encoding="utf-8", newline="\n")
    render_chart(points, destination/"assets/final-cohort-point-models.png")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_checkout_name": root.name,
                "source_git_history_copied": False, "final_run_sha256": hash_file(output/"run.json"),
                "matplotlib_version": matplotlib.__version__, "remote_ci_verified": False,
                "files": snapshot_hashes(destination)}
    (destination/"docs/release/GITHUB_EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_export(destination)
    print(f"Publication snapshot: {destination}")
    print("Original project, data, final outputs and Git history remain in the source checkout.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-export", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify_export:
            verify_export(ROOT)
        else:
            destination = args.destination or ROOT.parent/(ROOT.name+"-github")
            build(ROOT, destination)
    except (ValueError, OSError, KeyError, ImportError, TypeError) as exc:
        parser.exit(2, f"Publication preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()

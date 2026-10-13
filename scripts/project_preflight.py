"""Check the assembled checkout; no training, publication or credential reads."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["pyproject.toml", "configs/m5.yaml", "README.md", ".github/workflows/ci.yml",
            "requirements-release.txt", "experiments/run_l1_ablation.py",
            "experiments/run_interval_benchmark.py", "experiments/run_temporal_calibration.py",
            "dashboard/app.py", "dashboard/dashboard_data.py", "docs/MODEL_CARD_V02.md",
            "tests/test_l1_ablation.py", "tests/test_prediction_intervals.py",
            "tests/test_temporal_calibration.py", "tests/test_dashboard_addon.py"]
INTERFACES = {"demandcast.config": ["load_config"], "demandcast.data": ["to_panel"],
              "demandcast.features": ["OriginFeatures"], "demandcast.models": ["make_model"],
              "demandcast.evaluation": ["run_backtest", "summarize"]}


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hashes(root):
    root = Path(root).resolve()
    files = []
    for directory in [root/"src/demandcast", root/"demandcast", root/"experiments", root/"dashboard"]:
        if directory.is_dir():
            files.extend(directory.rglob("*.py"))
    return {str(path.relative_to(root)).replace("\\", "/"): hash_file(path)
            for path in sorted(set(files)) if "__pycache__" not in path.parts and path.resolve().is_relative_to(root)}


def inspect_project(root, mode="local"):
    root = Path(root).resolve()
    errors, notes = [], []
    files = {name: (root/name).is_file() for name in EXPECTED}
    errors.extend(f"Missing required file: {name}" for name, exists in files.items() if not exists)
    metadata = {}
    if files["pyproject.toml"]:
        project = tomllib.loads((root/"pyproject.toml").read_text(encoding="utf-8")).get("project", {})
        metadata = {"name": project.get("name"), "requires_python": project.get("requires-python"),
                    "console_scripts": project.get("scripts", {}),
                    "optional_dependency_groups": sorted(project.get("optional-dependencies", {}))}
    for path in [root, root/"src", root/"dashboard"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    interfaces = {}
    for name, required in INTERFACES.items():
        try:
            module = importlib.import_module(name)
            missing = [symbol for symbol in required if not hasattr(module, symbol)]
            file = Path(module.__file__).resolve()
            inside = file.is_relative_to(root)
            passed = not missing and inside
            interfaces[name] = {"passed": passed, "missing_symbols": missing, "inside_checkout": inside}
            if not passed:
                errors.append(f"Core interface check failed: {name}; install this checkout with pip install -e .")
        except (ImportError, OSError, TypeError, AttributeError) as exc:
            interfaces[name] = {"passed": False, "error": str(exc)}
            errors.append(f"Cannot import core interface: {name}")
    versions = {}
    for package in ["numpy", "pandas", "lightgbm", "scikit-learn", "streamlit", "plotly", "pytest"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            errors.append(f"Required dependency not installed: {package}")
    if sys.version_info < (3, 12):
        errors.append("The release tooling requires Python 3.12 or later")
    reference_ok, receipt = False, None
    try:
        from dashboard_data import load_bundle
        _, receipts, _ = load_bundle(root, "bundled")
        receipt = receipts["e03"]
        reference_ok = True
    except (ImportError, OSError, ValueError, KeyError) as exc:
        errors.append(f"Bundled reference check failed: {exc}")
    panel = root/"data/processed/m5.csv"
    data = {"checked": mode == "local", "exists": panel.is_file(), "sha256": None, "matches_reference": None}
    if mode == "local":
        if not panel.is_file():
            errors.append("Prepared data/processed/m5.csv is missing; use the original core preparation instructions")
        elif receipt:
            data["sha256"] = hash_file(panel)
            data["matches_reference"] = data["sha256"] == receipt["source_sha256"]
            if not data["matches_reference"]:
                errors.append("Prepared M5 data differs from the frozen development reference")
    else:
        notes.append("CI omits M5 data and model-training runs; it checks code, fixtures and bundled reference summaries")
    git = {"is_repository": False, "tracked_generated_files": [], "head": None}
    try:
        inside = subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
        git["is_repository"] = inside.returncode == 0 and inside.stdout.strip() == "true"
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        git["head"] = head.stdout.strip() if head.returncode == 0 else None
        if git["is_repository"]:
            tracked = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "."], capture_output=True, check=True).stdout.decode().split("\0")
            git["tracked_generated_files"] = [name for name in tracked if
                name.startswith(("data/raw/", "data/processed/", "outputs/")) or
                (name == ".env" or (name.startswith(".env.") and name != ".env.example"))]
            if git["tracked_generated_files"]:
                errors.append("Generated data, outputs or local environment files are already tracked by git; ignore rules do not untrack them")
    except (OSError, subprocess.SubprocessError):
        notes.append("Git metadata is unavailable")
    other_workflows = [p.name for p in (root/".github/workflows").glob("*.y*ml") if p.name != "ci.yml"]
    if other_workflows:
        notes.append("Additional workflows need review alongside the new test dependencies: "+", ".join(other_workflows))
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "ready_for_local_review": not errors, "project_metadata": metadata,
            "python": platform.python_version(), "platform": platform.system(),
            "versions": versions, "required_files": files, "core_interfaces": interfaces,
            "source_hashes": source_hashes(root), "bundled_reference_valid": reference_ok,
            "prepared_data": data, "git": git, "errors": errors, "notes": notes,
            "remote_ci_verified": False, "independent_final_evaluation_completed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=["local", "ci"], default="local")
    parser.add_argument("--output", type=Path, default=Path("reports/release/project-preflight.json"))
    args = parser.parse_args(argv)
    report = inspect_project(args.root, args.mode)
    output = args.output if args.output.is_absolute() else args.root/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Preflight: {'passed' if report['ready_for_local_review'] else 'needs attention'}")
    for error in report["errors"]:
        print(f"- {error}")
    print(f"Report: {output}")
    return 0 if report["ready_for_local_review"] else 1


if __name__ == "__main__":
    sys.exit(main())

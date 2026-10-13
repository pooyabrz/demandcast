"""Repair the first publication manifest without rehashing frozen source/results.

Only generated installation metadata is removed. The two publication-tool
updates must match the exact delivered before/after digests. No file is deleted.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath

from build_github_export import hash_file, is_install_metadata, verify_export

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "docs/release/GITHUB_EXPORT_MANIFEST.json"
BACKUP = "docs/release/GITHUB_EXPORT_MANIFEST.before_install_metadata_fix.json"
# Exact before/after digests for the delivered publication-tool fix.
TOOL_UPDATES = {
    "scripts/build_github_export.py": {
        "before": "7e5b8feaa8f62acdf4b86d322403af9592178c2da292fcc0b51695c859cbae3e",
        "after": "c2c540296452e3371198354dd5a3de0d78dd3572f4c75bca824997b9aa4711e5"},
    "tests/test_github_export.py": {
        "before": "7fe123a9a373c29fced8b921aca6b3fa89a2d11fe2a5c28bcd854392bb807b49",
        "after": "e226b4a570589fe426ea398364cebc146b1bd946b260d2c03b6280765afad0a2"}}


def checked_path(root, name):
    relative = PurePosixPath(name)
    path = root/name
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
            or path.is_symlink() or not path.resolve().is_relative_to(root)):
        raise ValueError(f"Invalid manifest path: {name}")
    return path


def repair(root):
    root = Path(root).resolve()
    manifest_path, backup_path = root/MANIFEST, root/BACKUP
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    files = manifest["files"]
    if not set(TOOL_UPDATES).issubset(files):
        raise ValueError("The manifest does not contain the expected original publication tools")
    repaired, removed, changes = {}, [], {}
    for name, wanted in files.items():
        path = checked_path(root, name)
        if is_install_metadata(name):
            removed.append(name)
            continue
        actual = hash_file(path)
        if name in TOOL_UPDATES:
            before, after = TOOL_UPDATES[name]["before"], TOOL_UPDATES[name]["after"]
            if wanted not in {before, after} or actual != after:
                raise ValueError(f"Publication tool does not match the delivered fix: {name}")
            repaired[name] = after
            if wanted != after:
                changes[name] = {"before": wanted, "after": after}
        else:
            if actual != wanted:
                raise ValueError(f"Frozen export file changed; repair refused: {name}")
            repaired[name] = wanted
    own_name = "scripts/repair_export_install_metadata.py"
    own_path = checked_path(root, own_name)
    if hash_file(own_path) != hash_file(Path(__file__)):
        raise ValueError("Repair script differs from the running fix")
    repaired[own_name] = hash_file(own_path)
    if manifest.get("install_metadata_fix") and not removed and not changes and files == repaired:
        verify_export(root)
        print("Metadata repair already applied; retained the original backup.")
        return
    if backup_path.exists():
        prior = manifest.get("install_metadata_fix", {}).get("original_manifest_sha256")
        if hash_file(backup_path) != (prior or hash_file(manifest_path)):
            raise ValueError("Existing backup differs from this export; preserve it and review before repairing")
    else:
        with backup_path.open("xb") as handle:
            handle.write(original)
    manifest["files"] = repaired
    manifest["install_metadata_fix"] = {
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_manifest_sha256": hash_file(backup_path),
        "removed_metadata_entries": sorted(removed), "publication_tool_updates": changes,
        "frozen_source_and_result_hashes_recomputed": False}
    temporary = manifest_path.with_suffix(".json.repair.tmp")
    temporary.write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
    temporary.replace(manifest_path)
    verify_export(root)
    print(f"Removed {len(removed)} generated metadata entries from the manifest; no installed file was deleted.")
    print("Frozen forecast-source and result hashes retained. No training or Git operation performed.")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        repair(ROOT)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f"Metadata repair stopped: {exc}") from None


if __name__ == "__main__":
    main()

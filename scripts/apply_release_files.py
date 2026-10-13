"""Install repository presentation and CI files while retaining prior setup docs."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "# BEGIN DEMANDCAST GENERATED FILES"
END = "# END DEMANDCAST GENERATED FILES"


def apply(root):
    root = Path(root).resolve()
    if not (root/"pyproject.toml").is_file():
        raise ValueError("Run this inside the original DemandCast checkout containing pyproject.toml")
    templates = root/"release_templates"
    readme = (root/"README_GITHUB.md").read_bytes()
    workflow = (templates/"ci.yml").read_bytes()
    patterns = (templates/"gitignore.txt").read_text(encoding="utf-8").strip()
    documents = root/"docs/release"
    documents.mkdir(parents=True, exist_ok=True)
    old_readme, backup = root/"README.md", documents/"CORE_SETUP_ORIGINAL.md"
    if not backup.exists():
        if not old_readme.exists():
            raise ValueError("The original README is needed to retain the core data-preparation instructions")
        backup.write_bytes(old_readme.read_bytes())
    destination = root/".github/workflows/ci.yml"
    previous_ci = documents/"CI_BEFORE_RELEASE.yml.txt"
    if destination.exists() and not previous_ci.exists():
        previous_ci.write_bytes(destination.read_bytes())
    ignore = root/".gitignore"
    original = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    if START in original:
        before, remainder = original.split(START, 1)
        if END not in remainder:
            raise ValueError("Incomplete managed .gitignore block; repair it before applying")
        _, after = remainder.split(END, 1)
        original = before.rstrip()+"\n"+after.lstrip("\r\n")
    managed = f"{START}\n{patterns}\n{END}\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    old_readme.write_bytes(readme)
    destination.write_bytes(workflow)
    ignore.write_text(original.rstrip()+"\n\n"+managed, encoding="utf-8")
    print("Installed README.md, .github/workflows/ci.yml and generated-file ignore rules.")
    print("Original setup documentation and any previous CI are retained under docs/release.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        apply(args.root)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot apply release files: {exc}\n")


if __name__ == "__main__":
    main()

"""Build the offline source-code package required by the course."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

INCLUDE_PARTS = {"src", "dataset", "training", "evaluation", "configs", "tests", "docs", "examples"}
INCLUDE_FILES = {"README.md", "pyproject.toml", ".gitignore", "requirements-training.txt"}
EXCLUDE_PARTS = {"data", "weights", "checkpoints", ".git", "node_modules", "dist", ".next", ".venv", "__pycache__"}


def files_for_archive(root: Path) -> list[Path]:
    selected = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in EXCLUDE_PARTS:
            continue
        if any(part in EXCLUDE_PARTS for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in INCLUDE_PARTS:
            selected.append(path)
        elif len(relative.parts) == 1 and relative.name in INCLUDE_FILES:
            selected.append(path)
    return sorted(selected)


def build_archive(root: Path, group_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"group_{group_id}_code.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
        for path in files_for_archive(root):
            zipped.write(path, path.relative_to(root).as_posix())
    manifest = {
        "group_id": group_id,
        "archive": archive.name,
        "contains_dataset": False,
        "contains_model_weights": False,
        "contains_external_links": False,
        "files": [path.relative_to(root).as_posix() for path in files_for_archive(root)],
    }
    (output_dir / "SUBMISSION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("submissions"))
    args = parser.parse_args()
    print(build_archive(Path(__file__).resolve().parents[1], args.group_id, args.output_dir))


if __name__ == "__main__":
    main()

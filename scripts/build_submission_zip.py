"""Build the offline source-code package required by the course."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

INCLUDE_PARTS = {"src", "dataset", "training", "evaluation", "experiments", "scripts", "configs", "tests", "docs", "examples", "reporting"}
INCLUDE_FILES = {"README.md", "pyproject.toml", ".gitignore", "requirements-training.txt", "THIRD_PARTY_NOTICES.md", "LICENSE"}
EXCLUDE_PARTS = {"data", "weights", "checkpoints", "runs", "outputs", "results", "evidence", ".git", "node_modules", "dist", ".next", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__"}
EXCLUDE_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pyc"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".yaml", ".yml", ".csv"}
EXCLUDE_NAMES = {"predictions.json", "cases.json", "cardd_manifest.json", ".env"}
URL = re.compile(r"https?://[^\s<>\"']+", re.I)
CLOUD_HOST = re.compile(r"(?:drive\.google\.com|docs\.google\.com|colab\.research\.google\.com|dropbox\.com|1drv\.ms|onedrive\.[^/]+|[^/]*sharepoint\.com|pan\.baidu\.com|huggingface\.co/[^/]+/[^/]+/resolve)/?", re.I)


def is_dataset_payload(value: object) -> bool:
    """Detect COCO annotations and exported instance masks, not numeric summaries."""
    if isinstance(value, dict):
        if isinstance(value.get("images"), list) and isinstance(value.get("annotations"), list):
            return True
        if "segmentation" in value and ("image_id" in value or "bbox" in value):
            return True
        return any(is_dataset_payload(item) for item in value.values())
    return isinstance(value, list) and any(is_dataset_payload(item) for item in value)


def files_for_archive(root: Path) -> list[Path]:
    selected = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        if path.name in EXCLUDE_NAMES or path.name.startswith("instances_"):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in INCLUDE_FILES:
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in EXCLUDE_PARTS:
            continue
        if any(part in EXCLUDE_PARTS for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in INCLUDE_PARTS:
            selected.append(path)
        elif relative.parts[:2] == ("CarDD project", "cardd_code"):
            # Retain team-provided baseline source, not notebooks, data or result dumps.
            selected.append(path)
        elif len(relative.parts) == 1 and relative.name in INCLUDE_FILES:
            selected.append(path)
    return sorted(selected)


def build_archive(root: Path, group_id: str, output_dir: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", group_id):
        raise ValueError("Group ID must be a short identifier, not a file path")
    root = root.resolve()
    selected = files_for_archive(root)
    payloads = {}
    for path in selected:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        if len(content) > 2_000_000:
            raise ValueError(f"Unexpected large source/summary file: {relative}")
        text = content.decode("utf-8")
        if path.suffix == ".json" and is_dataset_payload(json.loads(text)):
            raise ValueError(f"Dataset or per-instance prediction payload: {relative}")
        urls = URL.findall(text)
        if any(CLOUD_HOST.search(url.split("://", 1)[1]) for url in urls):
            raise ValueError(f"Cloud-hosted substitute-file link: {relative}")
        if urls:
            raise ValueError(f"External link not allowed in course submission: {relative}")
        payloads[relative] = content
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"group_{group_id}_code.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
        for relative, content in payloads.items():
            zipped.writestr(relative, content)
    with ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        for relative, content in payloads.items():
            if zipped.read(relative) != content:
                raise ValueError(f"Archive content mismatch: {relative}")
    manifest = {
        "group_id": group_id,
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "contains_dataset": False,
        "contains_model_weights": False,
        "contains_cloud_file_links": False,
        "contains_external_links": False,
        "link_policy": "Course rule: no external links. Any http(s) URL in a packaged file fails the build.",
        "files": list(payloads),
        "file_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
        "verification": "UTF-8 allowlist, JSON payload scan, cloud-link and external-link scan, size limit, no symlinks, ZIP CRC and byte-for-byte verification",
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

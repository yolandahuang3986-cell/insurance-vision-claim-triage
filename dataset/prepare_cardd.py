"""Create a deterministic image manifest for a local CarDD checkout."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_manifest(source_dir: Path, seed: int = 42) -> dict:
    images = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found under {source_dir}")

    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.70)
    validation_end = train_end + int(n * 0.15)

    records = []
    for index, path in enumerate(shuffled):
        split = "train" if index < train_end else "validation" if index < validation_end else "test"
        records.append({
            "image_id": path.stem,
            "relative_path": path.relative_to(source_dir).as_posix(),
            "split": split,
            "annotations": [],
        })
    return {"dataset": "CarDD", "seed": seed, "classes": [], "images": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_manifest(args.source_dir, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(manifest['images'])} images -> {args.output}")


if __name__ == "__main__":
    main()

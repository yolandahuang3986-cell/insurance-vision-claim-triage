"""Build a manifest from the official CarDD COCO instance annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SPLITS = {
    "train2017": "train",
    "val2017": "validation",
    "test2017": "test",
}


def resolve_coco_dir(source_dir: Path) -> Path:
    """Accept either ``CarDD_release`` or its nested ``CarDD_COCO`` directory."""
    candidates = [source_dir, source_dir / "CarDD_COCO"]
    for candidate in candidates:
        if (candidate / "annotations").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find CarDD_COCO/annotations under {source_dir}; "
        "pass the extracted CarDD_release or CarDD_COCO directory"
    )


def _load_split(coco_dir: Path, split_dir: str) -> tuple[list[dict], dict[int, str]]:
    annotation_file = coco_dir / "annotations" / f"instances_{split_dir}.json"
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Missing CarDD annotation file: {annotation_file}")

    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        category_id = int(annotation["category_id"])
        annotations_by_image[int(annotation["image_id"])].append(
            {
                "annotation_id": int(annotation["id"]),
                "category_id": category_id,
                "category": categories[category_id],
                "segmentation": annotation.get("segmentation", []),
                "bbox": annotation.get("bbox", []),
                "area": annotation.get("area", 0.0),
                "iscrowd": int(annotation.get("iscrowd", 0)),
                "attributes": annotation.get("attributes", {}),
            }
        )

    records = []
    split = SPLITS[split_dir]
    for image in data.get("images", []):
        image_id = int(image["id"])
        relative_path = (Path(split_dir) / image["file_name"]).as_posix()
        image_path = coco_dir / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by annotations is missing: {image_path}")
        records.append(
            {
                "image_id": f"{split}:{image_id}",
                "source_id": image_id,
                "relative_path": relative_path,
                "split": split,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": annotations_by_image[image_id],
            }
        )
    return records, categories


def build_manifest(source_dir: Path, seed: int = 42) -> dict:
    coco_dir = resolve_coco_dir(source_dir)
    records = []
    category_maps = []
    for split_dir in SPLITS:
        split_records, categories = _load_split(coco_dir, split_dir)
        records.extend(split_records)
        category_maps.append(categories)

    if not records:
        raise FileNotFoundError(f"No CarDD images found under {coco_dir}")
    if any(mapping != category_maps[0] for mapping in category_maps[1:]):
        raise ValueError("CarDD category mappings differ between splits")

    categories = category_maps[0]
    split_counts = {split: sum(item["split"] == split for item in records) for split in SPLITS.values()}
    annotation_counts = {
        split: sum(len(item["annotations"]) for item in records if item["split"] == split)
        for split in SPLITS.values()
    }
    return {
        "dataset": "CarDD",
        "format": "COCO-instance-segmentation",
        "source_dir": str(coco_dir),
        "seed": seed,
        "classes": [categories[index] for index in sorted(categories)],
        "class_map": {str(index): categories[index] for index in sorted(categories)},
        "summary": {"images": split_counts, "annotations": annotation_counts},
        "images": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Recorded for experiment provenance")
    args = parser.parse_args()
    manifest = build_manifest(args.source_dir, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Indexed {len(manifest['images'])} images and "
        f"{sum(manifest['summary']['annotations'].values())} annotations -> {args.output}"
    )


if __name__ == "__main__":
    main()

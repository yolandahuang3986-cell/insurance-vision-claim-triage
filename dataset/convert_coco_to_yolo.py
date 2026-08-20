"""Convert the official CarDD COCO split to YOLO segmentation format.

Images are symlinked by default so the conversion does not duplicate the
5+ GB CarDD image tree. Use ``--link-mode copy`` when a training environment
does not support symlinks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

try:
    from dataset.prepare_cardd import resolve_coco_dir
except ModuleNotFoundError:  # Direct execution: python dataset/convert_coco_to_yolo.py
    from prepare_cardd import resolve_coco_dir

YOLO_SPLITS = {"train2017": "train", "val2017": "val", "test2017": "test"}


def _normalise_polygon(polygon: list[float], width: int, height: int) -> list[float]:
    if len(polygon) < 6 or len(polygon) % 2:
        raise ValueError("A polygon must contain at least three x/y pairs")
    values = []
    for index, value in enumerate(polygon):
        normalised = value / width if index % 2 == 0 else value / height
        values.append(min(1.0, max(0.0, normalised)))
    return values


def _write_link_or_copy(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        destination.unlink()
    if link_mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        shutil.copy2(source, destination)


def _yaml(names: list[str]) -> str:
    quoted_names = ", ".join(json.dumps(name, ensure_ascii=False) for name in names)
    return "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"names: [{quoted_names}]",
            "",
        ]
    )


def convert(source_dir: Path, output_dir: Path, link_mode: str = "symlink") -> dict:
    coco_dir = resolve_coco_dir(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    category_maps = []
    summary = {"images": {}, "annotations": {}, "labels": {}, "invalid_polygons": 0}

    for coco_split, split in YOLO_SPLITS.items():
        annotation_file = coco_dir / "annotations" / f"instances_{coco_split}.json"
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        categories = {int(item["id"]): item["name"] for item in data["categories"]}
        category_maps.append(categories)
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in data.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

        image_count = 0
        annotation_count = 0
        label_count = 0
        for image in data.get("images", []):
            image_count += 1
            image_id = int(image["id"])
            source_image = coco_dir / coco_split / image["file_name"]
            destination_image = output_dir / "images" / split / image["file_name"]
            _write_link_or_copy(source_image, destination_image, link_mode)

            label_path = output_dir / "labels" / split / f"{Path(image['file_name']).stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for annotation in annotations_by_image.get(image_id, []):
                segmentation = annotation.get("segmentation", [])
                if not isinstance(segmentation, list) or len(segmentation) != 1:
                    summary["invalid_polygons"] += 1
                    continue
                try:
                    polygon = _normalise_polygon(segmentation[0], int(image["width"]), int(image["height"]))
                except (TypeError, ValueError, ZeroDivisionError):
                    summary["invalid_polygons"] += 1
                    continue
                class_index = sorted(categories).index(int(annotation["category_id"]))
                lines.append("{} {}".format(class_index, " ".join(f"{value:.6f}" for value in polygon)))
                annotation_count += 1
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            label_count += 1

        summary["images"][split] = image_count
        summary["annotations"][split] = annotation_count
        summary["labels"][split] = label_count

    if any(mapping != category_maps[0] for mapping in category_maps[1:]):
        raise ValueError("CarDD category mappings differ between splits")
    names = [category_maps[0][category_id] for category_id in sorted(category_maps[0])]
    (output_dir / "data.yaml").write_text(_yaml(names), encoding="utf-8")
    result = {
        "dataset": "CarDD",
        "format": "YOLO-segmentation",
        "source_dir": str(coco_dir),
        "output_dir": str(output_dir),
        "link_mode": link_mode,
        "classes": names,
        "summary": summary,
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()
    result = convert(args.source_dir, args.output_dir, args.link_mode)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

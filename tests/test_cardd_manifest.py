import json
from pathlib import Path

from dataset.prepare_cardd import build_manifest


def _write_split(root: Path, split_dir: str, image_id: int, filename: str) -> None:
    (root / split_dir).mkdir(parents=True)
    (root / split_dir / filename).write_bytes(b"mock image")
    (root / "annotations").mkdir(exist_ok=True)
    payload = {
        "images": [{"id": image_id, "width": 100, "height": 80, "file_name": filename}],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [[1, 1, 10, 1, 10, 10]],
                "bbox": [1, 1, 9, 9],
                "area": 81.0,
                "iscrowd": 0,
                "attributes": {"occluded": False},
            }
        ],
        "categories": [{"id": 1, "name": "dent"}],
    }
    (root / "annotations" / f"instances_{split_dir}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_manifest_reads_cardd_coco_splits(tmp_path: Path):
    coco = tmp_path / "CarDD_COCO"
    for index, split in enumerate(("train2017", "val2017", "test2017"), start=1):
        _write_split(coco, split, index, f"{index:06d}.jpg")

    manifest = build_manifest(tmp_path)

    assert manifest["classes"] == ["dent"]
    assert manifest["summary"] == {
        "images": {"train": 1, "validation": 1, "test": 1},
        "annotations": {"train": 1, "validation": 1, "test": 1},
    }
    assert manifest["images"][0]["annotations"][0]["category"] == "dent"

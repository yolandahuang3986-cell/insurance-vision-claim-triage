import json
from pathlib import Path

from dataset.convert_coco_to_yolo import convert


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
                "segmentation": [[10, 8, 30, 8, 30, 24]],
                "bbox": [10, 8, 20, 16],
                "area": 160.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "dent"}],
    }
    (root / "annotations" / f"instances_{split_dir}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_convert_coco_to_yolo_creates_labels_yaml_and_symlinks(tmp_path: Path):
    coco = tmp_path / "CarDD_COCO"
    for index, split in enumerate(("train2017", "val2017", "test2017"), start=1):
        _write_split(coco, split, index, f"{index:06d}.jpg")

    output = tmp_path / "cardd_yolo"
    result = convert(tmp_path, output)

    assert result["summary"]["images"] == {"train": 1, "val": 1, "test": 1}
    assert result["summary"]["annotations"] == {"train": 1, "val": 1, "test": 1}
    assert (output / "images" / "train" / "000001.jpg").is_symlink()
    assert (output / "labels" / "train" / "000001.txt").read_text() == "0 0.100000 0.100000 0.300000 0.100000 0.300000 0.300000\n"
    assert 'names: ["dent"]' in (output / "data.yaml").read_text()

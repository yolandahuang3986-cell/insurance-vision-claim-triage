import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from training.cardd_dataset import CarDDInstanceDataset


def test_cardd_dataset_returns_torchvision_detection_target(tmp_path: Path):
    source = tmp_path / "CarDD_COCO"
    (source / "train2017").mkdir(parents=True)
    (source / "annotations").mkdir()
    image_path = source / "train2017" / "000001.jpg"
    Image.new("RGB", (20, 10), "white").save(image_path)
    manifest = {
        "dataset": "CarDD",
        "source_dir": str(source),
        "classes": ["dent"],
        "class_map": {"1": "dent"},
        "images": [
            {
                "image_id": "train:1",
                "relative_path": "train2017/000001.jpg",
                "split": "train",
                "width": 20,
                "height": 10,
                "annotations": [
                    {
                        "category_id": 1,
                        "segmentation": [[2, 2, 10, 2, 10, 8]],
                        "bbox": [2, 2, 8, 6],
                        "area": 24,
                        "iscrowd": 0,
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    image, target = CarDDInstanceDataset(manifest_path, "train")[0]

    assert tuple(image.shape) == (3, 10, 20)
    assert tuple(target["boxes"].shape) == (1, 4)
    assert target["labels"].tolist() == [1]
    assert tuple(target["masks"].shape) == (1, 10, 20)
    assert int(target["masks"].sum()) > 0

"""Torchvision Dataset adapter for the CarDD manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


class CarDDInstanceDataset(Dataset):
    """Load one official CarDD split as torchvision detection targets."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transforms: Optional[Callable[[torch.Tensor, dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.split = split
        self.transforms = transforms
        self.class_map = {int(key): value for key, value in payload["class_map"].items()}
        self.class_to_index = {name: index for index, name in enumerate(payload["classes"], start=1)}
        self.records = [record for record in payload["images"] if record["split"] == split]
        self.source_dir = self._resolve_source_dir(Path(payload["source_dir"]))

    def _resolve_source_dir(self, source_dir: Path) -> Path:
        candidates = [
            source_dir,
            self.manifest_path.parent / source_dir,
            Path.cwd() / source_dir,
        ]
        for candidate in candidates:
            if (candidate / "annotations").is_dir():
                return candidate.resolve()
        raise FileNotFoundError(f"CarDD source_dir does not exist: {source_dir}")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _polygon_mask(width: int, height: int, polygon: list[float]) -> Image.Image:
        mask = Image.new("L", (width, height), 0)
        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
        return mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        image = Image.open(self.source_dir / record["relative_path"]).convert("RGB")
        width, height = image.size
        image_tensor = F.pil_to_tensor(image).float().div(255.0)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        for annotation in record.get("annotations", []):
            polygons = annotation.get("segmentation", [])
            if not isinstance(polygons, list) or len(polygons) != 1:
                continue
            polygon = polygons[0]
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                continue
            category_id = int(annotation["category_id"])
            if category_id not in self.class_map:
                continue
            x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2 = min(float(width), x + max(0.0, box_width))
            y2 = min(float(height), y + max(0.0, box_height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_index[self.class_map[category_id]])
            masks.append(F.pil_to_tensor(self._polygon_mask(width, height, polygon)).squeeze(0).to(torch.uint8))
            areas.append(float(annotation.get("area", (x2 - x1) * (y2 - y1))))
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)
        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)

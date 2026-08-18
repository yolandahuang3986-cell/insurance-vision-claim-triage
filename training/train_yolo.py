"""Optional YOLO segmentation training skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(data_yaml: Path, output_dir: Path, epochs: int = 50, image_size: int = 640) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional 'ultralytics' dependency to run YOLO training") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n-seg.pt")
    result = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        project=str(output_dir),
        name="yolo_segmentation",
    )
    summary = {
        "method": "yolo_segmentation",
        "data_yaml": str(data_yaml),
        "epochs": epochs,
        "image_size": image_size,
        "status": "completed",
        "result_type": type(result).__name__,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    print(json.dumps(train(args.data_yaml, args.output_dir, args.epochs, args.image_size), indent=2))


if __name__ == "__main__":
    main()

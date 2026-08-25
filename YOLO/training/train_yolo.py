"""YOLO segmentation training entrypoint for the CarDD fair-comparison protocol.

Trains a YOLO segmentation baseline, then runs a separate evaluation pass on
the official CarDD *test* split (not the training-time val split) so the
reported numbers match what ``docs/course_project_plan.md`` asks for:
mask/box precision, recall, mAP@50, mAP@50:95 and per-image latency, under a
fixed seed and documented confidence / IoU thresholds.

Note: Ultralytics does not report a standalone "mean mask IoU" metric, so
that number is intentionally left out of this script. It should be computed
separately in ``evaluation/`` from matched prediction/ground-truth mask
pairs, then merged into this script's output before filling the report.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def _extract_metrics(results_dict: dict, suffix: str) -> dict:
    """Pull precision/recall/mAP out of an Ultralytics results_dict.

    ``suffix`` is "B" for box metrics or "M" for mask metrics; Ultralytics
    keys look like "metrics/precision(B)", "metrics/mAP50-95(M)", etc.
    """
    return {
        "precision": results_dict.get(f"metrics/precision({suffix})"),
        "recall": results_dict.get(f"metrics/recall({suffix})"),
        "mAP50": results_dict.get(f"metrics/mAP50({suffix})"),
        "mAP50_95": results_dict.get(f"metrics/mAP50-95({suffix})"),
    }


def train(
    data_yaml: Path,
    output_dir: Path,
    epochs: int = 50,
    image_size: int = 640,
    seed: int = 42,
    device: str | None = None,
    conf_threshold: float = 0.5,
    iou_match_threshold: float = 0.5,
    model_weights: str = "yolo11n-seg.pt",
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional 'ultralytics' dependency to run YOLO training") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_weights)
    train_start = time.time()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        seed=seed,
        device=device,
        project=str(output_dir),
        name="yolo_segmentation",
    )
    train_seconds = time.time() - train_start

    # Fair-comparison protocol requires reporting on the held-out *test*
    # split, not the val split Ultralytics uses during training.
    test_metrics = model.val(
        data=str(data_yaml),
        split="test",
        conf=conf_threshold,
        iou=iou_match_threshold,
        device=device,
    )
    results_dict = getattr(test_metrics, "results_dict", {}) or {}
    speed = getattr(test_metrics, "speed", {}) or {}
    latency_ms = sum(speed.get(key, 0.0) for key in ("preprocess", "inference", "postprocess"))

    summary = {
        "method": "yolo_segmentation",
        "data_yaml": str(data_yaml),
        "model_weights": model_weights,
        "epochs": epochs,
        "image_size": image_size,
        "seed": seed,
        "device": device or "auto",
        "confidence_threshold": conf_threshold,
        "iou_match_threshold": iou_match_threshold,
        "train_seconds": round(train_seconds, 1),
        "status": "completed",
        "evaluation_split": "test",
        "box_metrics": _extract_metrics(results_dict, "B"),
        "mask_metrics": _extract_metrics(results_dict, "M"),
        "latency_ms_per_image": round(latency_ms, 2) if speed else None,
        "latency_breakdown_ms": speed or None,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or None,
        },
        "note": (
            "mask IoU (mean matched-pair IoU) is not produced by this script; "
            "compute it in evaluation/ and merge before filling the report."
        ),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42, help="Fixed per experiments.json fair-comparison protocol")
    parser.add_argument("--device", type=str, default=None, help="e.g. 'cpu', '0', 'mps'; default lets Ultralytics choose")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold used for test-split evaluation")
    parser.add_argument("--iou-match-threshold", type=float, default=0.5, help="IoU matching threshold used for test-split evaluation")
    parser.add_argument("--model-weights", type=str, default="yolo11n-seg.pt", help="Base checkpoint to fine-tune from")
    args = parser.parse_args()
    summary = train(
        args.data_yaml,
        args.output_dir,
        epochs=args.epochs,
        image_size=args.image_size,
        seed=args.seed,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_match_threshold=args.iou_match_threshold,
        model_weights=args.model_weights,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

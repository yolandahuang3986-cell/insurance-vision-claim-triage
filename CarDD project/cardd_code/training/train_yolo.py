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
    batch: int | float = -1,
    cache: str | bool = "ram",
    workers: int = 8,
) -> dict:
    """Train a YOLO segmentation model and evaluate it on the test split.

    ``batch``/``cache``/``workers`` are speed knobs that do NOT change the
    number of epochs or the reported metrics' fairness: ``batch=-1`` lets
    Ultralytics auto-pick the largest batch size that fits in GPU memory
    (AMP is on by default, so this is safe on a T4), ``cache='ram'`` caches
    decoded images in memory after the first epoch (skips repeated JPEG
    decode -- worthwhile once images are already on local disk), and
    ``workers`` should roughly match the number of CPU cores available
    (Colab free tier typically has 2).
    """
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
        batch=batch,
        cache=cache,
        workers=workers,
    )
    train_seconds = time.time() - train_start

    # Ultralytics already reloads best.pt into `model` at the end of train(), but
    # reload it explicitly here too so this doesn't depend on that internal
    # behavior across versions -- the test-split numbers below are guaranteed
    # to come from the best checkpoint, not whatever epoch ran last.
    best_weights = output_dir / "yolo_segmentation" / "weights" / "best.pt"
    if best_weights.exists():
        model = YOLO(str(best_weights))

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
        "best_checkpoint": str(best_weights) if best_weights.exists() else None,
        "speed_settings": {"batch": batch, "cache": cache, "workers": workers},
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
    parser.add_argument("--batch", type=float, default=-1, help="Batch size; -1 lets Ultralytics auto-fill GPU memory")
    parser.add_argument("--cache", type=str, default="ram", help="'ram', 'disk', or empty string to disable caching")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers; match available CPU cores")
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
        batch=args.batch,
        cache=args.cache or False,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

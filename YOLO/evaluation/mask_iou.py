"""Mean matched-pair mask IoU for the CarDD test split.

training/train_yolo.py intentionally does not report mask IoU, because
Ultralytics' validation metrics are AP-style (precision/recall/mAP), not a
mean IoU over matched instances. This script fills that gap: it re-runs
inference on the official CarDD *test* split with the same confidence /
IoU-match thresholds used in train_yolo.py, matches each prediction to a
ground-truth instance mask (same class, best IoU, one prediction per GT
instance), and reports:

- mean mask IoU over matched pairs (the number course_project_plan.md wants)
- per-class mean mask IoU
- error category counts, reusing error_analysis.classify_error

Ground truth comes from the manifest produced by dataset/prepare_cardd.py,
so mask IoU is computed against the original COCO polygons at full image
resolution rather than YOLO-normalized coordinates. This keeps the mask IoU
number independent of any resizing/letterboxing train_yolo.py's data.yaml
pipeline applies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from evaluation.error_analysis import classify_error, count_errors
except ImportError:  # direct execution: python evaluation/mask_iou.py
    from error_analysis import classify_error, count_errors


def _rasterize(polygon_xy: list[float], width: int, height: int) -> np.ndarray:
    """Rasterize a flat [x1, y1, x2, y2, ...] pixel-space polygon to a boolean mask."""
    points = list(zip(polygon_xy[0::2], polygon_xy[1::2]))
    mask_img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_img).polygon(points, outline=1, fill=1)
    return np.array(mask_img, dtype=bool)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def _gt_masks_for_image(record: dict) -> list[tuple[str, np.ndarray]]:
    """Rasterize GT polygons, skipping the same malformed cases
    dataset/convert_coco_to_yolo.py skips (so GT counts stay comparable)."""
    width, height = record["width"], record["height"]
    masks = []
    for annotation in record["annotations"]:
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list) or len(segmentation) != 1:
            continue
        polygon = segmentation[0]
        if len(polygon) < 6 or len(polygon) % 2:
            continue
        masks.append((annotation["category"], _rasterize(polygon, width, height)))
    return masks


def evaluate(
    manifest_path: Path,
    source_dir: Path,
    weights_path: Path,
    conf_threshold: float = 0.5,
    iou_match_threshold: float = 0.5,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional 'ultralytics' dependency to run mask IoU evaluation") from exc

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = manifest["classes"]
    test_records = [record for record in manifest["images"] if record["split"] == "test"]
    if not test_records:
        raise ValueError(f"No test-split images found in manifest {manifest_path}")

    model = YOLO(str(weights_path))

    matched_ious: list[float] = []
    per_class_ious: dict[str, list[float]] = {name: [] for name in class_names}
    error_categories: list[str] = []
    unmatched_predictions = 0

    for record in test_records:
        image_path = source_dir / record["relative_path"]
        gt_masks = _gt_masks_for_image(record)

        result = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=iou_match_threshold,
            verbose=False,
        )[0]

        pred_masks: list[tuple[str, np.ndarray, float]] = []
        if result.masks is not None:
            for polygon_xy, cls_idx, confidence in zip(
                result.masks.xy, result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                class_name = class_names[int(cls_idx)]
                mask = _rasterize(polygon_xy.flatten().tolist(), record["width"], record["height"])
                pred_masks.append((class_name, mask, float(confidence)))

        # Greedy one-to-one matching, highest IoU first, same class only.
        gt_used = [False] * len(gt_masks)
        for pred_class, pred_mask, confidence in pred_masks:
            best_iou, best_index = 0.0, -1
            for index, (gt_class, gt_mask) in enumerate(gt_masks):
                if gt_used[index] or gt_class != pred_class:
                    continue
                iou = mask_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou, best_index = iou, index

            if best_index >= 0 and best_iou >= iou_match_threshold:
                gt_used[best_index] = True
                matched_ious.append(best_iou)
                per_class_ious[pred_class].append(best_iou)
                error_categories.append(classify_error(pred_class, pred_class, confidence, best_iou))
            else:
                unmatched_predictions += 1  # extra detection, no GT of this class nearby

        for used, (gt_class, _gt_mask) in zip(gt_used, gt_masks):
            if not used:
                error_categories.append(classify_error(gt_class, None, 0.0, 0.0))

    summary = {
        "evaluation_split": "test",
        "num_test_images": len(test_records),
        "conf_threshold": conf_threshold,
        "iou_match_threshold": iou_match_threshold,
        "mean_mask_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else None,
        "matched_pairs": len(matched_ious),
        "unmatched_predictions": unmatched_predictions,
        "per_class_mean_mask_iou": {
            name: (round(float(np.mean(values)), 4) if values else None)
            for name, values in per_class_ious.items()
        },
        "error_category_counts": count_errors(error_categories),
        "note": (
            "unmatched_predictions are extra detections with no same-class GT instance "
            "above iou_match_threshold. classify_error() has no dedicated false-positive "
            "category, so it is only applied to matched pairs and to missed GT instances; "
            "unmatched_predictions is reported separately instead of being forced into it."
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Output of dataset/prepare_cardd.py")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="CarDD_release (or CarDD_COCO) directory, same one passed to prepare_cardd.py --source-dir",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Trained weights, e.g. runs/yolo_segmentation/weights/best.pt")
    parser.add_argument("--output", type=Path, default=Path("evaluation/mask_iou_summary.json"))
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Match this to train_yolo.py's --conf-threshold")
    parser.add_argument("--iou-match-threshold", type=float, default=0.5, help="Match this to train_yolo.py's --iou-match-threshold")
    args = parser.parse_args()
    summary = evaluate(
        args.manifest,
        args.source_dir,
        args.weights,
        conf_threshold=args.conf_threshold,
        iou_match_threshold=args.iou_match_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

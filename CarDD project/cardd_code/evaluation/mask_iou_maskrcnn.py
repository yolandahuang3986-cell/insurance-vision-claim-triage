"""Mean matched-pair mask IoU for the Mask R-CNN baseline on the CarDD test split.

Mirrors evaluation/mask_iou.py (the YOLO version) so that both methods produce
a mask_iou_summary.json with the identical schema, and evaluation/merge_results.py
can be reused unmodified for either method's row in the comparison table.

training/train_mask_rcnn.py's own evaluate() reports precision/recall/mAP at
--iou-match-threshold, but does not report the mean IoU over matched pairs --
that number (what course_project_plan.md's "mask IoU" column wants) is computed
here instead, using the same greedy same-class matching protocol as the YOLO
script for a fair comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

try:
    from training.train_mask_rcnn import build_model
except ImportError:  # direct execution: python evaluation/mask_iou_maskrcnn.py
    from train_mask_rcnn import build_model

try:
    from evaluation.error_analysis import classify_error, count_errors
except ImportError:
    from error_analysis import classify_error, count_errors


def _rasterize(polygon_xy: list[float], width: int, height: int) -> np.ndarray:
    points = list(zip(polygon_xy[0::2], polygon_xy[1::2]))
    mask_img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_img).polygon(points, outline=1, fill=1)
    return np.array(mask_img, dtype=bool)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def _gt_masks_for_image(record: dict) -> list[tuple[str, np.ndarray]]:
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
    checkpoint_path: Path,
    images_root: Path | None = None,
    conf_threshold: float = 0.5,
    iou_match_threshold: float = 0.5,
    device: str | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = manifest["classes"]
    test_records = [record for record in manifest["images"] if record["split"] == "test"]
    if not test_records:
        raise ValueError(f"No test-split images found in manifest {manifest_path}")

    if images_root is None:
        # Same self-contained reference train_mask_rcnn.py's own _default_images_root()
        # uses: read the raw-data location back out of the manifest (recorded by
        # dataset/prepare_cardd.py) rather than requiring a second path to track by hand.
        if "source_dir" not in manifest:
            raise ValueError(
                f"{manifest_path} has no 'source_dir' field -- rebuild it with "
                "dataset/prepare_cardd.py, or pass --images-root explicitly."
            )
        images_root = Path(manifest["source_dir"])
    else:
        images_root = Path(images_root)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    category_id_to_label = {int(k): v for k, v in checkpoint["category_id_to_label"].items()}
    label_to_class_name = {label: classes[label - 1] for label in category_id_to_label.values()}
    image_size = checkpoint.get("image_size", 800)

    model = build_model(num_classes=len(classes) + 1, image_size=image_size, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    matched_ious: list[float] = []
    per_class_ious: dict[str, list[float]] = {name: [] for name in class_names}
    error_categories: list[str] = []
    unmatched_predictions = 0

    with torch.no_grad():
        for record in test_records:
            image_path = images_root / record["relative_path"]
            gt_masks = _gt_masks_for_image(record)

            image = Image.open(image_path).convert("RGB")
            image_t = TF.to_tensor(image).to(device)
            output = model([image_t])[0]

            keep = output["scores"] >= conf_threshold
            pred_masks_raw = output["masks"][keep]
            pred_labels = output["labels"][keep].cpu().tolist()
            pred_scores = output["scores"][keep].cpu().tolist()

            pred_masks: list[tuple[str, np.ndarray, float]] = []
            for mask_t, label, score in zip(pred_masks_raw, pred_labels, pred_scores):
                if label not in label_to_class_name:
                    continue
                class_name = label_to_class_name[label]
                mask = (mask_t.squeeze(0).cpu().numpy() > 0.5)
                pred_masks.append((class_name, mask, float(score)))

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
                    unmatched_predictions += 1

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
            "Same greedy same-class matched-pair protocol as evaluation/mask_iou.py "
            "(the YOLO version), applied to the Mask R-CNN checkpoint, so both methods "
            "are directly comparable."
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Output of dataset/prepare_cardd.py")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="CarDD_release (or CarDD_COCO) directory (default: read from --manifest's own 'source_dir' field)",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="training/train_mask_rcnn.py's final.pt")
    parser.add_argument("--output", type=Path, default=Path("evaluation/mask_iou_summary_mask_rcnn.json"))
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--iou-match-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    summary = evaluate(
        args.manifest,
        args.checkpoint,
        images_root=args.images_root,
        conf_threshold=args.conf_threshold,
        iou_match_threshold=args.iou_match_threshold,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

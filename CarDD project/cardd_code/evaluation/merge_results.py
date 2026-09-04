"""Merge train_yolo.py's training_summary.json with mask_iou.py's mask IoU
summary into one row matching course_project_plan.md's result table
(mask IoU, Precision, Recall, mAP@50, mAP@50:95, Latency).

Precision/recall/mAP are taken from the *mask* metrics (not box metrics),
since the comparison is about segmentation quality; box metrics are kept
alongside as extra context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from evaluation.metrics import summarize_method
except ImportError:  # direct execution: python evaluation/merge_results.py
    from metrics import summarize_method


def merge(training_summary_path: Path, mask_iou_summary_path: Path, method_label: str) -> dict:
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    mask_iou_summary = json.loads(mask_iou_summary_path.read_text(encoding="utf-8"))

    if training_summary.get("evaluation_split") != mask_iou_summary.get("evaluation_split"):
        raise ValueError("training_summary and mask_iou_summary were computed on different splits")

    box_metrics = training_summary["box_metrics"]
    mask_metrics = training_summary["mask_metrics"]
    if mask_iou_summary["mean_mask_iou"] is None:
        raise ValueError("mean_mask_iou is null in mask_iou_summary.json - no predictions matched any GT instance")

    return summarize_method(
        method=method_label,
        mask_iou=mask_iou_summary["mean_mask_iou"],
        precision=mask_metrics["precision"],
        recall=mask_metrics["recall"],
        latency_ms=training_summary["latency_ms_per_image"],
        mAP50=mask_metrics["mAP50"],
        mAP50_95=mask_metrics["mAP50_95"],
        box_precision=box_metrics["precision"],
        box_recall=box_metrics["recall"],
        matched_pairs=mask_iou_summary["matched_pairs"],
        unmatched_predictions=mask_iou_summary["unmatched_predictions"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, required=True, help="Output of training/train_yolo.py")
    parser.add_argument("--mask-iou-summary", type=Path, required=True, help="Output of evaluation/mask_iou.py")
    parser.add_argument("--method-label", type=str, default="YOLO Segmentation", help="Row label for the report table")
    parser.add_argument("--output", type=Path, default=Path("evaluation/final_results_row.json"))
    args = parser.parse_args()
    row = merge(args.training_summary, args.mask_iou_summary, args.method_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

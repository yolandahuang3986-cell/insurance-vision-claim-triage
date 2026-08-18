"""Dependency-light metric helpers for smoke tests and result aggregation."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

Box = Sequence[float]


def box_iou(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def precision_recall(outcomes: Iterable[Tuple[bool, bool]]) -> Tuple[float, float]:
    """Calculate precision/recall from (predicted_positive, actual_positive) pairs."""
    tp = fp = fn = 0
    for predicted_positive, actual_positive in outcomes:
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive:
            fp += 1
        elif actual_positive:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def summarize_method(method: str, mask_iou: float, precision: float, recall: float, latency_ms: float, **extra: float) -> dict:
    return {
        "method": method,
        "mask_iou": mask_iou,
        "precision": precision,
        "recall": recall,
        "latency_ms": latency_ms,
        **extra,
    }

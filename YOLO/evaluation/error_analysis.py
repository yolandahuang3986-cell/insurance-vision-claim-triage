"""Consistent error categories for qualitative analysis."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional


def classify_error(
    expected_class: Optional[str],
    predicted_class: Optional[str],
    confidence: float,
    iou: float,
    image_quality_passed: bool = True,
) -> str:
    if not image_quality_passed:
        return "bad_input_quality"
    if expected_class is not None and predicted_class is None:
        return "missed_damage"
    if expected_class is not None and predicted_class != expected_class:
        return "wrong_damage_class"
    if expected_class is not None and iou < 0.5:
        return "poor_localization"
    if predicted_class is not None and confidence < 0.5:
        return "low_confidence_correct"
    return "correct"


def count_errors(categories: Iterable[str]) -> dict[str, int]:
    return dict(Counter(categories))

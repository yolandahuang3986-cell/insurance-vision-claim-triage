from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    vision_provider: str = "mock"
    min_image_width: int = 640
    min_image_height: int = 480
    min_blur_score: int = 60
    min_brightness: int = 35
    max_brightness: int = 230
    duplicate_phash_threshold: float = 0.92
    low_damage_confidence: float = 0.60

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            vision_provider=os.getenv("VISION_PROVIDER", "mock"),
            min_image_width=_int("MIN_IMAGE_WIDTH", 640),
            min_image_height=_int("MIN_IMAGE_HEIGHT", 480),
            min_blur_score=_int("MIN_BLUR_SCORE", 60),
            min_brightness=_int("MIN_BRIGHTNESS", 35),
            max_brightness=_int("MAX_BRIGHTNESS", 230),
            duplicate_phash_threshold=_float("DUPLICATE_PHASH_THRESHOLD", 0.92),
            low_damage_confidence=_float("LOW_DAMAGE_CONFIDENCE", 0.60),
        )


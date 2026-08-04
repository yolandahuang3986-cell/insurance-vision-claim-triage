from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol

from .models import ClaimImage, DamageDetection


class VisionProvider(Protocol):
    """Stable boundary for a future detector/segmenter."""

    def detect(self, images: Iterable[ClaimImage]) -> List[DamageDetection]: ...


@dataclass
class MockVisionProvider:
    """Deterministic provider for development and tests.

    The values represent model output, not ground truth. Keeping them outside
    the pipeline makes replacement with a real model explicit.
    """

    detections_by_image: Dict[str, List[DamageDetection]]

    def detect(self, images: Iterable[ClaimImage]) -> List[DamageDetection]:
        detections: List[DamageDetection] = []
        for image in images:
            detections.extend(self.detections_by_image.get(image.image_id, []))
        return detections


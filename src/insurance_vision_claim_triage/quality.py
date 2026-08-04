from __future__ import annotations

from typing import Iterable, List, Tuple

from .config import Settings
from .models import ClaimImage, DuplicateMatch, HistoricalImage, QualityCheck


def check_quality(image: ClaimImage, settings: Settings) -> QualityCheck:
    issues: List[str] = []
    if image.width < settings.min_image_width or image.height < settings.min_image_height:
        issues.append("resolution_below_minimum")
    if image.blur_score < settings.min_blur_score:
        issues.append("blur_below_minimum")
    if not settings.min_brightness <= image.brightness <= settings.max_brightness:
        issues.append("brightness_out_of_range")
    if not image.has_vehicle:
        issues.append("vehicle_not_detected")
    return QualityCheck(image_id=image.image_id, passed=not issues, issues=issues)


def _hamming(left: str, right: str) -> int:
    if not left or not right or len(left) != len(right):
        return 10_000
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 10_000


def _phash_similarity(left: str, right: str) -> float:
    distance = _hamming(left, right)
    if distance == 10_000:
        return 0.0
    bits = max(len(left), 1) * 4
    return round(1 - distance / bits, 4)


def find_duplicate_matches(
    images: Iterable[ClaimImage],
    history: Iterable[HistoricalImage],
    threshold: float,
) -> List[DuplicateMatch]:
    matches: List[DuplicateMatch] = []
    for image in images:
        for historical in history:
            if image.sha256 and historical.sha256 and image.sha256 == historical.sha256:
                matches.append(DuplicateMatch(image.image_id, historical.claim_id, 1.0, "exact_sha256"))
                continue
            similarity = _phash_similarity(image.phash or "", historical.phash or "")
            if similarity >= threshold:
                matches.append(DuplicateMatch(image.image_id, historical.claim_id, similarity, "near_phash"))
    return matches


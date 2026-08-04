from __future__ import annotations

from typing import Iterable, List

from .config import Settings
from .models import DamageDetection, DuplicateMatch, QualityCheck, RiskSignal


def _severity(damages: Iterable[DamageDetection]) -> str:
    total_area = sum(max(0.0, damage.area_ratio) for damage in damages)
    if total_area >= 0.10:
        return "severe"
    if total_area >= 0.03:
        return "medium"
    return "light"


def build_risk_signals(
    quality_checks: Iterable[QualityCheck],
    damages: Iterable[DamageDetection],
    duplicates: Iterable[DuplicateMatch],
    settings: Settings,
) -> List[RiskSignal]:
    signals: List[RiskSignal] = []
    for check in quality_checks:
        if not check.passed:
            signals.append(RiskSignal("image_quality_failed", "high", "图片质量或车辆完整性不足，需要补拍。", {"image_id": check.image_id, "issues": check.issues}))
    for match in duplicates:
        signals.append(RiskSignal("duplicate_image_risk", "high", "图片与历史案件高度相似，需人工核验；这不是欺诈结论。", {"image_id": match.image_id, "matched_claim_id": match.matched_claim_id, "similarity": match.similarity, "match_type": match.match_type}))
    for damage in damages:
        if damage.confidence < settings.low_damage_confidence:
            signals.append(RiskSignal("low_damage_confidence", "medium", "损伤识别置信度偏低，需要人工确认。", {"image_id": damage.image_id, "damage_id": damage.damage_id, "confidence": damage.confidence}))
    return signals


def choose_action(signals: Iterable[RiskSignal]) -> str:
    codes = {signal.code for signal in signals}
    if "duplicate_image_risk" in codes or "low_damage_confidence" in codes:
        return "manual_review"
    if "image_quality_failed" in codes:
        return "request_more_evidence"
    return "continue_claim_assessment"


def explanation(action: str, signals: List[RiskSignal]) -> str:
    if action == "request_more_evidence":
        return "请补充清晰、完整且包含车辆的照片；系统不会据此自动作出赔付结论。"
    if action == "manual_review":
        return "检测到需要人工核验的视觉风险信号；图片相似不等于欺诈，系统不会自动拒赔。"
    return "图片通过初步质量与风险检查，可进入后续理赔评估；这不是承保、责任或赔付决定。"


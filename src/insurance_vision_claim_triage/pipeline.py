from __future__ import annotations

from typing import Iterable, Optional

from .config import Settings
from .models import ClaimInput, HistoricalImage, PipelineResult, RiskSignal
from .quality import check_quality, find_duplicate_matches
from .triage import _severity, build_risk_signals, choose_action, explanation
from .vision import MockVisionProvider, VisionProvider


class ClaimTriagePipeline:
    def __init__(self, settings: Optional[Settings] = None, vision_provider: Optional[VisionProvider] = None, history: Iterable[HistoricalImage] = ()):
        self.settings = settings or Settings.from_env()
        self.vision_provider = vision_provider or MockVisionProvider({})
        self.history = list(history)

    def run(self, claim: ClaimInput) -> PipelineResult:
        quality_checks = [check_quality(image, self.settings) for image in claim.images]
        damages = self.vision_provider.detect(claim.images)
        duplicates = find_duplicate_matches(claim.images, self.history, self.settings.duplicate_phash_threshold)
        signals = build_risk_signals(quality_checks, damages, duplicates, self.settings)
        action = choose_action(signals)
        return PipelineResult(
            claim_id=claim.claim_id,
            quality_checks=quality_checks,
            damages=damages,
            risk_signals=signals,
            severity=_severity(damages),
            action=action,
            explanation=explanation(action, signals),
        )

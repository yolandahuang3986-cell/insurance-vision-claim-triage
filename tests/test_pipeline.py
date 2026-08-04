from insurance_vision_claim_triage.config import Settings
from insurance_vision_claim_triage.models import ClaimImage, ClaimInput, DamageDetection, HistoricalImage, Vehicle
from insurance_vision_claim_triage.pipeline import ClaimTriagePipeline
from insurance_vision_claim_triage.vision import MockVisionProvider


def claim(image: ClaimImage) -> ClaimInput:
    return ClaimInput("CLM-TEST-001", "POL-TEST-001", "2026-08-04T00:00:00Z", Vehicle("Demo", "Model", 2023), [image])


def good_image(**overrides) -> ClaimImage:
    values = dict(image_id="IMG-1", uri="mock://1", view="left", width=1920, height=1080, blur_score=80, brightness=120, has_vehicle=True, sha256="new", phash="aabbccddeeff0011")
    values.update(overrides)
    return ClaimImage(**values)


def test_good_mock_claim_continues():
    damage = DamageDetection("DMG-1", "IMG-1", "scratch", "left_door", 0.9, 0.02, [0.1, 0.2, 0.3, 0.4])
    result = ClaimTriagePipeline(vision_provider=MockVisionProvider({"IMG-1": [damage]})).run(claim(good_image()))
    assert result.action == "continue_claim_assessment"
    assert result.severity == "light"
    assert result.damages[0].damage_type == "scratch"


def test_bad_quality_requests_more_evidence():
    result = ClaimTriagePipeline().run(claim(good_image(width=320, blur_score=10, has_vehicle=False)))
    assert result.action == "request_more_evidence"
    assert {"resolution_below_minimum", "blur_below_minimum", "vehicle_not_detected"} <= set(result.quality_checks[0].issues)


def test_duplicate_image_routes_to_manual_review():
    history = [HistoricalImage("CLM-HIST-1", "IMG-HIST-1", sha256="same", phash="0000000000000000")]
    result = ClaimTriagePipeline(history=history).run(claim(good_image(sha256="same")))
    assert result.action == "manual_review"
    assert result.risk_signals[0].code == "duplicate_image_risk"
    assert "不是欺诈结论" in result.risk_signals[0].message


def test_low_confidence_damage_routes_to_manual_review():
    damage = DamageDetection("DMG-1", "IMG-1", "dent", "bumper", 0.45, 0.03, [0, 0, 1, 1])
    result = ClaimTriagePipeline(vision_provider=MockVisionProvider({"IMG-1": [damage]})).run(claim(good_image()))
    assert result.action == "manual_review"
    assert any(signal.code == "low_damage_confidence" for signal in result.risk_signals)


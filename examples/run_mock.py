from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insurance_vision_claim_triage.models import ClaimImage, ClaimInput, DamageDetection, HistoricalImage, Vehicle
from insurance_vision_claim_triage.pipeline import ClaimTriagePipeline
from insurance_vision_claim_triage.vision import MockVisionProvider


def main() -> None:
    claim_data = json.loads((ROOT / "examples/mock_claim.json").read_text(encoding="utf-8"))
    history_data = json.loads((ROOT / "examples/mock_history.json").read_text(encoding="utf-8"))
    claim = ClaimInput(
        claim_id=claim_data["claim_id"], policy_id=claim_data["policy_id"], incident_at=claim_data["incident_at"],
        vehicle=Vehicle(**claim_data["vehicle"]), customer_description=claim_data["customer_description"],
        images=[ClaimImage(**image) for image in claim_data["images"]],
    )
    detections = [DamageDetection("DMG-001", "IMG-001", "scratch", "left_front_door", 0.89, 0.018, [0.2, 0.3, 0.5, 0.5])]
    result = ClaimTriagePipeline(vision_provider=MockVisionProvider({"IMG-001": detections}), history=[HistoricalImage(**item) for item in history_data]).run(claim)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Vehicle:
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    plate_last4: Optional[str] = None


@dataclass(frozen=True)
class ClaimImage:
    image_id: str
    uri: str
    view: str
    width: int
    height: int
    blur_score: float
    brightness: float
    has_vehicle: bool
    sha256: Optional[str] = None
    phash: Optional[str] = None


@dataclass(frozen=True)
class HistoricalImage:
    claim_id: str
    image_id: str
    sha256: Optional[str] = None
    phash: Optional[str] = None


@dataclass(frozen=True)
class ClaimInput:
    claim_id: str
    policy_id: str
    incident_at: str
    vehicle: Vehicle
    images: List[ClaimImage]
    customer_description: str = ""


@dataclass(frozen=True)
class DamageDetection:
    damage_id: str
    image_id: str
    damage_type: str
    vehicle_part: Optional[str]
    confidence: float
    area_ratio: float
    bbox: List[float]


@dataclass(frozen=True)
class DuplicateMatch:
    image_id: str
    matched_claim_id: str
    similarity: float
    match_type: str


@dataclass(frozen=True)
class QualityCheck:
    image_id: str
    passed: bool
    issues: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RiskSignal:
    code: str
    severity: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    claim_id: str
    quality_checks: List[QualityCheck]
    damages: List[DamageDetection]
    risk_signals: List[RiskSignal]
    severity: str
    action: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

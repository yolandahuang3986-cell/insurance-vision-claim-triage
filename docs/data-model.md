# Data model

The MVP keeps the contract intentionally small and serialisable.

| Object | Purpose | Important fields |
|---|---|---|
| `ClaimInput` | One submitted claim | claim/policy IDs, incident time, vehicle, images, description |
| `ClaimImage` | Image metadata used by the gate | view, dimensions, blur/brightness, vehicle presence, hashes |
| `DamageDetection` | Provider observation | damage type, vehicle part, confidence, area ratio, bounding box |
| `HistoricalImage` | Authorized comparison reference | historical claim ID, SHA-256, pHash |
| `RiskSignal` | Evidence requiring attention | code, severity, message, structured evidence |
| `PipelineResult` | Auditable output | checks, detections, signals, severity, action, explanation |

The fields are deliberately not a claim adjudication schema. There is no coverage decision, liability field, repair quote, or payout amount in the MVP contract.


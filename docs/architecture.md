# Architecture and boundaries

```text
Claim intake
  -> quality gate (resolution / blur / brightness / vehicle presence)
  -> VisionProvider (mock now, real detector later)
  -> duplicate signal (SHA-256 / pHash against authorized history)
  -> deterministic triage rules
  -> structured, auditable result
```

The pipeline separates **observations** from **decisions**:

- A provider emits damage instances and confidence, but never decides coverage or payout.
- Quality and similarity are evidence signals, not a fraud verdict.
- The router only recommends the next operational step; a human remains responsible for claim decisions.

The MVP has no database, object storage, vector index, model weights, or external API. The `ClaimInput`, `HistoricalImage`, `DamageDetection`, and `PipelineResult` dataclasses are the first stable contracts. Later adapters may be added behind `VisionProvider`.


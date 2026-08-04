"use client";

import { useMemo, useState } from "react";

type Scenario = "clear" | "quality" | "duplicate" | "confidence";

const scenarios: Record<Scenario, { label: string; title: string; subtitle: string; action: string; tone: string; damage: string; confidence: string; area: string; evidence: string[] }> = {
  clear: {
    label: "Clear evidence",
    title: "Left-door scratch",
    subtitle: "One vehicle · left-side close-up · 1 image",
    action: "Continue assessment",
    tone: "safe",
    damage: "Scratch · left front door",
    confidence: "0.89",
    area: "1.8%",
    evidence: ["Image quality passed", "One vehicle detected", "No duplicate signal"],
  },
  quality: {
    label: "Poor image quality",
    title: "Dark, blurred photo",
    subtitle: "Resolution and vehicle visibility are insufficient",
    action: "Request more evidence",
    tone: "warn",
    damage: "No reliable damage observation",
    confidence: "—",
    area: "—",
    evidence: ["Resolution below minimum", "Blur score below minimum", "Vehicle not detected"],
  },
  duplicate: {
    label: "Duplicate-image risk",
    title: "Highly similar historical image",
    subtitle: "Exact SHA-256 match with CLM-HIST-101",
    action: "Manual review",
    tone: "risk",
    damage: "Scratch · left front door",
    confidence: "0.89",
    area: "1.8%",
    evidence: ["Exact image hash match", "Matched claim: CLM-HIST-101", "Similarity is a signal, not a fraud verdict"],
  },
  confidence: {
    label: "Low model confidence",
    title: "Possible bumper dent",
    subtitle: "The mock detector is not confident enough to self-route",
    action: "Manual review",
    tone: "risk",
    damage: "Dent · front bumper",
    confidence: "0.45",
    area: "3.0%",
    evidence: ["Damage confidence below 0.60", "Human confirmation required", "No automated claim decision"],
  },
};

export default function Home() {
  const [scenario, setScenario] = useState<Scenario>("clear");
  const current = useMemo(() => scenarios[scenario], [scenario]);

  return (
    <main className="shell">
      <nav className="topbar"><span className="brand-mark">IV</span><span className="brand-name">Insurance Vision</span><span className="nav-pill">CLAIM TRIAGE · MOCK MODE</span><span className="top-spacer" /><span className="status-dot" /> Offline-safe demo</nav>
      <section className="hero">
        <div className="hero-copy"><p className="eyebrow">VEHICLE CLAIM INTAKE</p><h1>See the evidence.<br /><em>Route with care.</em></h1><p className="lede">A deterministic visual pre-triage layer for motor claims. It checks image quality, structures damage observations, surfaces duplicate-image signals, and keeps final decisions with people.</p><div className="hero-note"><span>01</span><p>Mock inputs today.<br /><strong>Real model adapter reserved.</strong></p></div></div>
        <div className="hero-art" aria-label="Abstract vehicle damage visualization"><div className="scan-line" /><div className="car-outline"><span className="wheel left" /><span className="wheel right" /><span className="damage-mark" /></div><div className="art-label label-a">DAMAGE INSTANCE <b>01</b></div><div className="art-label label-b">CONFIDENCE <b>{current.confidence}</b></div><div className="art-label label-c">HUMAN-IN-LOOP</div></div>
      </section>
      <section className="workspace">
        <aside className="scenario-panel"><div className="section-kicker">MOCK SCENARIOS</div><h2>Choose an intake</h2><p className="muted">Each case runs through the same quality → vision → risk → routing flow.</p><div className="scenario-list">{(Object.keys(scenarios) as Scenario[]).map((key) => <button key={key} className={`scenario-button ${key === scenario ? "selected" : ""}`} onClick={() => setScenario(key)}><span className={`scenario-index ${scenarios[key].tone}`}>{String((Object.keys(scenarios) as Scenario[]).indexOf(key) + 1).padStart(2, "0")}</span><span><strong>{scenarios[key].label}</strong><small>{scenarios[key].title}</small></span><span className="arrow">→</span></button>)}</div><div className="boundary-card"><span className="lock">⊙</span><p><strong>Decision boundary</strong><br />This demo never approves, denies, prices, or declares fraud.</p></div></aside>
        <section className="result-panel"><div className="result-header"><div><div className="section-kicker">LIVE TRIAGE RESULT</div><h2>{current.title}</h2><p className="muted">{current.subtitle}</p></div><div className={`action-badge ${current.tone}`}>{current.action}</div></div><div className="result-grid"><div className="image-card"><div className="image-placeholder"><div className="corner tl" /><div className="corner tr" /><div className="corner bl" /><div className="corner br" /><div className="image-grid" /><span className="image-icon">▱</span><span className="image-caption">mock://vehicle-photo.jpg</span></div><div className="image-meta"><span>LEFT VIEW</span><span>1920 × 1080</span><span>SHA-256 tracked</span></div></div><div className="facts-card"><div className="fact-row"><span className="fact-label">DAMAGE OBSERVATION</span><strong>{current.damage}</strong></div><div className="fact-row"><span className="fact-label">CONFIDENCE</span><strong>{current.confidence}</strong></div><div className="fact-row"><span className="fact-label">MASK AREA RATIO</span><strong>{current.area}</strong></div><div className="fact-row"><span className="fact-label">SEVERITY LABEL</span><strong className="severity">{scenario === "confidence" ? "medium" : scenario === "quality" ? "unknown" : "light"}</strong></div></div></div><div className="evidence-block"><div className="evidence-head"><span className="section-kicker">AUDITABLE EVIDENCE</span><span className="evidence-count">{current.evidence.length} signals checked</span></div><div className="evidence-list">{current.evidence.map((item, index) => <div className="evidence-item" key={item}><span className={`evidence-icon ${current.tone}`}>{current.tone === "safe" ? "✓" : "!"}</span><span>{item}</span><span className="evidence-num">0{index + 1}</span></div>)}</div></div></section>
      </section>
      <footer><span>INSURANCE VISION CLAIM TRIAGE</span><span>MODEL-AGNOSTIC · EVIDENCE-FIRST · HUMAN-OWNED</span></footer>
    </main>
  );
}

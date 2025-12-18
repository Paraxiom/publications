# Augmented Democracy Paper - Section Cut Plan

## Overview

**Current state:** 50 pages (full-technical-draft.pdf)
**Target state:** 25-30 page core + appendices + Beamer deck

---

## Paper Core (paper-core.tex) — Target: 25-30 pages

### What stays in core (narrative-driven, no long code blocks)

| Section | Content | Target Pages |
|---------|---------|--------------|
| **1. Problem** | Democracy under adversarial load | 2 |
| **2. Philosophy** | Artifacts not truth, semantic authority problem | 3 |
| **3. Definition** | What augmented democracy is (and is not) | 2 |
| **4. Coherence Pipeline** | 6-stage overview with diagram (no code) | 4 |
| **5. Procedural Infrastructure** | Test grids, credentials, quadratic voting (condensed) | 5 |
| **6. Governance as Control** | Control theory mapping (condensed, diagram-heavy) | 4 |
| **7. Coherence Constraints** | γ definition, bounded influence, rollback | 3 |
| **8. Historical Evolution** | 2017 → EOS → Substrate → Quantum Harmony | 2 |
| **9. Conclusion** | Democracy as infrastructure | 2 |
| **References** | | 1 |
| **Total** | | ~28 pages |

### What gets cut/condensed from core

- All Rust code blocks (move to Appendix D)
- Detailed error taxonomy (move to Appendix B)
- Life-sustaining NFT implementation details (move to Appendix B)
- Full credential lifecycle code (move to Appendix B)
- Detailed attack analysis (move to Appendix C)
- Agent-mediated participation details (move to Appendix B or cut)
- UX section (condense to 1 paragraph or cut entirely)

---

## Appendices (paper-appendix.tex)

### Appendix A: Formal Definitions (~5 pages)

- Control system tuple (S, A, T, I, R)
- Coherence functional C: P → [0,1]
- Entropy variance definition (σ²η)
- Admissible Fact Artifact (formal)
- Token-Curated Test Grid (formal)
- Process Coherence (formal)
- Democratic Coherence invariant (6 conditions)
- Bounded influence constraints

### Appendix B: Protocol Mechanics (~8 pages)

- State machine diagrams
- Proposal status transitions
- Error taxonomy table
- Credential lifecycle (DynamicCredential struct)
- Life-sustaining NFT logic (LifeSustainingCredential)
- Delegation bounds (agent participation)
- Domain-specific credentials

### Appendix C: Security & Adversarial Analysis (~5 pages)

- Threat model overview
- Sybil attack resistance (how γ detects)
- Bribery resistance (how γ detects)
- Coordination attack resistance
- Entropy exhaustion scenarios
- What γ catches vs what it doesn't
- Mass agent attack defense
- Emergency governance procedures

### Appendix D: Code & Implementation Evidence (~6 pages)

- Rust excerpts from lib.rs
- Pallet structure overview
- Parameter values (thresholds, periods)
- Where each invariant is enforced
- Weight calculation code
- Quantum confidence calculation
- Vote finalization logic

---

## Diagrams to Add

### 1. coherence-pipeline.dot
Full 6-stage pipeline with gates:
```
Proposal → Review → Grid → Engagement → Vote → Result
```

### 2. control-loop.dot
Control theory view:
```
Sensors (participants) → Gains (credentials) →
Aggregation → Coherence Check (γ) → State Transition
```

### 3. threat-defense.dot
Attack → blocked by which gate:
```
Sybil → γ
Bribery → γ
Unengaged → Test Grid
Bot → Both
```

---

## Beamer Deck (beamer-deck.tex) — 15-20 slides

| Slide | Content |
|-------|---------|
| 1 | Title: Augmented Democracy |
| 2 | Problem: Democracy fails under adversarial load |
| 3 | Why traditional responses fail |
| 4 | Key insight: legitimacy = process quality |
| 5 | The semantic authority problem |
| 6 | Artifacts, not truth |
| 7 | Test grids: who defines facts? Nobody. |
| 8 | The coherence pipeline (diagram) |
| 9 | Coherence gate: what is γ? |
| 10 | What attacks this stops (diagram) |
| 11 | Why this is NOT technocracy |
| 12 | Why this is deployable (code exists) |
| 13 | Historical evolution (2017 → now) |
| 14 | Relationship to ERLHS / Proof of Coherence |
| 15 | The scope limitation (how, not what) |
| 16 | Closing: democracy as infrastructure |

---

## File Structure After Completion

```
augmented-democracy/
├── full-technical-draft.pdf    # Frozen 50-page version
├── paper-core.tex              # 25-30 page narrative
├── paper-core.pdf
├── paper-appendix.tex          # Technical appendices
├── paper-appendix.pdf
├── beamer-deck.tex             # Presentation slides
├── beamer-deck.pdf
├── diagrams/
│   ├── coherence-pipeline.dot
│   ├── coherence-pipeline.png
│   ├── control-loop.dot
│   ├── control-loop.png
│   ├── threat-defense.dot
│   └── threat-defense.png
├── main.tex                    # Original full version (for reference)
└── SECTION-CUT-PLAN.md         # This file
```

---

## Execution Order

1. [x] Freeze current as full-technical-draft.pdf
2. [ ] Create section cut plan (this document)
3. [ ] Generate .dot diagrams
4. [ ] Compile diagrams to .png
5. [ ] Create paper-core.tex
6. [ ] Create paper-appendix.tex
7. [ ] Create beamer-deck.tex
8. [ ] Compile all PDFs
9. [ ] Update publications README

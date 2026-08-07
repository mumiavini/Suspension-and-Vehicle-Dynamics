---
description: "Maps vdcore analysis outputs to the FSAE Design Event Score Sheet. What judges ask about suspension geometry, how results should be presented. Use when preparing Design Event presentations, reports, or reviewing whether outputs meet judging expectations."
disable-model-invocation: true
---

## FSAE Design Event — Suspension & Steering

### What judges look for

Design judges evaluate whether the team **understands** their design choices, not just whether the car exists. They ask:

1. **Why did you choose those values?** (target, rationale, trade-off)
2. **How did you verify them?** (simulation, bench test, track test, correlation)
3. **What compromises did you make?** (honesty about trade-offs wins points)
4. **How confident are you?** (data source, measurement uncertainty)

### Result presentation format

Every KPI in a report or presentation should follow this structure:

| Field | Content |
|---|---|
| **Parameter** | Name and symbol |
| **Target** | Value ± tolerance, with the FSAE event that drives it |
| **Achieved** | Computed or measured value |
| **Source** | cad / measured / estimate |
| **Confidence** | High (measured, correlated) / Medium (CAD, uncorrelated) / Low (estimate, no tire data) |
| **Rationale** | Why this target — link to vehicle behaviour, event, or competition rule |

### Key parameters judges ask about

#### Camber
- "What is your static camber and why?"
- "What is your camber gain in bump? How does that interact with your roll camber?"
- "At 1g lateral, what is your outside wheel camber?"
- Target: -1° to -3° static; gains negative in bump; outside wheel -1° to -2° at 1g lat.

#### Roll centre
- "Where is your roll centre? Front and rear?"
- "How does it migrate in roll?"
- "Is your roll axis nose-down? Why?"
- Target: 25–75 mm height; nose-down axis for understeer balance; lateral migration < 50 mm at 1g.

#### Anti-features
- "What anti-dive percentage did you target? Why?"
- Target: 20–50% anti-dive; 0–30% anti-squat. Higher = less pitch, but more bump steer.

#### Bump steer
- "Is your bump steer controlled?"
- Target: < 0.05 deg/mm in bump. Zero is ideal but rarely achievable.

#### Steering
- "What is your steering ratio?"
- "What is your parking effort? How did you calculate it?"
- "What Ackermann percentage and why?"
- Target: 3:1–5:1 ratio; < 20 Nm parking effort; 60–100% Ackermann (depends on event weighting).

### Confidence levels

| Level | Criteria | How to present |
|---|---|---|
| High | Measured on car, correlated with prediction | "Measured: X, predicted: Y, delta: Z%" |
| Medium | Computed from CAD geometry, not yet correlated | "Computed from CAD geometry. Correlation pending." |
| Low | Estimated, no tire data, significant assumptions | "Estimated. Assumes [state assumptions]. Tire data pending." |

**Never present a Low-confidence result as if it were High.** Judges respect honesty about uncertainty far more than false precision.

### Common pitfalls

1. **Presenting numbers without rationale**: "Our roll centre is at 55 mm" — but *why* 55 mm?
2. **Over-precision**: reporting camber to 0.001° from a CAD model with ±2 mm hardpoint tolerance.
3. **Missing the trade-off**: caster provides self-centering but increases steering effort. Discuss both.
4. **Ignoring uncertainty propagation**: a ±2 mm hardpoint tolerance propagates to ±0.5° of camber.
5. **Not knowing your numbers**: if asked "what happens to camber at 25 mm bump?", you should have the sweep plot ready.

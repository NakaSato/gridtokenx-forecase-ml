# PEA Hackathon 2026 — System Architecture v2

## Topic 4: Energy Resource Optimization (Ko Tao)

**Core Stack:** TCN + LightGBM + Ridge Meta-Learner / MILP + ISCA / pandapower + PyPSA
**Augment Layer:** Gemma 4 (E4B) — Apache 2.0
**Target:** Address D / V / F requirements directly based on the Problem Statement + Gemma as a unique value proposition.

---

## 0. Design Philosophy

> **"The numerical core remains unchanged — Gemma 4 serves as a cognitive layer that enables the system to explain, validate, generate, and act as an agent."**

```
Most Teams:                   Our Team:
[ML/MILP] → output             [ML/MILP] → output → [Gemma 4] → action+reason
   ↓                                                      ↓
Operator reads numbers         Operator receives recommendation + reason + action plan
```

**3 Golden Rules:**

1. **LLM must not touch forecast/optimization** — TCN/LightGBM/MILP handle this 10x better.
2. **LLM is strictly for language + reasoning + orchestration**.
3. **Every decision the LLM touches must have an SOP audit trail** — operator-in-the-loop is always required.

---

## 1. Mapping → Requirement D / V / F

### D1. Prototype system supports ≥ 4 sources

| Source            | Managed By                         | Gemma Augmentation    |
| ----------------- | ---------------------------------- | --------------------- |
| Main Grid (KMB-3) | pandapower + PyPSA (constraint)    | —                     |
| Diesel Genset     | MILP commit (binary on/off + ramp) | SOP validation        |
| BESS 50 MWh       | MILP dispatch (continuous)         | Cycle-life advice     |
| PV                | Forecast (TCN + weather)           | Curtailment reasoning |

→ **Deliverable:** Decision engine orchestrating 4 sources via MILP, **Gemma 4 = orchestrator + explainer**

### D2. AI/ML Forecast 24h MAPE ≤ 10%

**Original Stack (Unchanged):**

```
input → [feature eng] → TCN (sequence) ─┐
                     ├──→ LightGBM (tab) ─┴→ Ridge Meta-Learner → forecast 24h
                     └──→ Weather/Calendar
```

**Gemma Augmentation:**

- Generate a **forecast narrative** every hour — "Why the model predicts the peak will arrive 15 mins earlier than yesterday."
- **Confidence reasoning** — "Historical MAPE during a similar period is 7.2%; reliable."
- Ingest **textual context** invisible to TCN (e.g., "Songkran festival", "ferries suspended") → augment features.

→ **Deliverable:** TCN+LGBM+Ridge ensemble + **Gemma narrative report**

### D3. Optimization Model + Recommended Schedule

**Original Stack:**

```
Forecast + Constraint → MILP (Gurobi/CBC) → Optimal schedule (24h × 4 sources)
                     → ISCA (metaheuristic) → Refinement / robustness
                     → pandapower power flow → Validate no physics violations
```

**Requirement Prompt:** Compare "Adopting vs. Not Adopting" → **Liters of Diesel Fuel**

**Gemma Augmentation (Counterfactual Narrator):**

```python
# 2 schedules:
optimized_schedule = milp.solve(...)        # Our optimization
baseline_schedule  = current_practice(...)  # Manual operator baseline

diff = compute_diff(optimized, baseline)
gemma.narrate(diff)
# Output:
# "Following recommendations saves 145 liters/day because:
#  - 14:00–16:00 Used BESS instead of Diesel (Sufficient PV to charge)
#  - 19:00–21:00 Committing DG-1 at 4.2 MW instead of 6 MW (Reduced ramp loss)
#  Yearly ROI: ~52,800 liters = 1.85M THB (@35 THB/liter)"
```

→ **Deliverable:** MILP optimizer + **Gemma counterfactual report**

### D4. Early Warning + Actionable Recommendation

**Original Stack:**

```
Real-time SCADA stream → Anomaly detector (Isolation Forest)
                       → Forecast deviation check
                       → pandapower N-1 contingency
                       → Trigger if risk score > threshold
```

**Gemma Augmentation (Action Plan Agent):**

```python
# When detector triggers:
incident = {
  "type": "PV_ramp_down_predicted",
  "severity": 0.78,
  "lead_time_min": 25,
  "context": {pandapower_state, forecast, sop_db}
}

gemma_agent.plan(
  incident=incident,
  tools=[forecast_tool, dispatch_tool, sop_validator]
)
# Output: 3-step action plan + SOP reference + alternatives
```

→ **Deliverable:** Anomaly detector + **Gemma 4 agent generating action plans**

### V. Viability (THB/Year + Installation Months)

**Gemma Augmentation (Auto Financial Reporter):**

- Input: Reads MILP backtest logs covering 12 months → calculates savings projection.
- Output: **Auto-generated Executive Summary PDF** ready for Proposal inclusion.

```
Yearly Savings Projection (Gemma generated):
- Diesel reduction:     -52,800 liters × 35 THB = -1,848,000 THB/year
- BESS utilization:     +45% → defer hardware capex
- CO₂ avoided:          ~145 tons/year
- Estimated ROI:        Payback < 9 months
- System ready:         12 months (full deployment)
```

### F. Feasibility (Power System + ML expertise)

**Team Proof Points:**

1. Repo `gridtokenx-forecase-ml` — 2 years of actual implementation.
2. **This Architecture Document** — Demonstrates pandapower / MILP / TCN knowledge.
3. Gemma 4 integration = **Tech Innovation differentiator** (Triple Transformation).

---

## 2. Updated System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Web Dashboard         │  Edge Console  │  Mobile (E4B)  │  │
│  │  • Recommended Schedule (table)                          │  │
│  │  • Cost Curve Visualizer ⭐ (P2 anchor)                  │  │
│  │  • Counterfactual Comparison (liters/day, THB/year)      │  │
│  │  • Early Warning Banner + Action Plan                    │  │
│  │  • Auto Shift/Board Report Download                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────────┘
                       │ REST/WebSocket
┌──────────────────────▼─────────────────────────────────────────┐
│  GEMMA 4 AGENT LAYER (Apache 2.0)                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  EDGE: Gemma 4 E4B (Jetson Orin / On-prem at center)     │  │
│  │   • Realtime explanation                                 │  │
│  │   • Voice command (audio native)                         │  │
│  │   • Anomaly action plan                                  │  │
│  │                                                           │  │
│  │  CLOUD: Gemma 4 26B MoE (heavy reasoning, training)      │  │
│  │   • Counterfactual narrator                              │  │
│  │   • SOP validator (128K context, full manual)            │  │
│  │   • Executive report generator (multimodal)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────────┘
                       │ Function calling (native)
┌──────────────────────▼─────────────────────────────────────────┐
│  TOOLS LAYER (your stack, exposed as functions)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  forecast_tool()    → TCN + LightGBM + Ridge MetaLearner │  │
│  │  dispatch_tool()    → MILP + ISCA optimizer              │  │
│  │  physics_tool()     → pandapower + PyPSA power flow      │  │
│  │  anomaly_tool()     → Isolation Forest                   │  │
│  │  sop_tool()         → SOP RAG (vector DB)                │  │
│  │  baseline_tool()    → Current-practice simulator         │  │
│  │  report_tool()      → docx/pptx generator                │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────┐
│  DATA LAYER                                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Sandbox (PEA): SCADA replay, weather, BESS telem        │  │
│  │  Existing: GeoJSON topology, EGAT/PEA assets             │  │
│  │  SOP knowledge base (PDF/DOCX → vector embeddings)       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Forecast Module (D2)

**Stack:**

- TCN (PyTorch) — sequence-to-sequence 24h
- LightGBM — tabular features (weather, calendar, lag)
- Ridge Meta-Learner — blend 2 models
- Output: load + excess power + PV (multi-target)

**Targets (3 series, MAPE ≤ 10% each):**

1. Demand load (5–10 MW)
2. Excess power from main grid (0–16 MW)
3. PV generation (0–X MW depending on capacity)

**Gemma role:**

- Post-forecast → generate "forecast narrative" for operators to read.
- Confidence reasoning tied to historical MAPE over similar windows.
- (Optional) Few-shot text feature injection e.g., "long weekend", "storm approaching".

### 3.2 Optimization Module (D3)

**Stack:**

- MILP (Gurobi free / CBC open-source)
- Decision variables:
  - Diesel: binary commit + continuous power (0–10 MW, ramp constraint)
  - BESS: charge/discharge continuous (-50 to +50 MW), SoC tracking
  - PV: curtailment (0 to PV_forecast)
  - Main grid: continuous import (subject to KMB-3 limit)
- Objective: **minimize Diesel fuel liters** (primary) + cost (secondary)
- Constraints:
  - Power balance: ∑sources = demand
  - KMB-3 thermal limit (from pandapower)
  - BESS SoC bounds (20–90%)
  - Diesel min runtime, ramp rate
  - SOP-imposed (ramp/commit rules)

**ISCA (metaheuristic):** Used for robust optimization under uncertainty.

**Gemma role:**

- **Counterfactual Narrator** — Compare "MILP schedule vs current practice" → liters diff + reasoning.
- **SOP Validator** — Pre-check schedule before outputting to operator (long context parsing the entire SOP).

### 3.3 Physics Validation (D1)

**Stack:**

- pandapower 6-bus 115 kV model (custom)
- PyPSA for power flow / contingency
- Validate: MILP schedule → run AC power flow → check thermal/voltage.

**Gemma role:**

- Generate "physics violation explanation" if MILP schedule causes violations.
- Suggest alternatives that respect physics.

### 3.4 Early Warning (D4)

**Stack:**

- Isolation Forest on live SCADA (anomaly score)

- pandapower N-1 contingency simulation (lookahead)
- Trigger logic: combined score > threshold → alert

**Gemma role (CORE):**

- Receive incident dictionary → generate **3-step action plan**.
- Reference SOP.
- Translate technical jargon → actionable operator language.
- Addresses the "Actionable Recommendation" prompt directly.

**Example output:**

```
🚨 EARLY WARNING — Ko Tao (24 May 18:35)

Detected:  PV ramp-down predicted -3.5 MW in 25 mins
Cause:     Rainclouds approaching from SW (weather forecast)
Severity:  0.78 / 1.0

ACTION PLAN (3 Steps):
1. [Now 18:35]       Pre-charge BESS +12 MWh from main grid
                     Ref: SOP-BESS-2569-12 Section 4.1
2. [+15 min 18:50]   Begin warming DG-2 (idle, no commit)
                     Ref: SOP-DG-2569-04 Section 3.2
3. [+25 min 19:00]   If PV drops → commit DG-2 at 3 MW

If ignored: Risk of brownout pulse 2–5 mins (load ~4.1 MW unmet)

[ APPROVE ]  [ MODIFY ]  [ REJECT ]
```

### 3.5 Auto-Report (V — Viability)

**Use case:**

- End of shift → P1 handover summary
- End of month → P2 KPI report
- End of quarter → ESG/Board deck (PPTX)

**Tech:**

- Gemma 4 26B MoE (cloud) reads decision logs + SCADA replays.
- Generates via `report_tool()` → docx/pptx.

**Scoring Value:** Viability asks for "THB/Year figures" — Gemma generates the report instantly.

---

## 4. Gemma 4 Integration — 4 Use Cases (Select what is necessary)

| #   | Use Case                                  | Gemma Size  | Phase    | ROI    |
| --- | ----------------------------------------- | ----------- | -------- | ------ |
| 1   | **Decision Explainer** (D3 supplementary) | E4B         | Sprint 1 | ⭐⭐⭐ |
| 2   | **Action Plan Agent** (D4 core)           | E4B + tools | Sprint 2 | ⭐⭐⭐ |

---

## 5. 20-Day Sprint Plan

### Sprint 1: Exploration & Baseline (May 19–26)

| Day | Task                                                       |
| --- | ---------------------------------------------------------- |
| 1   | AWS Sandbox onboarding (PEA credits), data review          |
| 2   | Calibrate TCN+LGBM on Sandbox data (MAPE baseline)         |
| 3   | Setup MILP with 4 sources + KMB-3 constraint               |
| 4   | pandapower 6-bus model + verify topology with PEA          |
| 5   | Setup Gemma 4 E4B on Colab (test inference)                |
| 6   | Wrap tools (forecast, dispatch) into function-calling spec |
| 7   | Use Case #1 prototype (Decision Explainer)                 |
| 8   | PO Check-in 30 min (sprint review)                         |

**Deliverable Sprint 1:** Baseline forecast (MAPE ≤ 12% on Sandbox), MILP runs end-to-end, Gemma generates 1 sample explanation.

### Sprint 2: Development & Iteration (May 27–June 3)

| Day | Task                                           |
| --- | ---------------------------------------------- |
| 1–2 | Tune TCN+LGBM → MAPE ≤ 10% target              |
| 3   | Build SOP RAG (sample SOPs provided by PEA)    |
| 4   | Use Case #2 (SOP Validator) — Gemma 4 26B MoE  |
| 5   | Anomaly detector (IsolationForest)             |
| 6   | Use Case #3 (Action Plan Agent) — agentic loop |
| 7   | pandapower N-1 contingency + integration test  |
| 8   | PO Check-in + iterate based on feedback        |

**Deliverable Sprint 2:** Forecast MAPE ≤ 10%, Optimization end-to-end, Early Warning + Action Plan functioning, SOP Validator functioning.

### Sprint 3: Validation & Pitching (June 4–11)

| Day | Task                                                   |
| --- | ------------------------------------------------------ |
| 1   | Run "PEA-defined Incident" scenario — full system test |
| 2   | Counterfactual comparison (12 months replay)           |
| 3   | Use Case #4 (Auto-Report Generator)                    |
| 4   | Edge cases: KMB-3 trip, Diesel fault, BESS unavailable |
| 5   | Web Dashboard polish (Cost Curve, Counterfactual UI)   |
| 6   | (Optional) Edge demo on Jetson if budget allows        |
| 7   | Demo rehearsal + backup plan                           |
| 8   | PO Check-in final + code freeze                        |

**Deliverable Sprint 3:** Full demo pipeline, ESG/Board report sample, Pre-MVP video.

### Hack Days (June 15–17)

| Day | Task                                        |
| --- | ------------------------------------------- |
| 15  | Onsite — final Demo prep with PO            |
| 16  | Edge case stress test + Mentor consultation |
| 17  | **Final Pitch Day**                         |

## 7. Success Metrics & KPI

### Technical KPI

| Metric                            | Target       | Measurement                        |
| --------------------------------- | ------------ | ---------------------------------- |
| Forecast MAPE (24h backtest)      | **≤ 10%** ⭐ | Backtest 12-month, low+high season |
| Optimization solve time           | < 60 sec     | Per 24h schedule                   |
| Diesel liters savings vs baseline | ≥ 15%        | Counterfactual on historical       |
| Early Warning lead time           | ≥ 15 min     | Synthetic incidents                |
| SOP compliance rate               | 100%         | Validator pass rate                |
| Gemma response latency (E4B)      | < 3 sec      | On Sandbox VM                      |

### Business KPI (Viability)

| Metric                        | Estimate               |
| ----------------------------- | ---------------------- |
| Yearly Diesel savings         | ~52,800 liters         |
| Yearly cost savings           | ~1.85M THB (@35 THB/L) |
| CO₂ reduction                 | ~145 tons/year         |
| BESS utilization gain         | +30–45%                |
| Operator decision time        | -60% (with explainer)  |
| Time to install (full system) | **12 months**          |

---

## 8. Proposal Differentiators

| #   | Differentiator                             | Why Others Can't Replicate Easily                |
| --- | ------------------------------------------ | ------------------------------------------------ |
| 1   | **Hybrid stack** — TCN+LGBM+Ridge ensemble | Most use single models                           |
| 2   | **Physics validation** via pandapower      | Most optimize pure math without checking physics |
| 3   | **MILP + ISCA hybrid optimizer**           | Most rely purely on heuristics                   |
| 4   | **Gemma 4 Apache 2.0** — data sovereignty  | Cloud-only solutions lack this flexibility       |
| 5   | **Edge-ready (E4B on Jetson)**             | Hard to deploy an edge setup in 20 days          |
| 6   | **SOP-aware Recommender**                  | Requires SOP knowledge base + 128K context       |
| 7   | **Auto Executive Report**                  | Gemma generates defense-ready meeting PDFs       |
| 8   | **Counterfactual fuel tracking**           | Most show standard cost (THB), missing the mark  |

---

## 9. Triple Transformation Mapping (Strategic Relevance 5%)

| Aspect       | Proposition Delivered                                                                        |
| ------------ | -------------------------------------------------------------------------------------------- |
| **Business** | Reduce Diesel costs ~1.85M THB/year + scalability                                            |
| **Tech**     | AI/ML (TCN+LGBM) + Optimization (MILP+ISCA) + Physics (pandapower) + Generative AI (Gemma 4) |
| **People**   | Reduce P1 workload (Burnout prevention) + explainable AI fostering operator trust            |

---

## 10. Risk & Mitigation

| Risk                         | Mitigation                                                               |
| ---------------------------- | ------------------------------------------------------------------------ |
| Gemma 4 hallucinates numbers | Prevent Gemma from calculating; use ML/MILP outputs only, Gemma narrates |
| Missed MAPE ≤ 10% target     | Fallback: Ensemble with persistence baseline, or reduce forecast horizon |
| MILP solver doesn't converge | Fallback: ISCA metaheuristic, time-limited solve                         |
| Limited Sandbox data         | Augment with public datasets (NREL PERFORM) available in the repo        |
| Missing PEA SOPs             | Build mock SOPs using standard utility practices                         |
| Gemma fails to run on edge   | Fallback: Cloud-only inference                                           |

---

## 🎯 Summary — Proposal Deliverables

### Core (D / V / F):

1. **Forecast Engine** — TCN + LightGBM + Ridge, MAPE ≤ 10%
2. **Optimization Engine** — MILP + ISCA, supports 4 sources
3. **Physics Validator** — pandapower + PyPSA
4. **Early Warning Engine** — Isolation Forest
5. **Counterfactual Comparator** — Adopting vs. Not adopting recommendations → Fuel liters

### Gemma 4 Augmentation:

1. **Decision Explainer** (E4B) — Human-readable outputs
2. **Action Plan Agent** (E4B + tools) — Actionable Early Warnings

### Business Outcome (V):

- Savings of ~1.85M THB/year
- ROI < 9 months
- Full system installed in 12 months

---

_This system fulfills all D / V / F requirements — Gemma 4 serves as the cognitive layer creating a unique differentiator without compromising the integrity of the numerical core._

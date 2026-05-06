# GridTokenX: 20-Day Proof of Concept (POC) Execution Plan

This Proof of Concept (POC) execution plan synthesizes the system architecture, risk assessments, physical grid topology, and PEA integration requirements into a concrete 20-day roadmap for deploying the GridTokenX Predictive Intelligence Layer MVP.

---

## Sprint 1: POC Data Generation & Baseline Calibration (Days 1–7)
**Goal:** Generate high-fidelity theoretical datasets and establish performance baselines to prove the concept without relying on live PEA SCADA.

* **Day 1: Synthetic Data Pipeline**
  * Finalize `data/generate_dataset.py` to create highly realistic multi-year theoretical datasets.
  * Inject synthetic anomalies (e.g., PV drop-offs, sudden load spikes) for testing.
* **Day 2: Baseline Calibration**
  * Train the TCN + LightGBM pipeline on the theoretical dataset.
* **Day 3: Accuracy Audit (POC Data)**
  * Execute validation runs (`just eval`).
  * **KPI Target:** Prove MAPE ≤ 10.0% on the theoretical test set.
* **Day 4: Physics Topology Simulation**
  * Construct and validate the `pandapower` 6-bus theoretical model (Khanom-Samui bottleneck and Phangan-Tao link).
* **Day 5: Simulated Dispatch Validation**
  * Run the MILP optimizer on the theoretical forecast.
  * Compare fuel usage vs reactive dispatch baseline to prove efficiency.
* **Day 6: Automated Drift Simulation**
  * Simulate model drift over time and test the Ridge Meta-Learner's adaptability on the theoretical data.
* **Day 7: Sprint 1 Review & Buffer**
  * PO Check-in. Generate initial backtest reports on theoretical data.

---

## Sprint 2: Gemma 4 Cognitive Layer POC (Days 8–14)
**Goal:** Prove the viability of anomaly detection engines and the LLM operator assistant.

* **Day 8: Anomaly Engine Integration**
  * Deploy the Isolation Forest detector on the simulated SCADA stream.
  * Connect to the SQLite-backed `StreamingEngine`.
* **Day 9: Gemma 4 Knowledge Base Setup**
  * Vectorize PEA Standard Operating Procedures (SOPs).
  * Configure the RAG pipeline ensuring Gemma adheres to the "Zero-Math Policy" (no direct calculations).
* **Day 10: Use Case 1 (Decision Explainer)**
  * Deploy the counterfactual narrator.
  * Compare MILP dispatch vs Baseline practice to explain fuel savings to the operator in natural language.
* **Day 11: Use Case 2 (Action Plan Agent)**
  * Connect the Anomaly Engine to Gemma 4.
  * Ensure Gemma generates the required "3-step mitigation plan" with proper SOP citations.
* **Day 12: MILP Robustness Tuning**
  * Enforce strict 60-second timeouts on the Gurobi/CBC solver.
  * Implement the ISCA metaheuristic fallback to prevent solver hanging during tight constraint conflicts.
* **Day 13: N-1 Contingency Stress Testing**
  * Run automated stress tests simulating the loss of the Ko Samui 115 kV submarine cable.
  * Verify the system successfully commands the 10 MW Ko Tao Diesel units within ramp constraints.
* **Day 14: Sprint 2 Review & Buffer**
  * PO Check-in. Test Gemma 4 UI dashboards (Approve/Reject workflows).

**Sample Gemma 4 Output (POC Proof Points)**

**Use Case 1 — Decision Explainer (ภาษาไทย):**
```text
แนะนำเดิน Diesel DG-1 ที่ 4.2 MW เป็นเวลา 2 ชม. เริ่ม 19:00 น.

เหตุผล:
- Forecast บอก demand peak 8.5 MW ตอน 19:00–21:00 (TCN ensemble, MAPE 7.2%)
- BESS SoC ปัจจุบัน 65% — ไม่พอ discharge ตลอด peak (~30 MWh ขาด)
- KMB-3 ปัจจุบัน loaded 92% — ถ้า import เพิ่มเสี่ยง trip
- ทางเลือก BESS-only / full-grid ใช้น้ำมันเพิ่ม 80–120 ลิตร

อ้างอิง SOP-DC-2569-04 ข้อ 3.2 (Diesel commitment > 30 min)
```

**Use Case 3 — Action Plan Agent (Early Warning):**
```text
🚨 EARLY WARNING — เกาะเต่า (24 May 18:35)

ตรวจพบ: PV ramp-down คาด -3.5 MW ใน 25 นาที
สาเหตุ: เมฆฝนเข้าจาก SW (weather forecast)
Severity: 0.78 / 1.0

ACTION PLAN:
1. [ตอนนี้] pre-charge BESS +12 MWh จาก main grid → SOP-BESS-2569-12
2. [+15min] เริ่ม warm DG-2 (idle, ไม่ commit) → SOP-DG-2569-04
3. [+25min] ถ้า PV drop จริง → commit DG-2 ที่ 3 MW

หากไม่ทำ: เสี่ยง brownout pulse 2–5 นาที (load ~4.1 MW unmet)

[ APPROVE ]  [ MODIFY ]  [ REJECT ]
```

---

## Sprint 3: Sandbox Deployment & POC Validation (Days 15–20)
**Goal:** Deploy to the PEA Sandbox environment, validate theoretical economic savings, and finalize the POC commissioning report.

* **Day 15: Sandbox Environment Prep**
  * Configure Docker Compose stack for the Sandbox VM.
  * Ensure the FastAPI backend, Next.js dashboard, and MLflow tracking server run cleanly in the isolated demo environment.
* **Day 16: Cold Start & Outage Mitigation**
  * Implement fallback logic for simulated SCADA feeds dropping.
  * Add synthetic forecast bridging (ERA5 weather + calendar proxy) for outages lasting 1–6 hours.
* **Day 17: Multi-Island ADMM Testing**
  * Validate Alternating Direction Method of Multipliers (ADMM) logic for power sharing across Ko Tao, Phangan, and Samui (subject to PEA operator data-sharing agreements).
* **Day 18: Economic Viability Verification**
  * Execute Phase 3 of the Validation Plan using the theoretical KIREIP proxy dataset to prove the concept.
  * **KPI Target:** Prove >22% Diesel fuel savings vs legacy dispatch.
* **Day 19: Auto-Report Generation**
  * Utilize Gemma 4 to read the 12-month backtest logs and generate the ESG/Executive Board summary PDF (THB saved, CO2 reduced).
* **Day 20: Final POC Commissioning & Code Freeze**
  * Run `just report` to generate `results/pea_optimization_report.json`.
  * Secure internal validation sign-off for the POC demo.
  * Final Pitch/Demo Prep.

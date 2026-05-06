# GridTokenX Core Stack: Strengths, Risks, and Mitigations

This document outlines the architectural decisions for the GridTokenX Ko Tao deployment, detailing the specific benefits ("The Good") and potential vulnerabilities ("The Risks") of our chosen technologies, along with our mitigation strategies.

---

## 1. Forecasting: TCN + LightGBM + Ridge Meta-Learner

**Overview:** A hybrid ensemble model predicting 24-hour load, excess mainland power, and PV generation.

### ✅ The Good (Strengths)
* **Best of Both Worlds:** Temporal Convolutional Networks (TCN) perfectly capture smooth, continuous temporal cycles (daily/weekly inertia), while LightGBM (decision trees) excels at capturing sudden "shocks" in tabular data like sudden rainstorms or calendar events (Full Moon Parties).
* **High Accuracy:** By utilizing a Ridge Meta-Learner, the system automatically figures out which model to trust more at any given time, driving the Mean Absolute Percentage Error (MAPE) down to our target of ≤ 10%.
* **Parallelizable:** Unlike LSTMs, TCNs process time-series data using convolutions, allowing them to train significantly faster on modern GPUs.

### ⚠️ The Risks
* **Data Pipeline Complexity:** Maintaining two entirely different data engineering pipelines (sequential blocks for TCN, tabular rows for LightGBM) is prone to desynchronization errors.
* **Cold Start:** If a new sensor is added, the models require historical data to recalibrate accurately.

### 🛡️ Mitigation
* **Automated Data Contracts:** Strict schema validation via `pydantic` in our `preprocess.py` pipeline to ensure TCN and LightGBM inputs are perfectly aligned in time.

---

## 2. Dispatch Optimization: MILP + ISCA

**Overview:** The decision engine that dictates exactly when to turn on Diesel Generators, charge the BESS, and import mainland power.

### ✅ The Good (Strengths)
* **Mathematical Guarantees:** Mixed-Integer Linear Programming (MILP) guarantees the *absolute cheapest* operational schedule (minimizing diesel liters) mathematically, provided the constraints are met.
* **Constraint Awareness:** Naturally understands binary constraints ("Generator is ON or OFF") and ramping limits without complex workarounds.
* **Robust Fallback:** If the MILP struggles, the ISCA (metaheuristic) algorithm can rapidly search the solution space to find a "good enough" schedule instantly.

### ⚠️ The Risks
* **Solver Hanging:** MILP is NP-Hard. If constraints conflict (e.g., load is too high but diesel is capped), the solver might spin infinitely trying to find an impossible mathematical solution.

### 🛡️ Mitigation
* **Strict Timeouts:** The solver is strictly capped at 60 seconds.
* **Graceful Degradation:** If MILP times out, the system automatically falls back to ISCA or the reactive baseline schedule to ensure the operator always receives a plan.

---

## 3. Physics Validation: pandapower + PyPSA

**Overview:** Open-source power system analysis tools modeling the 6-bus 115 kV mainland connector to Ko Tao.

### ✅ The Good (Strengths)
* **Reality Check:** Prevents the optimizer from suggesting a schedule that looks cheap but would melt the XLPE submarine cables (thermal overloads).
* **Contingency Testing:** Allows us to run automated N-1 stress tests (e.g., simulating a sudden cable trip) to prove grid resilience to PEA.

### ⚠️ The Risks
* **Garbage In, Garbage Out:** Power flow physics rely completely on accurate impedance, admittance, and line ratings. If our static grid parameters are wrong, the simulation is completely invalid.

### 🛡️ Mitigation
* **PEA Calibration:** Dedicated `calibrate_with_real_data.py` script that aligns our baseline pandapower model perfectly with historical PEA SCADA truths.

---

## 4. Cognitive Augmentation: Gemma 4 + Isolation Forest

**Overview:** The Early Warning anomaly detector paired with an LLM operator assistant.

### ✅ The Good (Strengths)
* **Unsupervised Anomaly Detection:** Isolation Forest doesn't require labeled "failure" data. It learns what a "healthy" grid looks like and flags anything strange instantly.
* **Explainable AI:** Transforms a terrifying wall of numbers into a calm, human-readable 3-step action plan for the operator.
* **Data Sovereignty (The Open-Weights Advantage):** Because Gemma 4 is an open-weights model, it can be deployed on-premise or in a secure PEA Sandbox. Sensitive grid SCADA data and proprietary standard operating procedures never leave the utility's secure network. This is a massive compliance advantage over closed-source cloud APIs.
* **Reduced Cognitive Load:** Automates report generation (THB/Year saved) and SOP referencing, heavily reducing operator burnout during crisis moments.

### ⚠️ The Risks
* **LLM Hallucinations:** The most significant risk. An LLM could mathematically hallucinate fuel capacities or invent an SOP rule that doesn't exist.

### 🛡️ Mitigation
* **Zero-Math Policy:** Gemma 4 is strictly forbidden from calculating dispatch schedules. It only *narrates* the output provided by the MILP optimizer.
* **RAG Audit Trails:** Every action plan generated must cite an exact section from `mock_sops.json`.
* **Human-in-the-Loop:** All LLM plans output to the dashboard with strict `[ APPROVE ]`, `[ MODIFY ]`, or `[ REJECT ]` buttons. The LLM cannot execute grid actions directly.

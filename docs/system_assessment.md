# GridTokenX — System Assessment: Strengths & Weaknesses

**Date:** 2026-05-02 | **Scope:** Ko Tao-Phangan-Samui Predictive Intelligence Layer

---

## ✅ What Works Well

### Forecast Accuracy
- **MAPE: 1.67%** on held-out test set — beats the PEA target of 2.65% by a wide margin
- **R² = 0.977**, MAE = 0.128 MW — both within engineering targets
- The hybrid TCN + LightGBM + Ridge meta-learner architecture is well-suited to this problem: TCN captures temporal load curves, LightGBM handles weather non-linearities, Ridge blending prevents overfitting

### Dispatch Efficiency
- **89.25% fuel savings** vs reactive dispatch over 100 evaluated days
- BESS SoH estimated at 0.9961 after simulation — the 20–80% SoC operating band effectively extends battery life
- Proactive diesel scheduling reduces unnecessary runtime hours

### Feature Engineering
- Heat Index, temperature gradient, and rolling weather trends are physically meaningful and improve model generalization
- Lag features (1h, 24h, 168h) correctly encode daily and weekly load periodicity
- Calendar features (high season flag, hour of day) capture tourism-driven demand shifts

### Data Pipeline Robustness
- `preprocess.py` handles both real and synthetic data paths cleanly
- BESS SoC re-simulation fallback prevents constant-value columns from corrupting training
- Scaler correctly excludes load/lag features from normalization — avoids unit mismatch with TCN target

### Observability
- MLflow tracks every training run, hyperparameter search, and API inference trace
- Live streaming metrics (MAE, RMSE, MAPE) update per observation via `/stream/actual`
- Commissioning report auto-generated with KPIs, ROI, and physics verification

### Infrastructure
- FastAPI streaming buffer pattern is clean and stateless-friendly
- Docker + docker-compose makes deployment reproducible
- Optuna hyperparameter search is integrated and logged

---

## ❌ What Is Weak or Risky

### Real-Time Data Dependency
- The TCN requires a **48-hour warm-up buffer** before producing any forecast
- If SCADA feed is interrupted, there is no automated fallback — the system silently stops forecasting
- No built-in data quality checks on incoming telemetry (outlier detection, sensor drift, missing values)

### Synthetic Data Dominates Training
- The "real" dataset (`ko_tao_grid_2023_locked.parquet`) is itself calibrated from synthetic data — it is not true SCADA telemetry
- The 89% fuel savings figure is computed on synthetic test data, not real grid operations
- Distribution shift between synthetic training and actual PEA SCADA is unquantified until `pea_onboard.py` is run against real data

### LCOE Impact Is Negligible
- The commissioning report shows only **0.02% LCOE reduction** (727.69 → 727.54 THB/MWh)
- The 20-year NPV fuel savings is only **95,336 THB** against a 430,000,000 THB shared CapEx
- The economic case depends entirely on the fuel savings percentage being validated on real operations, not synthetic replay

### No Retraining Pipeline
- Models are trained once and frozen. There is no scheduled retraining or drift detection
- As real load patterns evolve (new resorts, EV adoption, grid expansion), model accuracy will degrade silently
- `pea_onboard.py` refits only the Ridge meta-learner — TCN and LightGBM weights are never updated from real data

### Single-Point API Architecture
- `StreamingEngine` state lives in memory — a server restart loses the 48-row buffer and all running metrics
- No message queue (Kafka, Redis) between SCADA and the API — data loss on API downtime
- No authentication on any API endpoint

### Tourist Index Is a Proxy
- In production, `Tourist_Index` is derived from `Is_High_Season * 0.4 + 0.6` — a binary approximation
- Actual tourism demand varies week-to-week (events, weather, ferry schedules) and this is not captured

### Multi-Island ADMM Is Unvalidated
- ADMM convergence is tested on synthetic data only
- Ko Phangan and Ko Samui have separate grid operators — data sharing agreements do not exist yet
- The 57.9 km cluster radius assumes a single coordinating entity

---

## Summary Table

| Area | Status | Risk |
|---|---|---|
| Forecast accuracy (synthetic) | ✅ Excellent | Low |
| Dispatch optimization (synthetic) | ✅ Strong | Low |
| Real SCADA integration | ❌ Not done | **High** |
| Economic validation on real data | ❌ Not done | **High** |
| Model retraining / drift handling | ❌ Missing | **High** |
| API resilience & auth | ❌ Weak | Medium |
| Multi-island coordination | ⚠️ Prototype only | Medium |
| Cold-start / SCADA outage handling | ⚠️ Manual workaround | Medium |

# GridTokenX — System Assessment: Strengths & Weaknesses

**Date:** 2026-05-03 | **Scope:** Ko Tao-Phangan-Samui Predictive Intelligence Layer

---

## What Works Well

### Forecast Accuracy
- **MAPE < 3%** on held-out synthetic test set — beats the PEA target of < 10.0%
- **R² > 0.95**, MAE < 0.25 MW — both within engineering targets
- The hybrid TCN + LightGBM + Ridge meta-learner architecture is well-suited to this problem: TCN captures temporal load curves, LightGBM handles weather non-linearities, Ridge blending prevents overfitting

### Dispatch Efficiency
- **>75% fuel savings** vs reactive dispatch over evaluated test periods (synthetic data)
- BESS SoH estimated at >0.99 after simulation — the 20–80% SoC operating band effectively extends battery life
- Ko Tao has **no local BESS** — dispatch optimization is diesel-only (10 MW rated)
- Proactive diesel scheduling reduces unnecessary runtime hours

### Feature Engineering
- Heat Index, temperature gradient, and rolling weather trends are physically meaningful and improve model generalization
- Lag features (1h, 24h, 168h) correctly encode daily and weekly load periodicity
- Calendar features (high season flag, hour of day, Thai holidays, Songkran) capture tourism-driven demand shifts
- Cluster spatial lags (`Phangan_Load_Lag_1h`, `Samui_Load_Lag_1h`, rolling means) now included in LightGBM FEATURES list

### Data Pipeline Robustness
- `preprocess.py` handles both real and synthetic data paths cleanly
- BESS SoC re-simulation fallback prevents constant-value columns from corrupting training
- Scaler correctly excludes load/lag features from normalization — avoids unit mismatch with TCN target
- Missing cluster features in real data splits zero-filled gracefully

### Observability
- MLflow tracks every training run, hyperparameter search, and API inference trace
- Experiment names: `GridTokenX_LGBM` (LightGBM), `GridTokenX_API` (serving), `GridTokenX` (general)
- Live streaming metrics (MAE, RMSE, MAPE) update per observation via `/stream/actual`
- Commissioning report auto-generated with KPIs, ROI, and physics verification

### Infrastructure
- FastAPI streaming engine with **SQLite-backed state** (`api_state.db`) — survives API restarts
- Docker + docker-compose makes deployment reproducible (API + Frontend + MLflow)
- Optuna hyperparameter search is integrated and logged
- Next.js 16 frontend with TypeScript, Tailwind CSS 4, Recharts, Leaflet/Mapbox
- pandapower 6-bus model + PyPSA for power flow analysis and OPF

---

## What Is Weak or Risky

### Real-Time Data Dependency
- The TCN requires a **48-hour warm-up buffer** (192 rows at 15-min intervals) before producing any forecast
- If SCADA feed is interrupted, there is no automated fallback — the system silently stops forecasting
- No built-in data quality checks on incoming telemetry (outlier detection, sensor drift, missing values)

### Synthetic Data Dominates Training
- The "real" dataset (`ko_tao_grid_2023_locked.parquet`) is itself calibrated from synthetic data — it is not true SCADA telemetry
- Fuel savings figures are computed on synthetic test data, not real grid operations
- Distribution shift between synthetic training and actual PEA SCADA is unquantified until `pea_onboard.py` is run against real data

### LCOE Impact Is Negligible
- The commissioning report shows marginal LCOE reduction
- The economic case depends entirely on the fuel savings percentage being validated on real operations, not synthetic replay

### No Retraining Pipeline
- Models are trained once and frozen. There is no scheduled retraining or drift detection
- As real load patterns evolve (new resorts, EV adoption, grid expansion), model accuracy will degrade silently
- `pea_onboard.py` refits only the Ridge meta-learner — TCN and LightGBM weights are never updated from real data

### API Security
- No authentication on any API endpoint
- No rate limiting or input sanitization beyond Pydantic schema validation

### Tourist Index Is a Proxy
- In production, `Tourist_Index` is derived from `Is_High_Season * 0.4 + 0.6` — a binary approximation
- Actual tourism demand varies week-to-week (events, weather, ferry schedules) and this is not captured

### Multi-Island ADMM Is Unvalidated on Real Data
- ADMM convergence is tested on synthetic data only
- Ko Phangan and Ko Samui have separate grid operators — data sharing agreements do not exist yet
- The 57.9 km cluster radius assumes a single coordinating entity

---

## Summary Table

| Area | Status | Risk |
|---|---|---|
| Forecast accuracy (synthetic) | Strong | Low |
| Dispatch optimization (synthetic) | Strong | Low |
| Streaming state persistence | Fixed (SQLite) | Low |
| Real SCADA integration | Not done | **High** |
| Economic validation on real data | Not done | **High** |
| Model retraining / drift handling | Missing | **High** |
| API security & auth | Weak | Medium |
| Multi-island coordination | Prototype only | Medium |
| Cold-start / SCADA outage handling | Manual workaround | Medium |

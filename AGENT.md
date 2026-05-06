# GridTokenX — AGENT.md

AI agent context for the Ko Tao-Phangan-Samui Predictive Intelligence codebase.

---

## Project Purpose

Physics-informed AI forecasting and dispatch system for three islanded microgrids connected to the Thai mainland via 115 kV XLPE submarine cables. Primary target: Ko Tao load forecast (MAPE < 10%, historically < 3% on synthetic benchmarks). Secondary: multi-island ADMM dispatch optimization. Ko Tao has **no BESS** — diesel-only dispatch.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 (`uv` + venv) |
| ML | PyTorch (TCN), LightGBM, scikit-learn (Ridge meta-learner) |
| Hyperparameter search | Optuna |
| Experiment tracking | MLflow (SQLite backend, `mlflow.db`) |
| Grid physics | pandapower, PyPSA |
| API | FastAPI + uvicorn |
| Task runner | `just` (see `justfile`) |
| Frontend | Next.js 16, TypeScript, Tailwind CSS 4, Recharts, Leaflet/Mapbox |
| Containerization | Docker + docker-compose |

---

## Key Commands

```bash
just setup         # uv sync
just generate      # regenerate synthetic dataset (data/generate_dataset.py)
just preprocess    # feature engineering → train/val/test parquets
just train         # preprocess → lgbm → tcn → hybrid pipeline
just lgbm          # train LightGBM only
just tcn           # train TCN only
just hybrid        # train meta-learner only
just eval          # evaluate forecast + dispatch vs PEA targets
just tune          # Optuna hyperparameter search (default 50 trials)
just optimize      # PEA MILP dispatch optimization (7-day test)
just api           # start FastAPI on :8000
just frontend      # start Next.js dashboard on :3000
just dev           # start full dev stack (API + Frontend concurrently)
just simulate      # replay test data through live API, print live metrics
just pea-onboard   # calibrate to real PEA SCADA CSV
just pea-backtest  # read-only backtest on PEA data (no refit)
just up / just down # docker-compose up/down (API + frontend + MLflow)

# 2026 Strategy & Commissioning
just backtest-12m      # 12-month backtest (May 2025 – Apr 2026)
just stress-test       # N-1 contingency stress test
just cluster-test      # Cluster-wide ADMM bottleneck test
just stochastic-test   # Monte Carlo stochastic resilience (N=500)
just opf-test          # Optimal Power Flow analysis
just diagnose          # holistic diagnostic report
just report            # run all commissioning tests + generate dashboard
```

---

## Architecture

```
data/generate_dataset.py    → data/processed/island_grid.parquet  (3-island synthetic, calibrated)
data/preprocess.py          → train/val/test.parquet + scaler.pkl
models/lgbm_model.py        → models/lgbm.pkl
models/tcn_model.py         → models/tcn.pt
models/hybrid_pipeline.py   → models/meta_learner.pkl
evaluate.py                 → results/evaluation_report.json
optimizer/run_optimization.py → results/pea_optimization_report.json
api/serve.py                → POST /stream/telemetry, /forecast, /warnings + GET /health, /metrics
optimizer/early_warning.py  → check_warnings() — uses pandapower AC power flow
optimizer/admm_resilience.py → multi-island ADMM consensus dispatch
optimizer/pea_dispatch_opt.py → HiGHS MILP dispatch optimizer
optimizer/isca.py           → ISCA metaheuristic optimizer
research/pandapower_model.py → 6-bus 115 kV physics model
research/pypsa_model.py     → PyPSA linear power flow analysis
research/backtest_12m.py    → 12-month rolling backtest
research/contingency_analysis.py → N-1 contingency stress test
research/cluster_stress_test.py  → multi-island ADMM bottleneck test
research/monte_carlo_engine.py   → stochastic resilience (N=500)
research/optimal_power_flow.py   → OPF analysis
research/diagnose_grid.py       → holistic grid diagnostics
```

---

## Data Schema

### Primary training target: `Island_Load_MW` (Ko Tao, ~6.7 MW base)

`island_grid.parquet` columns:
- `Island_Load_MW` — Ko Tao load (MW), stable, flat diurnal
- `Dry_Bulb_Temp` — 25.8–28.9°C (calibrated, tropical island)
- `Rel_Humidity` — 74–84%
- `Solar_Irradiance` — 0–1050 W/m²
- `Carbon_Intensity` — 400–484 gCO2/kWh (Thai grid)
- `Market_Price` — 35–120 THB/kWh
- `Tourist_Index` — 0.2–1.0
- `BESS_SoC_Pct` — 20–80%
- `Phangan_Load_MW` — Ko Phangan load (~18–26 MW, moderate volatility)
- `Phangan_Temp`, `Phangan_Circuit_MW`, `Phangan_SoC_Pct`
- `Samui_Load_MW` — Ko Samui load (~55–95 MW, high volatility)
- `Samui_Temp`, `Samui_Circuit_MW`, `Samui_SoC_Pct`

Engineered features (added by `preprocess.py`):
- `Load_Lag_1h`, `Load_Lag_24h`, `Load_Lag_168h`
- `Load_Roll_Mean_3h/6h`, `Load_Roll_Std_3h/6h`
- `Heat_Index` = Dry_Bulb_Temp × Rel_Humidity / 100
- `Temp_Roll_Mean_3h/6h`, `Humid_Roll_Mean_3h/6h`, `Temp_Gradient`
- `Hour_of_Day`, `Day_of_Week`, `Is_High_Season`, `Is_Thai_Holiday`, `Is_Songkran`
- `Max_Capacity_MW`, `Headroom_MW`
- Cluster spatial lags: `Phangan_Load_Lag_1h`, `Phangan_Load_Roll_Mean_3h/6h`, `Samui_Load_Lag_1h`, `Samui_Load_Roll_Mean_3h/6h`

**Scaling:** StandardScaler on weather/exogenous only. Load, lags, rolling load, calendar flags excluded.

---

## Grid Topology (pandapower 6-bus model)

```
Khanom (slack, 1.00 p.u.)
  └─ HVDC Koh Samui Connector ×2  (23.42 km, ⚡ bottleneck)
       └─ Samui HVDC Terminus
            └─ Samui HVDC–Samui3 Ring ×2  (8.13 km)
                 └─ Samui 3 Substation
                      └─ Koh Samui Export Seg1 ×2  (14.17 km)
                           └─ Samui Transition Sub
                                └─ Koh Samui Export Seg2 ×2  (15.57 km)
                                     └─ Ko Samui Distribution Sub  [load: 70% Samui]
                                          └─ Samui–Phangan Cable ×2  (30 km)
                                               └─ Ko Phangan  [load: Phangan]
                                                    └─ Phangan–Tao Cable  (40 km)
                                                         └─ Ko Tao  [load: Tao]
```

Cable type: 115 kV XLPE submarine, 630 mm² Cu — r=0.047 Ω/km, x=0.100 Ω/km, c=150 nF/km, max_i=530 A.
Shunt reactors: Samui dist 15 Mvar, Phangan 22 Mvar, Ko Tao 28 Mvar.

---

## Island Load Profiles

| Island | Base | Peak | Std dev | Character |
|---|---|---|---|---|
| Ko Tao | 6.7 MW | 7.7 MW | 0.22 MW | Stable, AC-dominated, flat diurnal |
| Ko Phangan | 18 MW | 26 MW | 1.9 MW | Moderate, Full Moon Party spikes |
| Ko Samui | 55 MW | 95 MW | 7.5 MW | Volatile, hotel/airport/commercial |

---

## Performance Targets (PEA — from config.yaml)

| Metric | Target | 
|---|---|
| MAPE | < 10.0% |
| R² | > 0.85 |
| MAE | < 0.75 MW |
| Fuel savings | > 22% |

---

## Real Data Status

- `data/processed/ko_tao_grid_2023_locked.parquet` — real Ko Tao proxy (calibrated from KIREIP + OpenMeteo)
- `data/processed/ko_tao_grid_calibrated.parquet` — multi-year calibrated synthetic
- `data/raw/kireip_proxy.parquet` — King Island BESS-Diesel proxy (used by `optimizer/run_optimization.py`)
- `data/raw/nrel_perform/` — NREL BA-level solar/wind/load actuals 2018
- `data/raw/public_datasets/` — district microgrid load+weather (US, 2012)
- `data/raw/ko_tao_network.geojson` — Ko Tao network topology for map visualization
- `data/raw/raw_tourism_samui.csv` — Samui tourism seasonality data
- Real PEA SCADA: **not yet integrated** — requires MOU with PEA

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/stream/telemetry` | Ingest 1 row; returns forecast when 48h buffer full |
| POST | `/stream/actual` | Record actual vs forecast → updates live MAPE |
| GET | `/stream/metrics` | Live MAE/RMSE/MAPE |
| POST | `/forecast` | Batch 24h forecast (requires full 48h history) |
| POST | `/warnings` | Early warning check (6h lookahead, pandapower AC power flow) |
| GET | `/health` | Liveness + buffer status |
| GET | `/metrics` | Last `evaluation_report.json` |

Streaming state: `StreamingEngine` persists telemetry buffer and metrics to SQLite (`api_state.db`), surviving API restarts.

---

## Frontend (Next.js 16)

App Router pages under `frontend/src/app/`:
- `/dashboard` — main overview
- `/forecast` — forecast visualization
- `/map` — geographic map view (Leaflet/Mapbox)
- `/meter` — smart meter simulator
- `/resilience` — resilience analysis
- `/topology` — grid topology viewer
- `/vpp` — virtual power plant
- `/lpc` — load profile charts
- `/adr` — automated demand response

Components: `dashboard/`, `console/`, `maps/`, `meters/`, `simulator/`, `providers/` (NetworkProvider, SimulatorProvider), `ui/` (GlobalNav, shared UI).

---

## Colab CLI Training

Training runs on Colab T4 GPU via `colab-cli`. All model artifacts are downloaded back after training.

### Workflow

```bash
# 1. Prepare data locally first
just generate && just preprocess

# 2. Assign T4 server (1.76 CCU/hr, token expires 60min)
colab-cli server assign --variant GPU --accelerator T4 --name gridtokenx

# 3. Build tar + install deps + run training (single command)
just colab-train

# 4. Download artifacts manually (colab-cli has no download command)
colab-cli server run bash -lc 'base64 /content/gridtokenx/models/tcn.pt' | base64 -d > models/tcn.pt
colab-cli server run bash -lc 'base64 /content/gridtokenx/models/lgbm.pkl' | base64 -d > models/lgbm.pkl
colab-cli server run bash -lc 'base64 /content/gridtokenx/models/meta_learner.pkl' | base64 -d > models/meta_learner.pkl
colab-cli server run bash -lc 'base64 /content/gridtokenx/data/processed/scaler.pkl' | base64 -d > data/processed/scaler.pkl
colab-cli server run bash -lc 'cat /content/gridtokenx/results/evaluation_report.json' > results/evaluation_report.json

# 5. Release server when done
colab-cli server rm gridtokenx
```

### What `just colab-train` does

1. Tars project (excludes `.venv`, `nrel_perform`, `public_datasets`, `node_modules`, existing model files)
2. Uploads tar to `/content/gridtokenx.tar.gz`
3. Extracts + installs: `lightgbm torch pandas numpy scikit-learn pyarrow pyyaml mlflow optuna`
4. Runs `colab_train.py`: preprocess → lgbm → tcn → hybrid → evaluate

### Key notes

- `COLAB_TRAIN=1` env var is set by `colab_train.py` — disables `mlflow.sklearn.log_model` (causes scikit-learn version conflict on Colab's Python 3.12)
- `colab-cli file` has no `download` subcommand — use `base64` pipe workaround above
- Token expires after 60 min — re-run `colab-cli server assign` if it expires mid-training
- Available accelerators: T4 (1.76 CCU/hr), L4 (4.82), A100 (11.77), H100 (14.43)
- Check balance: `colab-cli server ls --available`

---

## Known Issues / TODO

1. **No API authentication** — all endpoints open.
2. **No drift detection** — model accuracy degrades silently after deployment.
3. **Real PEA SCADA not integrated** — requires MOU with PEA. Onboarding pipeline ready (`just pea-onboard`).

---

## Conventions

- Use `just` recipes, not raw `python` calls.
- Val/test always come from real data (`ko_tao_grid_2023_locked.parquet`) when available. Synthetic only augments training.
- Cluster columns (`Phangan_*`, `Samui_*`) are excluded from StandardScaler — only present in synthetic training rows.
- LightGBM FEATURES list now includes cluster spatial lags (`Phangan_Load_Lag_1h`, `Samui_Load_Lag_1h`, etc.) — missing columns are zero-filled gracefully.
- MLflow experiment names: `GridTokenX_LGBM` (LightGBM training), `GridTokenX_API` (serving), `GridTokenX` (general training).
- Do not push directly to `main`. Do not use `git push --force`.

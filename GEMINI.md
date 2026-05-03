# GridTokenX: Predictive Intelligence for Islanded Microgrids

GridTokenX is a research and deployment platform for high-fidelity microgrid forecasting and dispatch optimization, specifically tuned for the **Ko Tao-Phangan-Samui** island cluster in Thailand.

## Project Overview
The system solves two primary challenges: **bottleneck congestion** (at the mainland 115 kV link) and **diesel generation efficiency**. It utilizes a hybrid meta-learning architecture to achieve a target MAPE of < 10% (historically < 3% in synthetic benchmarks) for the Ko Tao load profile.

### Core Architecture
- **Forecasting:** Hybrid ensemble using **Temporal Convolutional Networks (TCN)** for sequence modeling and **LightGBM** for tabular weather/holiday correlations, blended by a **Ridge Meta-Learner**.
- **Optimization:** MILP-based dispatch for Diesel coordination (Ko Tao has no BESS). ISCA metaheuristic optimizer also available.
- **Grid Physics:** pandapower 6-bus 115 kV model and PyPSA for power flow analysis, OPF, and N-1 contingency simulation.
- **Monitoring:** Integrated MLflow for experiment tracking and a FastAPI serving layer for real-time telemetry ingestion with SQLite-backed streaming state.

### Technology Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.11 (`uv` + venv) |
| ML | PyTorch (TCN), LightGBM, scikit-learn (Ridge meta-learner) |
| Hyperparameter search | Optuna |
| Experiment tracking | MLflow (SQLite backend, `mlflow.db`) |
| Grid physics | pandapower, PyPSA |
| API | FastAPI + uvicorn |
| Frontend | Next.js 16, TypeScript, Tailwind CSS 4, Recharts, Leaflet/Mapbox |
| Task runner | `just` (see `justfile`) |
| Containerization | Docker + docker-compose |

## Building and Running

### Prerequisites
- Python 3.11+
- `uv` for dependency management
- `just` command runner
- Node.js (for frontend)

### Key Commands
The project uses `just` for most tasks. Run `just` to see all available recipes.

| Category | Command | Description |
| :--- | :--- | :--- |
| **Setup** | `just setup` | Install dependencies via `uv sync`. |
| **Pipeline** | `just train` | Run full pipeline: Preprocess → Train LGBM/TCN → Fit Meta-Learner. |
| **Execution** | `just generate` | Generate synthetic multi-year research dataset. |
| **Execution** | `just preprocess` | Run feature engineering (Heat Index, lags, seasonal indices). |
| **Training** | `just lgbm` | Train LightGBM model only. |
| **Training** | `just tcn` | Train TCN model only. |
| **Training** | `just hybrid` | Train/fine-tune the meta-learner. |
| **Tuning** | `just tune` | Run Optuna hyperparameter search (default 50 trials). |
| **Evaluation** | `just eval` | Benchmark forecast + dispatch vs PEA targets. |
| **Serving** | `just api` | Start the FastAPI backend (port 8000). |
| **Frontend** | `just frontend` | Start the Next.js dashboard (port 3000). |
| **Dev** | `just dev` | Start full dev stack (API + Frontend concurrently). |
| **Simulate** | `just simulate` | Stream test data through API and print live RMSE/MAE/MAPE. |
| **Optimize** | `just optimize` | Run PEA MILP dispatch optimization (7-day test). |
| **Onboard** | `just pea-onboard` | Onboard real PEA SCADA data with calibration. |
| **Backtest** | `just pea-backtest` | Read-only backtest on PEA SCADA data (no refit). |
| **Docker** | `just up` / `just down` | Build and start/stop all Docker services. |
| **Colab** | `just colab-train` | Upload project and run full training on Colab T4 GPU. |
| **Clean** | `just clean` | Remove generated artifacts (parquets, models, reports). |

### Strategy & Commissioning

| Command | Description |
| :--- | :--- |
| `just backtest-12m` | 12-month backtest (May 2025 – Apr 2026). |
| `just stress-test` | N-1 contingency stress test (mainland cable failure). |
| `just cluster-test` | Cluster-wide ADMM bottleneck test. |
| `just stochastic-test` | Monte Carlo stochastic resilience test (N=500). |
| `just opf-test` | Optimal Power Flow (OPF) analysis. |
| `just diagnose` | Generate holistic diagnostic report. |
| `just report` | Run all commissioning tests above and generate dashboard. |

## Testing and Validation
- **Unit Tests:** Run `just test` (executes `pytest tests/`).
- **Test Suites:** `test_api.py`, `test_pipeline.py`, `test_preprocess.py`, `test_dispatch.py`, `test_early_warning.py`.
- **N-1 Contingency:** `just stress-test` simulates mainland cable failure survival.
- **Cluster ADMM:** `just cluster-test` validates multi-island coordination.
- **Monte Carlo:** `just stochastic-test` runs 500-sample resilience analysis.
- **Backtesting:** `just backtest-12m` evaluates performance over a full seasonal cycle.

## Development Conventions

### Code Structure
- `api/` — FastAPI serving logic (`serve.py`), real-time streaming simulator (`rt_simulator.py`).
- `data/` — Synthetic generation (`generate_dataset.py`), preprocessing (`preprocess.py`), real data calibration (`calibrate_with_real_data.py`), PEA onboarding (`pea_onboard.py`), and external data fetchers.
- `models/` — TCN (`tcn_model.py`), LightGBM (`lgbm_model.py`), Hybrid Meta-Learner (`hybrid_pipeline.py`), device detection (`device.py`).
- `optimizer/` — Dispatch optimization (`dispatch.py`), PEA MILP dispatch (`pea_dispatch_opt.py`), ISCA metaheuristic (`isca.py`), Optuna tuner (`tune.py`), ADMM resilience (`admm_resilience.py`), early warning system (`early_warning.py`).
- `research/` — Backtest engine, contingency analysis, cluster stress test, Monte Carlo engine, OPF analysis, pandapower/PyPSA grid models, SCADA simulator, sensitivity analysis, grid diagnostics.
- `frontend/` — Next.js 16 dashboard with pages: Dashboard, Forecast, Map, Meter, Resilience, Topology, VPP, LPC, ADR.
- `results/` — Evaluation reports and commissioning dashboard plots.
- `docs/` — Planning docs, PEA SCADA integration guide, validation plan, single-line diagram.
- `notebooks/` — Jupyter analysis notebook.

### Key Practices
- **MLflow Tracking:** All training runs logged to MLflow. Experiment names: `GridTokenX` (training), `GridTokenX_API` (serving). Local UI on port 5000 via `docker-compose`.
- **Subprocess Safety:** On macOS, LightGBM inference is run in isolated subprocesses (see `models/hybrid_pipeline.py`, `evaluate.py`) to avoid OpenMP conflicts with PyTorch.
- **Config Management:** Use `config.yaml` for grid parameters (diesel ratings, BESS capacity), training hyperparameters, cluster settings, and PEA targets.
- **Type Safety:** Use Pydantic schemas in `api/serve.py` for all telemetry and forecast requests.
- **Streaming State:** `StreamingEngine` persists telemetry buffer and metrics to SQLite (`api_state.db`), surviving API restarts.
- **Data Splits:** Val/test from real data (`ko_tao_grid_2023_locked.parquet`) when available. Synthetic only augments training. Cluster columns (`Phangan_*`, `Samui_*`) excluded from StandardScaler.
- **Git:** Do not push directly to `main`. Do not use `git push --force`.

### API Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| POST | `/stream/telemetry` | Ingest 1 row; returns forecast when 48h buffer full |
| POST | `/stream/actual` | Record actual vs forecast → updates live MAPE |
| GET | `/stream/metrics` | Live MAE/RMSE/MAPE |
| POST | `/forecast` | Batch 24h forecast (requires full 48h history) |
| POST | `/warnings` | Early warning check (6h lookahead, pandapower AC power flow) |
| GET | `/health` | Liveness + buffer status |
| GET | `/metrics` | Last `evaluation_report.json` |

## Grid Topology Context
The Ko Tao load (primary target) is connected radially from Ko Phangan via a 33 kV XLPE line. A critical bottleneck exists at the 115 kV mainland connector to Samui (from Khanom Power Station, 970 MW). Proactive diesel dispatch is required when Tao load exceeds available excess power during bottleneck periods. Ko Tao has **10 MW diesel** backup and **no local BESS**. The grid is modeled as a 6-bus pandapower network with 115 kV XLPE submarine cables (630 mm² Cu).

### Island Load Profiles

| Island | Base | Peak | Character |
| :--- | :--- | :--- | :--- |
| Ko Tao | ~6.7 MW | ~7.7 MW | Stable, AC-dominated, flat diurnal |
| Ko Phangan | ~18 MW | ~26 MW | Moderate, Full Moon Party spikes |
| Ko Samui | ~55 MW | ~95 MW | Volatile, hotel/airport/commercial |

# GridTokenX — AGENT.md

AI agent context for the Ko Tao-Phangan-Samui Predictive Intelligence codebase.

---

## Project Purpose

Physics-informed AI forecasting and dispatch system for three islanded microgrids connected to the Thai mainland via 115 kV XLPE submarine cables. Primary target: Ko Tao load forecast (MAPE < 10%, historically < 3% on synthetic benchmarks). Secondary: multi-island coordinated MILP dispatch optimization.

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
just optimize      # Coordinated Cluster MILP optimization (7-day test)
just api           # start FastAPI on :8000
just frontend      # start Next.js dashboard on :3000
just dev           # start full dev stack (API + Frontend concurrently)
just simulate      # replay test data through live API, print live metrics
just pea-full      # run full PEA integration + onboarding pipeline
just pea-onboard   # multi-target calibrate to real PEA SCADA ground truth
just pea-backtest  # read-only audit on PEA ground truth
just up / just down # docker-compose up/down (API + frontend + MLflow)

# 2026 Strategy & Commissioning
just backtest-12m      # 12-month backtest (May 2025 – Apr 2026)
just stress-test       # N-1 contingency stress test
just cluster-test      # Cluster-wide Coordinated MILP test
just stochastic-test   # Monte Carlo stochastic resilience (N=500)
just opf-test          # Optimal Power Flow analysis
just diagnose          # holistic diagnostic report
just report            # run all commissioning tests + generate dashboard
```

---

## Architecture

```
data/generate_dataset.py    → data/processed/island_grid.parquet  (3-island high-fidelity)
data/preprocess.py          → multi-target engineering + MSTL KMB decomposition
models/lgbm_model.py        → models/lgbm.pkl (Multi-Target)
models/tcn_model.py         → models/tcn.pt (Multi-Target)
models/hybrid_pipeline.py   → models/meta_learner.pkl (Parallel Ridge)
evaluate.py                 → results/evaluation_report.json
optimizer/run_optimization.py → Coordinated Cluster MILP runner
api/serve.py                → Multi-island real-time serving
optimizer/pea_dispatch_opt.py → Coordinated Cluster MILP (Radial Chain Aware)
research/pandapower_model.py → 6-bus 115 kV / 33 kV high-fidelity physics
research/cluster_stress_test.py → MILP verification on Songkran surge window
```

---

## Data Schema

### Multi-Target Monitoring
1. **Ko Samui Load** — High volatility, hotel/commercial driven.
2. **Ko Phangan Load** — Moderate volatility, Songkran/Full Moon spikes.
3. **Ko Tao Load** — Target load for local diesel commitment.
4. **KMB 115 kV Remaining Capacity** — MSTL-decomposed bottleneck signal.

Engineered features:
- `Load_Lag_1h/24h` for all islands.
- `KMB_Trend`, `KMB_Seasonal`, `KMB_Resid` (from MSTL).
- `Heat_Index`, `Temp_Roll_Mean`, `Temp_Gradient`.
- `Hour_of_Day`, `Day_of_Week`, `Is_High_Season`, `Is_Thai_Holiday`.

---

## Grid Topology (High-Fidelity Radial Chain)

```
Khanom (EGAT Slack)
  └─ KMA วงจร 2 / KMB วงจร 3 (Bottom neck) / 33kV XLPE+Oil  [4 cables, ~174 MW cap]
       └─ Koh Samui (Substations 1, 2, 3)  [Assets: 50 MWh BESS (on KMB), 3x Mobile Diesel]
            └─ Samui–Phangan 33 kV XLPE  [30 MW limit]
                 └─ Koh Phangan  [Assets: AVR, total radial dependency]
                      └─ Phangan–Tao 33 kV XLPE  [16 MW limit]
                           └─ Koh Tao  [Assets: 10 MW Diesel Plant, AVR]
```

**Optimization:** `cluster_optimize` MILP proactively manages these segment limits, committing Tao diesel when the Samui-Phangan or Phangan-Tao segments saturate.

---

## Performance Targets (PEA — from config.yaml)

| Metric | Target | 
|---|---|
| MAPE | < 10.0% |
| R² | > 0.85 |
| MAE | < 0.75 MW |
| Fuel savings | > 22% |

---

## Colab CLI Training

Training runs on Colab L4/T4 GPU. Multi-target pipeline generates `.pt` and `.pkl` artifacts.

```bash
just colab-train
# Follow manual download instructions in colab_train.py if needed
```

---

## Known Issues / TODO

1. **PEA AWS Sandbox Integration** — Onboarding scripts ready (`just pea-full`), pending raw file placement in `data/raw/pea_aws_sandbox`.
2. **BESS Logic** — Samui BESS currently optimized for KMB bottleneck; Phangan remains vulnerable with zero local generation/storage.
3. **API Auth** — Missing security layer for production telemetry ingestion.

# GridTokenX: Predictive Intelligence for Islanded Microgrids

GridTokenX is a research and deployment platform for high-fidelity microgrid forecasting and dispatch optimization, specifically tuned for the **Ko Tao-Phangan-Samui** island cluster in Thailand.

## 🎯 Project Overview
The system solves two primary challenges: **bottleneck congestion** (at the mainland 115 kV link) and **diesel generation efficiency**. It utilizes a hybrid meta-learning architecture to achieve a target MAPE of < 10% (historically < 3% in synthetic benchmarks) for the Ko Tao load profile.

### Core Architecture
- **Forecasting:** Hybrid ensemble using **Temporal Convolutional Networks (TCN)** for sequence modeling and **LightGBM** for tabular weather/holiday correlations, blended by a **Ridge Meta-Learner**.
- **Optimization:** MILP-based dispatch for Diesel/BESS (Battery Energy Storage System) coordination.
- **Monitoring:** Integrated MLflow for experiment tracking and a FastAPI serving layer for real-time telemetry ingestion.

## 🛠 Building and Running

### Prerequisites
- Python 3.11+
- `uv` for dependency management
- `just` command runner

### Key Commands
The project uses `just` for most tasks. Run `just` to see all available recipes.

| Category | Command | Description |
| :--- | :--- | :--- |
| **Setup** | `just setup` | Install dependencies and sync virtual environment. |
| **Pipeline** | `just train` | Run full pipeline: Preprocess → Train LGBM/TCN → Fit Meta-Learner. |
| **Execution** | `just generate` | Generate synthetic 4-year research dataset. |
| **Execution** | `just preprocess`| Run feature engineering (Heat Index, lags, seasonal indices). |
| **Training** | `just hybrid` | Train/fine-tune the meta-learner. |
| **Evaluation**| `just eval` | Benchmark forecast + dispatch vs PEA targets. |
| **Serving** | `just api` | Start the FastAPI backend (port 8000). |
| **Frontend** | `just frontend` | Start the Next.js dashboard (port 3000). |
| **Strategy** | `just report` | Run full 2026 strategy commissioning tests and generate reports. |

## 🧪 Testing and Validation
- **Unit Tests:** Run `just test` (executes `pytest tests/`).
- **N-1 Contingency:** `just stress-test` simulates mainland cable failure survival.
- **Backtesting:** `just backtest-12m` evaluates performance over a full seasonal cycle.

## 📝 Development Conventions

### Code Structure
- `api/`: FastAPI serving logic and real-time streaming engine.
- `data/`: Scripts for synthetic generation, preprocessing, and real data integration.
- `models/`: Implementation of TCN, LightGBM, and the Hybrid Meta-Learner.
- `optimizer/`: Dispatch optimization (MILP), tuning (Optuna), and early warning systems.
- `research/`: Specialized analysis tools for grid stability and commissioning.

### Key Practices
- **MLflow Tracking:** All training runs should be logged to MLflow. The local UI can be started if needed (default port 5000).
- **Subprocess Safety:** On macOS, LightGBM inference should be run in isolated subprocesses (see `models/hybrid_pipeline.py`) to avoid OpenMP conflicts with PyTorch.
- **Config Management:** Use `config.yaml` for grid parameters (diesel ratings, BESS capacity) and training hyperparameters.
- **Type Safety:** Use Pydantic schemas in `api/serve.py` for all telemetry and forecast requests.

## 🏝 Grid Topology Context
The Ko Tao load (primary target) is connected radially from Ko Phangan via a 33 kV XLPE line. A critical bottleneck exists at the 115 kV mainland connector to Samui. Proactive diesel dispatch is required when Tao load exceeds available excess power during bottleneck periods.

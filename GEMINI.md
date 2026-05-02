# GridTokenX — Ko Tao Microgrid Predictive Intelligence

This project is a high-fidelity research and deployment environment for AI-driven predictive dispatch in islanded microgrids, specifically focused on the Ko Tao-Phangan-Samui cluster.

## 🏗️ Architecture & Stack

The system utilizes a **Hybrid Meta-Learner** architecture to forecast island load and optimize dispatch:
1.  **Sequential Layer (TCN):** Temporal Convolutional Network for capturing long-term temporal patterns.
2.  **Tabular Layer (LightGBM):** Handles non-linear exogenous correlations (Temperature, Humidity, Tourist Index).
3.  **Meta-Learner (Ridge):** Blends predictions to achieve high precision (Target: MAPE < 10%).
4.  **Optimizer (ISCA/MILP):** Implements Improved Sine-Cosine Algorithm and MILP for proactive BESS/Diesel scheduling.

**Technologies:**
- **Backend:** Python 3.11, FastAPI, PyTorch, LightGBM, Optuna.
- **Frontend:** Next.js (TypeScript), Tailwind CSS (though Vanilla CSS is preferred for new components), Mapbox.
- **Infrastructure:** Docker, `just` (task runner), `uv` (package manager).

## 🚀 Key Commands

The project uses `just` for task automation. Ensure `just` and `uv` are installed.

### Setup & Data
- `just setup`: Initialize Python virtual environment using `uv`.
- `just generate`: Create synthetic 4-year research dataset.
- `just preprocess`: Run feature engineering (lags, rolling stats, heat index).

### Training & Evaluation
- `just train`: Execute full training pipeline (LGBM -> TCN -> Hybrid).
- `just eval`: Run comprehensive evaluation of forecast and dispatch logic.
- `just tune`: Perform hyperparameter optimization using Optuna.

### Serving & Simulation
- `just api`: Start the FastAPI forecast server (Port 8000).
- `just simulate`: Replay test data through the API for real-time performance tracking.
- `just up`: Launch full stack (API + Frontend) via Docker Compose.

### Deployment & Onboarding
- `just pea-onboard`: Calibrate models to real PEA SCADA telemetry.

## 📂 Project Structure

- `api/`: FastAPI service and real-time streaming simulation.
- `data/`: Data generation, preprocessing, and PEA integration logic.
    - `raw/`: External datasets and raw telemetry.
    - `processed/`: Training splits, scalers, and generated datasets.
- `models/`: Hybrid model implementations (LGBM, TCN, Meta-Learner).
- `optimizer/`: Dispatch logic, Early Warning System, and ISCA/MILP solvers.
- `frontend/`: Next.js dashboard for grid visualization and metrics.
- `docs/`: Master goals, integration plans, and architectural research.

## 🛠️ Development Conventions

- **Workflow:** Always follow the **Research -> Strategy -> Execution** lifecycle.
- **Task Runner:** Use `just` for all operations. Do not run python scripts directly if a recipe exists.
- **Data Integrity:** Real-world PEA data integration is handled via `data/pea_onboard.py` for distribution shift calibration.
- **Testing:** Verify changes using `just simulate` or `just eval` before deployment.

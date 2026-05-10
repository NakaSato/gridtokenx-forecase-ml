# GridTokenX — Ko Tao Microgrid ML Pipeline
# Install: brew install just

set dotenv-load

python := "uv run python"

# List available recipes
default:
    @just --list

# Run test suite
test *args='':
    {{python}} -m pytest tests/ -v --tb=short {{args}}

# ── Data ──────────────────────────────────────────────────────────────────────

# Generate synthetic dataset
generate:
    {{python}} data/generate_dataset.py

# Preprocess + feature engineering
preprocess:
    {{python}} data/preprocess.py

# ── Training ──────────────────────────────────────────────────────────────────

# Train LightGBM
lgbm:
    {{python}} models/lgbm_model.py

# Train TCN
tcn:
    {{python}} models/tcn_model.py

# Train hybrid meta-learner
hybrid:
    {{python}} models/hybrid_pipeline.py

# Full training pipeline
train: preprocess lgbm tcn hybrid

# ── Evaluation ────────────────────────────────────────────────────────────────

# Evaluate forecast + dispatch vs PEA targets
eval:
    {{python}} evaluate.py

# Run Coordinated Cluster MILP dispatch optimization (7-day test)
optimize:
    {{python}} optimizer/run_optimization.py

# ── API ───────────────────────────────────────────────────────────────────────

# Start MLflow server locally (port 5000)
mlflow:
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Start forecast API (port 8000)
api:
    {{python}} -m uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (port 3000)
frontend:
    cd frontend && npm run dev

# Start full dev stack (API + Frontend)
dev:
    @echo "🚀 Starting GridTokenX Dev Stack..."
    (trap 'kill 0' SIGINT; {{python}} -m uvicorn api.serve:app --port 8000 --reload & cd frontend && npm run dev)

# Stream test data through API and print live RMSE/MAE/MAPE
simulate rows="200":
    {{python}} api/rt_simulator.py --rows {{rows}}

# ── Colab ─────────────────────────────────────────────────────────────────────

# Upload project and run full training on Colab T4 GPU
colab-train:
    tar -czf /tmp/gridtokenx.tar.gz \
        --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
        --exclude='node_modules' --exclude='frontend' \
        --exclude='data/raw' \
        --exclude='mlruns' \
        --exclude='mlflow_colab.db' \
        --exclude='models/*.pkl' --exclude='models/*.pt' .
    colab-cli file upload /tmp/gridtokenx.tar.gz /content/gridtokenx.tar.gz
    colab-cli server run bash -lc \
        'mkdir -p /content/gridtokenx && tar -xzf /content/gridtokenx.tar.gz -C /content/gridtokenx && pip install -q lightgbm torch pandas numpy scikit-learn pyarrow pyyaml mlflow optuna && cd /content/gridtokenx && python colab_train.py'
    @echo "📥 Downloading artifacts..."
    mkdir -p /tmp/gridtokenx_dl
    colab-cli server run bash -lc "tar -cz -C /content/gridtokenx models results | base64" > /tmp/artifacts.b64
    # Clean up base64 output (remove any non-base64 lines if colab-cli adds headers)
    grep -E '^[A-Za-z0-9+/=]+$' /tmp/artifacts.b64 > /tmp/artifacts_clean.b64
    base64 -D -i /tmp/artifacts_clean.b64 -o /tmp/artifacts.tar.gz
    tar -xzf /tmp/artifacts.tar.gz -C .
    @echo "✅ Artifacts synced to local models/ and results/"

# ── PostGIS ────────────────────────────────────────────────────────────────────

# Start PostGIS container only
postgis-up:
    docker compose up postgis -d --wait

# Load power_plants GeoJSON into PostGIS
postgis-load: postgis-up
    uv run --extra geo python data/load_postgis.py

# Open psql shell into PostGIS
postgis-shell:
    docker compose exec postgis psql -U ${POSTGRES_USER:-gridtokenx} -d ${POSTGRES_DB:-gridtokenx_geo}

# ── Docker ────────────────────────────────────────────────────────────────────

# Build and start all services
up:
    docker compose up --build -d

# Stop all services
down:
    docker compose down

# Tail API logs
logs:
    docker compose logs -f api

# ── Hyperparameter tuning ─────────────────────────────────────────────────────

# Run Optuna hyperparameter search
tune trials="50":
    {{python}} optimizer/tune.py --trials {{trials}}

# ── PEA Onboarding (AWS Sandbox) ──────────────────────────────────────────────

# Aggregate raw PEA AWS Sandbox files into multi-island ground truth
pea-integrate dir="data/raw/pea_aws_sandbox":
    {{python}} data/integrate_pea_raw.py --dir {{dir}}

# Onboard real PEA SCADA data: multi-target distribution check + meta-learner refit
pea-onboard input="data/processed/pea_ground_truth.parquet" calib="3":
    {{python}} data/pea_onboard.py --input {{input}} --calib-months {{calib}}

# Backtest only (no model refit) on multi-target ground truth
pea-backtest input="data/processed/pea_ground_truth.parquet":
    {{python}} data/pea_onboard.py --input {{input}} --no-refit

# Run full integration and onboarding sequence
pea-full dir="data/raw/pea_aws_sandbox": pea-integrate pea-onboard

# ── Misc ──────────────────────────────────────────────────────────────────────

# Set up Python virtual environment
setup:
    uv sync

# Remove generated artifacts
clean:
    rm -f data/processed/*.parquet data/processed/*.pkl
    rm -f models/*.pkl models/*.pt
    rm -f results/*.json results/*.png
    rm -f mlflow.db api_state.db
    find . -name "__pycache__" -type d -exec rm -rf {} +

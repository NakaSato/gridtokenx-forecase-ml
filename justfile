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

# Run PEA MILP dispatch optimization (7-day test)
optimize:
    {{python}} optimizer/run_optimization.py

# ── API ───────────────────────────────────────────────────────────────────────

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
        --exclude='mlruns' --exclude='mlflow_colab.db' \
        --exclude='models/*.pkl' --exclude='models/*.pt' .
    colab-cli file upload /tmp/gridtokenx.tar.gz /content/gridtokenx.tar.gz
    colab-cli server run bash -lc \
        'mkdir -p /content/gridtokenx && tar -xzf /content/gridtokenx.tar.gz -C /content/gridtokenx && pip install -q lightgbm torch pandas numpy scikit-learn pyarrow pyyaml mlflow optuna && cd /content/gridtokenx && python colab_train.py 2>&1 | tee /tmp/train.log'

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

# ── PEA Onboarding ───────────────────────────────────────────────────────────

# Onboard real PEA SCADA data: distribution check, scaler refit, meta-learner refit, backtest
pea-onboard input="data/raw/pea_telemetry_raw.csv" calib="3":
    {{python}} data/pea_onboard.py --input {{input}} --calib-months {{calib}}

# Backtest only (no model refit) — safe for read-only audit
pea-backtest input="data/raw/pea_telemetry_raw.csv":
    {{python}} data/pea_onboard.py --input {{input}} --no-refit

# ── 2026 Strategy & Commissioning ──────────────────────────────────────────

# Run 12-month backtest (May 2025 - Apr 2026)
backtest-12m:
    PYTHONPATH=. {{python}} results/backtest_12m.py

# Run N-1 contingency stress test (April 2026)
stress-test:
    {{python}} optimizer/contingency_analysis.py

# Run Cluster-wide ADMM bottleneck test
cluster-test:
    {{python}} optimizer/cluster_stress_test.py

# Generate commissioning report and dashboard
report: backtest-12m stress-test cluster-test
    {{python}} results/plot_2026_strategy.py
    @echo "✅ Commissioning report and dashboard ready in results/"

# ── Misc ──────────────────────────────────────────────────────────────────────

# Set up Python virtual environment
setup:
    uv sync

# Remove generated artifacts
clean:
    rm -f data/processed/train.parquet data/processed/val.parquet data/processed/test.parquet data/processed/scaler.pkl
    rm -f models/lgbm.pkl models/tcn.pt models/meta_learner.pkl
    rm -f results/evaluation_report.json results/pea_optimization_report.json

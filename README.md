# GridTokenX: Predictive Intelligence Research Lab
**Ko Tao-Phangan-Samui AI Forecasting & Dispatch Research**

[![Training Accuracy](https://img.shields.io/badge/Best_MAPE-1.28%25-green.svg)](#)
[![R2 Stability](https://img.shields.io/badge/R2_Score-0.985-blue.svg)](#)
[![GPU Accelerated](https://img.shields.io/badge/Training-GPU_MPS_CUDA-blueviolet.svg)](#)

## Research Objective
This codebase provides a high-fidelity environment for training and benchmarking predictive AI models for islanded microgrids. It is specifically tuned to solve the **bottleneck congestion** and **diesel efficiency** problems of the Ko Tao-Phangan-Samui cluster using a hybrid TCN-LGBM architecture.

## AI Model Architecture (The Hybrid Meta-Learner)
The core of this project is a multi-stage forecasting engine:

```mermaid
graph TD
    subgraph "Data Acquisition & Processing"
        DS1[Synthetic Data Gen] --> PRE[Preprocessing]
        DS2[Thira/KIREIP/NREL Datasets] --> PRE
        PRE --> FE[Feature Engineering: Heat Index, Lags, Seasonal Indices]
    end

    subgraph "Hybrid Meta-Learner Architecture"
        FE --> TCN[Sequential Layer: TCN<br/>Causal Dilated Convolutions]
        FE --> LGBM[Tabular Layer: LightGBM<br/>Weather & Exogenous Correlations]
        
        TCN --> META[Meta-Learner: Ridge Blending]
        LGBM --> META
        
        META --> OUT[Load Forecast<br/>MAPE < 2.65%]
    end

    subgraph "Optimization & Evaluation"
        OPT[Optuna Tuner] -.-> |Hyperparams| TCN
        OPT -.-> |Hyperparams| LGBM
        OUT --> EVAL[Evaluation Engine<br/>Benchmark vs. PEA Targets]
    end

    subgraph "Application Layer"
        EVAL --> DISPATCH[Proactive Diesel/BESS Dispatch]
    end

    style META fill:#f96,stroke:#333,stroke-width:2px
    style OUT fill:#dfd,stroke:#333,stroke-width:2px
```

1. **Sequential Layer (TCN):** A Temporal Convolutional Network with causal dilated convolutions. It excels at capturing the long-term patterns of tourism-driven load curves.
2. **Tabular Layer (LightGBM):** Handles non-linear correlations between dry-bulb temperature, humidity (Heat Index), and peak A/C demand.
3. **Meta-Learner (Ridge):** A blending layer that intelligently weights the TCN and LGBM outputs to achieve the engineering target of **MAPE < 2.65%**.

## Experiment Tracking & Observability
We utilize **MLflow** for rigorous experiment governance and real-time inference profiling.

```mermaid
graph TD
    subgraph "MLflow Lifecycle & Governance"
        DB[(mlflow.db <br/>SQLite Local)]
        UI[MLflow UI <br/>Port 5000]
        REG[Model Registry]
        TRK[Tracking Server]
        
        DB --- TRK
        TRK --- UI
        TRK --- REG
    end

    subgraph "Training Phase (Research)"
        TUNE[optimizer/tune.py <br/>Optuna Search] --> |Log Params/Metrics| TRK
        LGBM_T[models/lgbm_model.py] --> |Log Model/Metrics| TRK
        TCN_T[models/tcn_model.py] --> |Log Model/Metrics| TRK
        HYB_T[models/hybrid_pipeline.py] --> |Log Meta-Learner| TRK
    end

    subgraph "Serving Phase (Production)"
        API[api/serve.py <br/>FastAPI] --> |"Load Models"| REG
        API --> |"@mlflow.trace"| TRK
    end
```

## Training Pipeline
To reproduce the research benchmarks, execute the following flow:

```bash
# 1. Generate 4-Year Synthetic Research Dataset
python data/generate_dataset.py

# 2. Preprocess & Feature Engineering
# (Calculates Heat Index, Lags, and Seasonal Tourist Indices)
python data/preprocess.py

# 3. Optimize Hyperparameters (Optuna)
# Automates search for filters, kernel sizes, and learning rates
python optimizer/tune.py

# 4. Train Hybrid Models
python models/lgbm_model.py
python models/tcn_model.py
python models/hybrid_pipeline.py

# 5. Evaluate vs. Real-World Benchmarks
python evaluate.py
```

## Benchmarking Datasets
This codebase supports benchmarking against real-world island telemetry:
- **Thira (Santorini):** Used for tourism-driven seasonality.
- **King Island (KIREIP):** Used for BESS-Diesel transition validation.
- **NREL PERFORM:** Used for solar-load coincidence research.

## Google Colab Integration
For high-speed GPU training, use the provided `colab_benchmark.ipynb` configuration. The system automatically detects CUDA/MPS hardware to accelerate the TCN training phase.

"""
Optuna hyperparameter optimization for LightGBM and TCN hybrid pipeline.
Optimizes for MAPE (accuracy) and training/inference time.
"""
import time
import yaml
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import os, sys

# Add root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from torch.utils.data import DataLoader
from models.tcn_model import TCN, WindowDataset
from models.lgbm_model import FEATURES as exog_cols, TARGET as target_col

import mlflow
import mlflow.sklearn

# Set experiment name
mlflow.set_experiment("GridTokenX_Hyperparameter_Optimization")

# Load Data once
train_df_all = pd.read_parquet("data/processed/train.parquet")
val_df_all   = pd.read_parquet("data/processed/val.parquet")

def objective(trial):
    with mlflow.start_run(nested=True):
        # 1. Access Data
        train_df = train_df_all.copy()
        val_df   = val_df_all.copy()

        # Handle missing cluster features in real data splits by filling with 0.0
        for col in exog_cols:
            if col not in train_df.columns:
                train_df[col] = 0.0
            if col not in val_df.columns:
                val_df[col] = 0.0
        
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        # 2. Suggest Hyperparameters
        # LightGBM
        lgbm_params = {
            "n_estimators": 500,
            "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("lgbm_leaves", 2, 256),
            "lambda_l1": trial.suggest_float("lgbm_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lgbm_l2", 1e-8, 10.0, log=True),
            "verbose": -1
        }
        
        # TCN
        tcn_params = {
            "filters": trial.suggest_int("tcn_filters", 32, 128),
            "kernel_size": trial.suggest_int("tcn_kernel", 2, 5),
            "layers": 4,
            "lr": trial.suggest_float("tcn_lr", 1e-4, 1e-2, log=True),
            "epochs": 3 # Reduced for speed
        }
        
        # Track parameters in MLflow
        mlflow.log_params(trial.params)

    start_time = time.time()

    # 3. Train LightGBM
    lgbm = lgb.LGBMRegressor(
        **{k.replace("lgbm_", ""): v for k, v in lgbm_params.items() if k != "verbose"},
        n_jobs=1,
        random_state=42
    )
    lgbm.fit(train_df[exog_cols], train_df[target_col])
    lgbm_preds = lgbm.predict(val_df[exog_cols])

    # 4. Train TCN
    from models.device import get_device
    device = get_device()
    
    window = cfg["model"]["tcn"]["window_size"]
    horizon = cfg["model"]["tcn"]["forecast_horizon"]
    
    train_ds = WindowDataset(train_df, window, horizon)
    val_ds   = WindowDataset(val_df,   window, horizon)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64)

    model = TCN(len(train_ds.X[0][0]), tcn_params["filters"], tcn_params["kernel_size"],
                tcn_params["layers"], horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcn_params["lr"])
    criterion = nn.MSELoss()

    for epoch in range(tcn_params["epochs"]):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    tcn_preds_list = []
    with torch.no_grad():
        for xb, _ in val_dl:
            tcn_preds_list.append(model(xb.to(device)).cpu().numpy())
    
    tcn_preds_raw = np.concatenate(tcn_preds_list)[:, 0]
    
    # 5. Hybrid Evaluation
    # TCN preds are for indices [window : window + len(tcn_preds_raw)]
    y_true = val_df["Island_Load_MW"].values[window : window + len(tcn_preds_raw)]
    lgbm_preds_aligned = lgbm_preds[window : window + len(tcn_preds_raw)]
    
    hybrid_preds = (lgbm_preds_aligned + tcn_preds_raw) / 2
    
    mape_val = mean_absolute_percentage_error(y_true, hybrid_preds)
    total_time = time.time() - start_time
    
    # Track metrics in MLflow
    mlflow.log_metric("mape", mape_val)
    mlflow.log_metric("inference_time_s", total_time)

    return mape_val, total_time

def run_optimization(n_trials=50):
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="ko_tao_grid_optimization"
    )
    study.optimize(objective, n_trials=n_trials)
    
    print("\nNumber of finished trials: ", len(study.trials))
    print("Pareto Front Trials:")
    
    for trial in study.best_trials:
        print(f"  Trial {trial.number}:")
        print(f"    Values: MAPE={trial.values[0]:.4f}, Time={trial.values[1]:.2f}s")
        print(f"    Params: {trial.params}")

    return study

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    
    study = run_optimization(args.trials)
    # Save best params to a file or config
    best_trial = study.best_trials[0]
    with open("results/best_hyperparams.yaml", "w") as f:
        yaml.dump(best_trial.params, f)

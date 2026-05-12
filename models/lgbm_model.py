"""
Multi-Target LightGBM model for tabular/exogenous feature forecasting.
Targets: tao_load_mw, cable_flow_mw, kmb_flow_mw.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
import mlflow
import mlflow.sklearn
from sklearn.multioutput import MultiOutputRegressor
from models.mlflow_utils import setup_mlflow
from models.schema import TAB_FEATURES as FEATURES, TARGETS

setup_mlflow()
mlflow.set_experiment("GridTokenX_LGBM")

# ── Feature Set ─────────────────────────────────
# Features and targets imported from schema.py


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def main():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    lc = cfg["model"]["lgbm"]
    
    train = pd.read_parquet("data/processed/train.parquet")
    val   = pd.read_parquet("data/processed/val.parquet")

    missing = [c for c in FEATURES if c not in train.columns]
    if missing:
        raise ValueError(f"CRITICAL: Dataframe missing required LGBM features: {missing}")

    X_tr, y_tr = train[FEATURES], train[TARGETS]
    X_val, y_val = val[FEATURES], val[TARGETS]

    base_model = lgb.LGBMRegressor(
        n_estimators=lc["n_estimators"],
        learning_rate=lc["learning_rate"],
        num_leaves=lc["num_leaves"],
        reg_alpha=lc.get("lambda_l1", 0.0),
        reg_lambda=lc.get("lambda_l2", 0.0),
        min_child_samples=lc["min_child_samples"],
        n_jobs=-1,
        random_state=42,
    )
    
    model = MultiOutputRegressor(base_model)

    with mlflow.start_run(run_name="lgbm_multi_target_aligned"):
        mlflow.log_params({
            "targets": TARGETS,
            "n_estimators": lc["n_estimators"],
            "learning_rate": lc["learning_rate"],
        })

        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        
        for i, target in enumerate(TARGETS):
            target_mape = mape(y_val[target].values, preds[:, i])
            mlflow.log_metric(f"val_mape_{target}", target_mape)
            print(f"Target {target:<25} | Val MAPE: {target_mape:.4f}%")

        avg_mape = mape(y_val.values, preds)
        mlflow.log_metric("val_mape_avg", avg_mape)
        print(f"\nAverage Val MAPE: {avg_mape:.4f}%")

        # Log and Register
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="lgbm_model",
            registered_model_name="GridTokenX_LGBM"
        )

    os.makedirs("models", exist_ok=True)
    with open("models/lgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved → models/lgbm.pkl")

if __name__ == "__main__":
    main()

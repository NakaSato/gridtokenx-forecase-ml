"""
Multi-Target LightGBM model for tabular/exogenous feature forecasting.
Targets: Samui Load, Phangan Load, Tao Load, KMB Capacity.
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
from sklearn.metrics import mean_absolute_error, r2_score
from models.mlflow_utils import setup_mlflow

setup_mlflow()
mlflow.set_experiment("GridTokenX_LGBM")

FEATURES = [
    "Dry_Bulb_Temp", "Rel_Humidity", "Solar_Irradiance", "Heat_Index",
    "Temp_Roll_Mean_3h", "Temp_Roll_Mean_6h",
    "Humid_Roll_Mean_3h", "Humid_Roll_Mean_6h", "Temp_Gradient",
    "Carbon_Intensity", "Market_Price",
    "Tourist_Index", "Is_High_Season",
    "Hour_of_Day", "Day_of_Week",
    "Is_Thai_Holiday", "Is_Songkran",
    "Island_Load_MW_Lag_1h", "Island_Load_MW_Lag_24h",
    "Phangan_Load_MW_Lag_1h", "Phangan_Load_MW_Lag_24h",
    "Samui_Load_MW_Lag_1h", "Samui_Load_MW_Lag_24h",
    "Samui_Circuit_MW_Lag_1h", "Samui_Circuit_MW_Lag_24h",
    "Max_Capacity_MW", "Headroom_MW",
    "KMB_Trend", "KMB_Seasonal", "KMB_Resid"
]

TARGETS = ["Samui_Load_MW", "Phangan_Load_MW", "Island_Load_MW", "Samui_Circuit_MW"]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    lc = cfg["model"]["lgbm"]
    
    train = pd.read_parquet("data/processed/train.parquet")
    val   = pd.read_parquet("data/processed/val.parquet")

    for col in FEATURES:
        if col not in train.columns: train[col] = 0.0
        if col not in val.columns: val[col] = 0.0

    X_tr, y_tr = train[FEATURES], train[TARGETS]
    X_val, y_val = val[FEATURES], val[TARGETS]

    from sklearn.multioutput import MultiOutputRegressor

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

    with mlflow.start_run(run_name="lgbm_multi_target"):
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
            print(f"Target {target:<20} | Val MAPE: {target_mape:.4f}%")

        avg_mape = mape(y_val.values, preds)
        mlflow.log_metric("val_mape_avg", avg_mape)
        print(f"\nAverage Val MAPE: {avg_mape:.4f}%")

    os.makedirs("models", exist_ok=True)
    with open("models/lgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved → models/lgbm.pkl")

if __name__ == "__main__":
    main()

"""
LightGBM model for tabular/exogenous feature forecasting.
Input:  data/processed/train.parquet, data/processed/val.parquet
Output: models/lgbm.pkl
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
    # Weather (from TMD forecast — available 72h ahead)
    "Dry_Bulb_Temp", "Rel_Humidity", "Solar_Irradiance", "Heat_Index",
    "Temp_Roll_Mean_3h", "Temp_Roll_Mean_6h",
    "Humid_Roll_Mean_3h", "Humid_Roll_Mean_6h", "Temp_Gradient",
    # Market/carbon (available from exchange)
    "Carbon_Intensity", "Market_Price",
    # Tourism proxy (calendar-derived, always available)
    "Tourist_Index", "Is_High_Season",
    # Calendar (always available)
    "Hour_of_Day", "Day_of_Week",
    "Is_Thai_Holiday", "Is_Songkran",
    # Load history (from SCADA — always available)
    "Load_Lag_1h", "Load_Lag_24h", "Load_Lag_168h",
    "Load_Roll_Mean_3h", "Load_Roll_Std_3h",
    "Load_Roll_Mean_6h", "Load_Roll_Std_6h",
    # Grid Dynamics (Stability & Capacity)
    "Max_Capacity_MW", "Headroom_MW",
    # Cluster spatial features (Neighbors' load)
    "Phangan_Load_Lag_1h", "Phangan_Load_Roll_Mean_3h", "Phangan_Load_Roll_Mean_6h",
    "Samui_Load_Lag_1h", "Samui_Load_Roll_Mean_3h", "Samui_Load_Roll_Mean_6h",
]
TARGET = "Island_Load_MW"

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    lc = cfg["model"]["lgbm"]
    
    # Override with best hyperparams if available
    best_path = "results/best_hyperparams.yaml"
    if os.path.exists(best_path):
        with open(best_path) as f:
            best = yaml.safe_load(f)
            print(f"Loading best hyperparameters from {best_path}...")
            # Map lgbm_lr -> learning_rate, etc.
            if "lgbm_lr" in best: lc["learning_rate"] = best["lgbm_lr"]
            if "lgbm_leaves" in best: lc["num_leaves"] = best["lgbm_leaves"]
            if "lgbm_l1" in best: lc["lambda_l1"] = best["lgbm_l1"]
            if "lgbm_l2" in best: lc["lambda_l2"] = best["lgbm_l2"]

    train = pd.read_parquet("data/processed/train.parquet")
    val   = pd.read_parquet("data/processed/val.parquet")

    # Handle missing cluster features in real data splits by filling with 0.0
    for col in FEATURES:
        if col not in train.columns:
            train[col] = 0.0
        if col not in val.columns:
            val[col] = 0.0

    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=lc["n_estimators"],
        learning_rate=lc["learning_rate"],
        num_leaves=lc["num_leaves"],
        reg_alpha=lc.get("lambda_l1", 0.0),
        reg_lambda=lc.get("lambda_l2", 0.0),
        min_child_samples=lc["min_child_samples"],
        n_jobs=-1,
        random_state=42,
    )
    with mlflow.start_run(run_name="lgbm_train"):
        mlflow.log_params({
            "n_estimators": lc["n_estimators"],
            "learning_rate": lc["learning_rate"],
            "num_leaves": lc["num_leaves"],
            "lambda_l1": lc.get("lambda_l1", 0.0),
            "lambda_l2": lc.get("lambda_l2", 0.0),
            "min_child_samples": lc["min_child_samples"],
        })

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(lc["early_stopping_rounds"], verbose=False),
                       lgb.log_evaluation(period=100)],
        )

        preds = model.predict(X_val)
        val_mape = mape(y_val, preds)
        val_mae  = mean_absolute_error(y_val, preds)
        val_r2   = r2_score(y_val, preds)

        mlflow.log_metrics({"val_mape": val_mape, "val_mae": val_mae, "val_r2": val_r2})
        from mlflow.models import infer_signature
        signature = infer_signature(X_val, preds)
        if not os.environ.get("COLAB_TRAIN"):
            mlflow.sklearn.log_model(model, "lgbm_model", signature=signature, input_example=X_val.iloc[:5])

        print(f"Val MAPE : {val_mape:.4f}%  (target <10.0%)")
        print(f"Val MAE  : {val_mae:.4f} MW")
        print(f"Val R²   : {val_r2:.4f}  (target >0.85)")

    os.makedirs("models", exist_ok=True)
    with open("models/lgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved → models/lgbm.pkl")

if __name__ == "__main__":
    main()

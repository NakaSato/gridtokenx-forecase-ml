"""
LightGBM model for tabular/exogenous feature forecasting.
Input:  data/train.parquet, data/val.parquet
Output: models/lgbm.pkl
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = [
    "Dry_Bulb_Temp", "Rel_Humidity", "Solar_Irradiance", "Wind_Speed",
    "Cloud_Cover", "Carbon_Intensity", "Market_Price", "Tourist_Index",
    "Circuit_Cap_MW", "Hour_of_Day", "Day_of_Week", "Is_Weekend",
    "Is_High_Season", "Heat_Index", "Time_Until_High_Cap",
    "Load_Lag_1h", "Load_Lag_24h", "Load_Lag_168h",
    "Load_Roll_Mean_3h", "Load_Roll_Std_3h",
    "Load_Roll_Mean_6h", "Load_Roll_Std_6h",
]
TARGET = "Island_Load_MW"

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    lc = cfg["model"]["lgbm"]

    train = pd.read_parquet("data/train.parquet")
    val   = pd.read_parquet("data/val.parquet")

    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=lc["n_estimators"],
        learning_rate=lc["learning_rate"],
        num_leaves=lc["num_leaves"],
        min_child_samples=lc["min_child_samples"],
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(lc["early_stopping_rounds"], verbose=False),
                   lgb.log_evaluation(period=100)],
    )

    preds = model.predict(X_val)
    print(f"Val MAPE : {mape(y_val, preds):.4f}%  (target <2.65%)")
    print(f"Val MAE  : {mean_absolute_error(y_val, preds):.4f} MW")
    print(f"Val R²   : {r2_score(y_val, preds):.4f}  (target >0.97)")

    os.makedirs("models", exist_ok=True)
    with open("models/lgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved → models/lgbm.pkl")

if __name__ == "__main__":
    main()

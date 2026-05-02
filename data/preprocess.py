"""
Feature engineering, train/val/test split, and normalization.
Input:  data/processed/island_grid.parquet
Output: data/processed/train.parquet, data/processed/val.parquet, data/processed/test.parquet, data/processed/scaler.pkl
"""
import os
import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dc = cfg["data"]
    freq = dc.get("frequency", "h")
    sph = 4 if freq == "15min" else 1
    d = df.copy()

    # ── Stability Features ──
    # Headroom = Capacity - Load (Predicting difference is more stable)
    if "Circuit_Cap_MW" in d.columns:
        d["Headroom_MW"] = d["Circuit_Cap_MW"] - d["Island_Load_MW"]
    
    # Historical Max Capacity as a feature
    d["Max_Capacity_MW"] = d["Circuit_Cap_MW"].expanding().max()

    # ── Temporal Features ──
    # Lag features
    for lag_h in [1, 24, 168]:
        lag_steps = lag_h * sph
        d[f"Load_Lag_{lag_h}h"] = d["Island_Load_MW"].shift(lag_steps)

    # Rolling stats
    for w_h in [3, 6]:
        w_steps = w_h * sph
        d[f"Load_Roll_Mean_{w_h}h"] = d["Island_Load_MW"].shift(1).rolling(w_steps).mean()
        d[f"Load_Roll_Std_{w_h}h"]  = d["Island_Load_MW"].shift(1).rolling(w_steps).std()

    # Cluster features
    for island in ["Phangan", "Samui"]:
        col = f"{island}_Load_MW"
        if col in d.columns:
            d[f"{island}_Load_Lag_1h"] = d[col].shift(sph)
            for w_h in [3, 6]:
                w_steps = w_h * sph
                d[f"{island}_Load_Roll_Mean_{w_h}h"] = d[col].shift(1).rolling(w_steps).mean()

    # Heat index
    d["Heat_Index"] = d["Dry_Bulb_Temp"] * d["Rel_Humidity"] / 100
    
    # Weather Trends
    for w_h in [3, 6]:
        w_steps = w_h * sph
        d[f"Temp_Roll_Mean_{w_h}h"] = d["Dry_Bulb_Temp"].rolling(w_steps).mean()
        d[f"Humid_Roll_Mean_{w_h}h"] = d["Rel_Humidity"].rolling(w_steps).mean()
        
    # Temperature Gradient
    d["Temp_Gradient"] = d["Dry_Bulb_Temp"].diff()

    # Time features
    d["Hour_of_Day"]   = d.index.hour
    d["Day_of_Week"]   = d.index.dayofweek
    d["Is_High_Season"] = d.index.month.isin(dc["high_season_months"]).astype(int)

    # ── Holiday Features ──
    if "Is_Thai_Holiday" not in d.columns:
        holiday_dates = []
        for h in dc["holidays"].values():
            holiday_dates.extend(h)
        md = d.index.strftime("%m-%d")
        d["Is_Thai_Holiday"] = np.isin(md, holiday_dates).astype(int)

    # Tourist_Index fallback
    if "Tourist_Index" not in d.columns:
        d["Tourist_Index"] = d["Is_High_Season"] * 0.4 + 0.6

    # Drop leaky/redundant features (Keep Circuit_Cap_MW for eval reference but not for features)
    # Actually, we keep it for now as it's needed for Headroom calculation in future rows
    d.dropna(inplace=True)
    return d


def main():
    cfg = load_cfg()
    dc = cfg["data"]

    path = dc["output_path"]
    if not os.path.exists(path):
        print(f"Dataset {path} not found. Run generate first.")
        return

    print(f"Loading dataset: {path}")
    df = pd.read_parquet(path)
    df = engineer_features(df, cfg)

    # ── 2026 Strategy Splitting ──
    # Training:   Jan 2024 -> Feb 2026
    # Validation: March 1 -> March 20, 2026
    # Testing:    March 21 -> April 30, 2026
    
    train = df[df.index <= dc["train_end"]]
    val   = df[(df.index >= dc["val_start"]) & (df.index <= dc["val_end"])]
    test  = df[(df.index >= dc["test_start"]) & (df.index <= dc["test_end"])]

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"Train Range: {train.index.min()} to {train.index.max()}")
    print(f"Val Range:   {val.index.min()} to {val.index.max()}")
    print(f"Test Range:  {test.index.min()} to {test.index.max()}")

    # Scaling
    exclude = {
        "Island_Load_MW", "Headroom_MW",
        "Is_High_Season", "Hour_of_Day", "Day_of_Week", "Is_Thai_Holiday", "Is_Songkran",
        "Load_Lag_1h", "Load_Lag_24h", "Load_Lag_168h",
        "Load_Roll_Mean_3h", "Load_Roll_Std_3h",
        "Load_Roll_Mean_6h", "Load_Roll_Std_6h",
    }
    cluster_cols = {c for c in df.columns if c.startswith(("Phangan_", "Samui_"))}
    exclude |= cluster_cols

    common_cols = set(train.columns) & set(val.columns) & set(test.columns)
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in exclude and c in common_cols]

    scaler = StandardScaler()
    train.loc[:, num_cols] = scaler.fit_transform(train[num_cols])
    val.loc[:, num_cols]   = scaler.transform(val[num_cols])
    test.loc[:, num_cols]  = scaler.transform(test[num_cols])

    os.makedirs("data/processed", exist_ok=True)
    train.to_parquet("data/processed/train.parquet")
    val.to_parquet("data/processed/val.parquet")
    test.to_parquet("data/processed/test.parquet")
    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved train/val/test parquets and data/processed/scaler.pkl")


if __name__ == "__main__":
    main()

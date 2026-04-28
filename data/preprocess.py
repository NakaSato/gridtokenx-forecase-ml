"""
Feature engineering, train/val/test split, and normalization.
Input:  data/ko_tao_grid.parquet
Output: data/train.parquet, data/val.parquet, data/test.parquet, data/scaler.pkl
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
    d = df.copy()

    # Lag features
    for lag in [1, 24, 168]:
        d[f"Load_Lag_{lag}h"] = d["Island_Load_MW"].shift(lag)

    # Rolling stats
    for w in [3, 6]:
        d[f"Load_Roll_Mean_{w}h"] = d["Island_Load_MW"].shift(1).rolling(w).mean()
        d[f"Load_Roll_Std_{w}h"]  = d["Island_Load_MW"].shift(1).rolling(w).std()

    # Heat index
    d["Heat_Index"] = d["Dry_Bulb_Temp"] * d["Rel_Humidity"] / 100

    # Time features
    d["Hour_of_Day"]   = d.index.hour
    d["Day_of_Week"]   = d.index.dayofweek
    d["Is_Weekend"]    = (d["Day_of_Week"] >= 5).astype(int)
    d["Is_High_Season"] = d.index.month.isin(dc["high_season_months"]).astype(int)

    # Hours until next high-capacity window (Circuit_Cap > 10 MW)
    high_cap = (d["Circuit_Cap_MW"] > 10).astype(int)
    # forward-fill distance: count steps until next 1
    arr = high_cap.values
    dist = np.zeros(len(arr), dtype=float)
    countdown = np.inf
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == 1:
            countdown = 0
        else:
            countdown = countdown + 1 if countdown < np.inf else np.inf
        dist[i] = countdown
    dist = np.where(np.isinf(dist), 48, dist).clip(0, 48)  # cap at 48h
    d["Time_Until_High_Cap"] = dist

    d.dropna(inplace=True)
    return d

def split(df: pd.DataFrame):
    n = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]

def main():
    cfg = load_cfg()
    df = pd.read_parquet(cfg["data"]["output_path"])
    df = engineer_features(df, cfg)

    train, val, test = split(df)
    train, val, test = train.copy(), val.copy(), test.copy()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"Train: {train.index[0]} → {train.index[-1]}")
    print(f"Val:   {val.index[0]} → {val.index[-1]}")
    print(f"Test:  {test.index[0]} → {test.index[-1]}")

    # Normalize numeric columns (fit on train only)
    exclude = {"Island_Load_MW", "Circuit_Cap_MW", "Is_Weekend", "Is_High_Season",
               "Hour_of_Day", "Day_of_Week"}
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in exclude]

    scaler = StandardScaler()
    train.loc[:, num_cols] = scaler.fit_transform(train[num_cols])
    val.loc[:, num_cols]   = scaler.transform(val[num_cols])
    test.loc[:, num_cols]  = scaler.transform(test[num_cols])

    os.makedirs("data", exist_ok=True)
    train.to_parquet("data/train.parquet")
    val.to_parquet("data/val.parquet")
    test.to_parquet("data/test.parquet")
    with open("data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved train/val/test parquets and data/scaler.pkl")

if __name__ == "__main__":
    main()

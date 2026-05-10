"""
Feature engineering, train/val/test split, and multi-target normalization.
Input:  data/processed/island_grid.parquet
Output: data/processed/train.parquet, data/processed/val.parquet, data/processed/test.parquet, data/processed/scaler.pkl
"""
import os
import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import MSTL

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def decompose_kmb(df: pd.DataFrame, sph: int) -> pd.DataFrame:
    """Apply MSTL decomposition to KMB Remaining Capacity (Headroom)."""
    print("Decomposing KMB Headroom...")
    series = df["Samui_Circuit_MW"].copy()
    
    periods = [24 * sph]
    if len(series) > 168 * sph:
        periods.append(168 * sph)
        
    try:
        res = MSTL(series, periods=periods).fit()
        df["KMB_Trend"] = res.trend
        df["KMB_Seasonal"] = res.seasonal.sum(axis=1)
        df["KMB_Resid"] = res.resid
    except Exception as e:
        print(f"MSTL failed: {e}. Falling back to naive decomposition.")
        df["KMB_Trend"] = series.rolling(24*sph).mean().fillna(method='bfill')
        df["KMB_Seasonal"] = 0.0
        df["KMB_Resid"] = series - df["KMB_Trend"]
        
    return df

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dc = cfg["data"]
    freq = dc.get("frequency", "h")
    sph = 4 if freq == "15min" else 1
    d = df.copy()

    if "Circuit_Cap_MW" in d.columns:
        d["Headroom_MW"] = d["Circuit_Cap_MW"] - d["Island_Load_MW"]
    d["Max_Capacity_MW"] = d["Circuit_Cap_MW"].expanding().max()

    targets = ["Island_Load_MW", "Phangan_Load_MW", "Samui_Load_MW", "Samui_Circuit_MW"]
    for t in targets:
        for lag_h in [1, 24]:
            lag_steps = lag_h * sph
            d[f"{t}_Lag_{lag_h}h"] = d[t].shift(lag_steps)

    for w_h in [3, 6]:
        w_steps = w_h * sph
        d[f"Load_Roll_Mean_{w_h}h"] = d["Island_Load_MW"].shift(1).rolling(w_steps).mean()
        d[f"Load_Roll_Std_{w_h}h"]  = d["Island_Load_MW"].shift(1).rolling(w_steps).std()

    d["Heat_Index"] = d["Dry_Bulb_Temp"] * d["Rel_Humidity"] / 100
    
    for w_h in [3, 6]:
        w_steps = w_h * sph
        d[f"Temp_Roll_Mean_{w_h}h"] = d["Dry_Bulb_Temp"].rolling(w_steps).mean()
        d[f"Humid_Roll_Mean_{w_h}h"] = d["Rel_Humidity"].rolling(w_steps).mean()
        
    d["Temp_Gradient"] = d["Dry_Bulb_Temp"].diff()

    d["Hour_of_Day"]   = d.index.hour
    d["Day_of_Week"]   = d.index.dayofweek
    d["Is_High_Season"] = d.index.month.isin(dc["high_season_months"]).astype(int)

    holiday_dates = []
    for h in dc["holidays"].values():
        holiday_dates.extend(h)
    md = d.index.strftime("%m-%d")
    d["Is_Thai_Holiday"] = np.isin(md, holiday_dates).astype(int)

    d = decompose_kmb(d, sph)

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

    train = df[df.index <= dc["train_end"]]
    val   = df[(df.index >= dc["val_start"]) & (df.index <= dc["val_end"])]
    test  = df[(df.index >= dc["test_start"]) & (df.index <= dc["test_end"])]

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    targets = ["Island_Load_MW", "Phangan_Load_MW", "Samui_Load_MW", "Samui_Circuit_MW", 
               "KMB_Trend", "KMB_Seasonal", "KMB_Resid"]
    exclude = set(targets) | {
        "Is_High_Season", "Hour_of_Day", "Day_of_Week", "Is_Thai_Holiday", "Is_Songkran",
        "Headroom_MW", "Max_Capacity_MW"
    }
    exclude |= {c for c in df.columns if "Lag_" in c or "Roll_" in c}

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

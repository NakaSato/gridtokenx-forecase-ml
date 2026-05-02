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
    
    # Weather Trends (Smoothing unexplained variance)
    for w in [3, 6]:
        d[f"Temp_Roll_Mean_{w}h"] = d["Dry_Bulb_Temp"].rolling(w).mean()
        d[f"Humid_Roll_Mean_{w}h"] = d["Rel_Humidity"].rolling(w).mean()
        
    # Temperature Gradient (Is it getting hotter or colder?)
    d["Temp_Gradient"] = d["Dry_Bulb_Temp"].diff()

    # Time features
    d["Hour_of_Day"]   = d.index.hour
    d["Day_of_Week"]   = d.index.dayofweek
    d["Is_Weekend"]    = (d["Day_of_Week"] >= 5).astype(int)
    d["Is_High_Season"] = d.index.month.isin(dc["high_season_months"]).astype(int)

    # Tourist_Index fallback: if not present (production), derive from Is_High_Season
    if "Tourist_Index" not in d.columns:
        d["Tourist_Index"] = d["Is_High_Season"] * 0.4 + 0.6

    # Drop leaky and unavailable-at-inference features
    drop_cols = ["Net_Delta_MW", "Circuit_Cap_MW", "Is_Weekend"]
    d.drop(columns=[c for c in drop_cols if c in d.columns], inplace=True)

    d.dropna(inplace=True)
    return d

def split(df: pd.DataFrame):
    n = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


def impute_bess_soc(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Re-simulate BESS SoC if the column is constant (placeholder data)."""
    bc = cfg["bess"]
    if df["BESS_SoC_Pct"].std() < 0.1:
        print("  BESS_SoC_Pct is constant — re-simulating from load/circuit balance.")
        cap = bc["capacity_mwh"]
        charge_rate = bc.get("charge_rate_mw", 8.0)
        soc = np.zeros(len(df))
        soc[0] = 0.65
        load = df["Island_Load_MW"].values
        circuit = df["Circuit_Cap_MW"].values
        for i in range(1, len(df)):
            delta = circuit[i] - load[i]
            if delta > 0:
                soc[i] = min(bc["soc_max"], soc[i - 1] + min(delta, charge_rate) / cap)
            else:
                soc[i] = max(bc["soc_min"], soc[i - 1] + delta / cap)
        df = df.copy()
        df["BESS_SoC_Pct"] = (soc * 100).round(1)
    return df


def main():
    cfg = load_cfg()

    locked_path   = "data/ko_tao_grid_2023_locked.parquet"
    synthetic_path = cfg["data"]["output_path"]

    if os.path.exists(locked_path):
        print(f"Using REAL dataset: {locked_path}")
        df_real = pd.read_parquet(locked_path)
        df_real = impute_bess_soc(df_real, cfg)   # Fix: re-simulate constant SoC

        # FIX: val/test come from real data only (chronological)
        # FIX: synthetic used for training only, appended AFTER sorting by index
        print(f"Augmenting training with synthetic: {synthetic_path}")
        df_syn = pd.read_parquet(synthetic_path)
        df_syn = impute_bess_soc(df_syn, cfg)
        df_syn_fe = engineer_features(df_syn, cfg)

        df_real_fe = engineer_features(df_real, cfg)
        n_real = len(df_real_fe)
        val_start  = int(n_real * 0.70)
        test_start = int(n_real * 0.85)

        train_real = df_real_fe.iloc[:val_start]
        val        = df_real_fe.iloc[val_start:test_start]
        test       = df_real_fe.iloc[test_start:]

        # Use real data as primary, synthetic as light augmentation (20% weight via sampling)
        n_syn_sample = min(len(df_syn_fe), int(len(train_real) * 0.2))
        df_syn_sample = df_syn_fe.sample(n=n_syn_sample, random_state=42)
        train = pd.concat([train_real, df_syn_sample]).sort_index()
        df = df_syn_fe  # reference for column selection
    else:
        print(f"Using synthetic dataset: {synthetic_path}")
        df = pd.read_parquet(synthetic_path)
        df = impute_bess_soc(df, cfg)
        df = engineer_features(df, cfg)
        train, val, test = split(df)

    train, val, test = train.copy(), val.copy(), test.copy()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"Train: {train.index[0]} → {train.index[-1]}")
    print(f"Val:   {val.index[0]} → {val.index[-1]}")
    print(f"Test:  {test.index[0]} → {test.index[-1]}")

    # FIX: exclude lag/rolling load features from scaling — they must stay in MW
    # scale so TCN sequential inputs are commensurate with the MW target
    exclude = {
        "Island_Load_MW",
        "Is_High_Season", "Hour_of_Day", "Day_of_Week",
        "Load_Lag_1h", "Load_Lag_24h", "Load_Lag_168h",
        "Load_Roll_Mean_3h", "Load_Roll_Std_3h",
        "Load_Roll_Mean_6h", "Load_Roll_Std_6h",
    }
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

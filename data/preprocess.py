"""
Feature engineering, train/val/test split, and multi-target normalization.
Aligned to 15-min multi-target side-by-side dataset structure.
"""
import os
import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import MSTL

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def decompose_kmb(df: pd.DataFrame, sph: int) -> pd.DataFrame:
    """Apply MSTL decomposition to kmb_flow_mw (Bottleneck)."""
    print("Decomposing kmb_flow_mw...")
    series = df["kmb_flow_mw"].copy()
    
    periods = [24 * sph]
    if len(series) > 168 * sph:
        periods.append(168 * sph)
        
    try:
        res = MSTL(series, periods=periods).fit()
        df["kmb_trend"] = res.trend
        df["kmb_seasonal"] = res.seasonal.sum(axis=1)
        df["kmb_resid"] = res.resid
    except Exception as e:
        print(f"MSTL failed: {e}. Falling back to naive decomposition.")
        df["kmb_trend"] = series.rolling(24*sph).mean().fillna(method='bfill')
        df["kmb_seasonal"] = 0.0
        df["kmb_resid"] = series - df["kmb_trend"]
        
    return df

def impute_bess_soc(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Re-simulate BESS SoC if the column is constant or missing.
    Ko Tao has no BESS in reality, but models expect the feature.
    """
    bc = cfg["bess"]
    if "bess_soc_pct" not in df.columns:
        df["bess_soc_pct"] = 65.0
    
    # If SoC is constant, it's likely a synthetic placeholder; re-simulate for dynamics
    if df["bess_soc_pct"].std() < 0.01:
        print("  impute_bess_soc: Constant SoC detected. Re-simulating dynamics...")
        n = len(df)
        soc = np.zeros(n)
        soc[0] = 0.65
        cap = bc.get("capacity_mwh", 50.0)
        charge_rate = bc.get("charge_rate_mw", 10.0)
        
        # We need a circuit capacity to simulate surplus/deficit
        # Use tao_load_mw vs capacity_mw (renamed in engineer_features)
        load = df["tao_load_mw"].values if "tao_load_mw" in df.columns else df.get("Island_Load_MW", np.zeros(n))
        capacity = df["capacity_mw"].values if "capacity_mw" in df.columns else df.get("Circuit_Cap_MW", np.zeros(n))
        
        if cap > 0:
            for i in range(1, n):
                delta = capacity[i] - load[i]
                if delta > 0:
                    soc[i] = min(bc.get("soc_max", 0.9), soc[i-1] + min(delta, charge_rate) / cap)
                else:
                    soc[i] = max(bc.get("soc_min", 0.2), soc[i-1] + delta / cap)
            df["bess_soc_pct"] = soc * 100
        else:
            # Fallback to noise if no capacity
            df["bess_soc_pct"] = 65.0 + np.random.normal(0, 2, n)
            
    return df

def split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> tuple:
    """
    Chronological split into train, val, and test sets.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]
    
    return train, val, test

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dc = cfg["data"]
    freq = dc.get("frequency", "h")
    sph = 4 if freq == "15min" else 1
    d = df.copy()

    # ── 0. Renaming to standard schema ──
    rename_map = {
        "Island_Load_MW": "tao_load_mw",
        "Phangan_Load_MW": "phangan_load_mw",
        "Samui_Load_MW": "samui_load_mw",
        "Phangan_Circuit_MW": "cable_flow_mw",
        "Samui_Circuit_MW": "kmb_flow_mw",
        "Dry_Bulb_Temp": "t2m_celsius",
        "Rel_Humidity": "rh_pct",
        "Solar_Irradiance": "ghi_w_m2",
        "Wind_Speed": "wind_ms",
        "Phangan_Temp": "phangan_t2m",
        "Samui_Temp": "samui_t2m",
        "BESS_SoC_Pct": "bess_soc_pct",
        "Tourist_Index": "tourist_index",
        "Carbon_Intensity": "carbon_intensity",
        "Market_Price": "market_price",
        "Is_Thai_Holiday": "is_holiday",
        "Is_Songkran": "is_songkran",
        "Circuit_Cap_MW": "capacity_mw",
        # Backward compatibility for old names if they appear in raw
        "Load_Lag_1h": "tao_load_mw_lag_1h",
        "Load_Lag_24h": "tao_load_mw_lag_24h",
    }
    d = d.rename(columns=rename_map)

    # ── 1. Mandatory Schema Alignment ──
    # Ensure targets and core features exist
    if "tao_load_mw" not in d.columns:
        # Fallback if rename didn't catch it
        cols = d.columns
        if "load" in [c.lower() for c in cols]:
            target = next(c for c in cols if c.lower() == "load")
            d = d.rename(columns={target: "tao_load_mw"})

    # ── 2. System State ──
    if "capacity_mw" in d.columns:
        d["headroom_mw"] = d["capacity_mw"] - d["tao_load_mw"]
        d["max_capacity_mw"] = d["capacity_mw"].expanding().max()
    else:
        d["capacity_mw"] = dc.get("circuit_cap_max", 13.3)
        d["headroom_mw"] = d["capacity_mw"] - d["tao_load_mw"]
        d["max_capacity_mw"] = d["capacity_mw"]

    if "Phangan_SoC_Pct" in d.columns: d = d.rename(columns={"Phangan_SoC_Pct": "phangan_soc_pct"})
    if "Samui_SoC_Pct" in d.columns: d = d.rename(columns={"Samui_SoC_Pct": "samui_soc_pct"})
    
    # Fill missing optional islands/soc if not in cluster mode
    for col in ["phangan_load_mw", "samui_load_mw", "phangan_t2m", "samui_t2m", "phangan_soc_pct", "samui_soc_pct"]:
        if col not in d.columns: d[col] = 0.0

    # ── 3. Lags & Rolls ──
    targets = ["tao_load_mw", "cable_flow_mw", "kmb_flow_mw"]
    for t in targets:
        if t not in d.columns: continue
        for lag_h in [1, 24]:
            lag_steps = lag_h * sph
            d[f"{t}_lag_{lag_h}h"] = d[t].shift(lag_steps)

    # Rolling stats for main target
    for w_h in [3, 6]:
        w_steps = w_h * sph
        d[f"tao_load_roll_mean_{w_h}h"] = d["tao_load_mw"].shift(1).rolling(w_steps).mean()
        d[f"tao_load_roll_std_{w_h}h"]  = d["tao_load_mw"].shift(1).rolling(w_steps).std()

    # Weather Trends
    if "t2m_celsius" in d.columns and "rh_pct" in d.columns:
        d["heat_index"] = d["t2m_celsius"] * d["rh_pct"] / 100
        d["temp_gradient"] = d["t2m_celsius"].diff()

    # ── 4. Calendar ──
    d["hour_of_day"]   = d.index.hour
    d["day_of_week"]   = d.index.dayofweek
    d["is_high_season"] = d.index.month.isin(dc.get("high_season_months", [4,5,6,7,8,9,10])).astype(int)
    if "is_holiday" not in d.columns: d["is_holiday"] = 0
    if "is_songkran" not in d.columns: d["is_songkran"] = 0

    # ── 6. Final Cleanup ──
    # Explicitly drop any remaining raw or intermediate columns that are not features/targets
    keep_patterns = [
        "tao_load_mw", "phangan_load_mw", "samui_load_mw",
        "cable_flow_mw", "kmb_flow_mw",
        "t2m_celsius", "phangan_t2m", "samui_t2m",
        "rh_pct", "ghi_w_m2", "wind_ms",
        "bess_soc_pct", "phangan_soc_pct", "samui_soc_pct",
        "capacity_mw", "headroom_mw", "max_capacity_mw",
        "tourist_index", "carbon_intensity", "market_price",
        "is_holiday", "is_songkran", "is_high_season",
        "hour_of_day", "day_of_week",
        "lag_", "roll_", "kmb_", "heat_index", "temp_gradient"
    ]
    
    final_cols = [c for c in d.columns if any(p in c for p in keep_patterns)]
    d = d[final_cols]

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

    # ── Splitting ──
    train = df[df.index <= dc["train_end"]]
    val   = df[(df.index >= dc["val_start"]) & (df.index <= dc["val_end"])]
    test  = df[(df.index >= dc["test_start"]) & (df.index <= dc["test_end"])]

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # ── Scaling ──
    targets = ["tao_load_mw", "cable_flow_mw", "kmb_flow_mw"]
    exclude = set(targets) | {
        "is_high_season", "hour_of_day", "day_of_week", "is_holiday", "is_songkran",
        "headroom_mw", "max_capacity_mw"
    }
    
    # Exclude all lags and rolls from scaling to keep them in original units for TCN
    exclude |= {c for c in df.columns if "_lag_" in c or "_roll_" in c}
    exclude |= {"kmb_trend", "kmb_seasonal", "kmb_resid"}

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

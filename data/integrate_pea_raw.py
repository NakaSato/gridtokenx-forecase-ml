"""
PEA AWS Sandbox Integration Script
===================================
Aggregates per-substation SCADA files into a single multi-island cluster dataset.

Expects a directory containing:
  - Substation load files (Samui 1/2/3, Phangan, Tao)
  - Cable flow files (KMA/KMB 115kV, 33kV circuits)
  - Weather data files (12 months history)

Output: data/processed/pea_ground_truth.parquet
"""
import os
import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path

def load_scada_file(path: Path) -> pd.DataFrame:
    """Load a single SCADA file (CSV or Parquet) and normalize index."""
    ext = path.suffix.lower()
    df = pd.read_csv(path) if ext == ".csv" else pd.read_parquet(path)
    
    # Standardize timestamp
    ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    if ts_col:
        df.index = pd.to_datetime(df[ts_col])
        df = df.drop(columns=[ts_col])
    
    df = df.sort_index()
    df.index = df.index.tz_localize(None)
    # Resample to 15-min if needed (forward fill gaps up to 1h)
    df = df.resample("15min").ffill(limit=4)
    return df

def main():
    parser = argparse.ArgumentParser(description="PEA AWS Sandbox raw integration")
    parser.add_argument("--dir", required=True, help="Directory containing raw PEA export")
    parser.add_argument("--output", default="data/processed/pea_ground_truth.parquet", help="Output path")
    args = parser.parse_args()

    raw_dir = Path(args.dir)
    if not raw_dir.exists():
        print(f"❌ Directory not found: {raw_dir}")
        return

    print(f"Ingesting PEA AWS Sandbox data from: {raw_dir}")
    
    # ── 1. Aggregating Loads ──
    # Samui (Sum of 3 substations)
    samui_files = list(raw_dir.glob("*SAMUI*[123]*LOAD*"))
    if not samui_files:
        print("  ⚠️ No Samui load files found. Searching by island name...")
        samui_files = list(raw_dir.glob("*SAMUI*LOAD*"))
    
    samui_df = None
    for f in samui_files:
        print(f"  + Loading Samui substation: {f.name}")
        df = load_scada_file(f)
        val_col = next((c for c in df.columns if "MW" in c or "LOAD" in c.upper()), df.columns[0])
        sub_load = df[[val_col]].rename(columns={val_col: "Samui_Load_MW"})
        if samui_df is None:
            samui_df = sub_load
        else:
            samui_df = samui_df.add(sub_load, fill_value=0)

    # Phangan
    phangan_file = next(raw_dir.glob("*PHANGAN*LOAD*"), None)
    if phangan_file:
        print(f"  + Loading Phangan: {phangan_file.name}")
        df = load_scada_file(phangan_file)
        val_col = next((c for c in df.columns if "MW" in c or "LOAD" in c.upper()), df.columns[0])
        phangan_df = df[[val_col]].rename(columns={val_col: "Phangan_Load_MW"})
    else:
        print("  ⚠️ Phangan load file not found.")
        phangan_df = pd.DataFrame()

    # Tao
    tao_file = next(raw_dir.glob("*TAO*LOAD*"), None)
    if tao_file:
        print(f"  + Loading Tao: {tao_file.name}")
        df = load_scada_file(tao_file)
        val_col = next((c for c in df.columns if "MW" in c or "LOAD" in c.upper()), df.columns[0])
        tao_df = df[[val_col]].rename(columns={val_col: "Island_Load_MW"})
    else:
        print("  ⚠️ Tao load file not found.")
        tao_df = pd.DataFrame()

    # ── 2. Cable Flows (Bottleneck) ──
    # KMB 115kV is the primary bottleneck
    kmb_file = next(raw_dir.glob("*KMB*FLOW*"), None)
    if kmb_file:
        print(f"  + Loading KMB 115kV flow: {kmb_file.name}")
        df = load_scada_file(kmb_file)
        val_col = next((c for c in df.columns if "MW" in c or "FLOW" in c.upper()), df.columns[0])
        kmb_df = df[[val_col]].rename(columns={val_col: "Samui_Circuit_MW"})
    else:
        print("  ⚠️ KMB flow file not found. Falling back to default capacity logic.")
        kmb_df = pd.DataFrame()

    # ── 3. Weather ──
    weather_file = next(raw_dir.glob("*WEATHER*"), None)
    if weather_file:
        print(f"  + Loading weather: {weather_file.name}")
        weather_df = load_scada_file(weather_file)
        # Map common names
        weather_df = weather_df.rename(columns={
            "TEMP": "Dry_Bulb_Temp",
            "HUMID": "Rel_Humidity",
            "SOLAR": "Solar_Irradiance",
            "temp": "Dry_Bulb_Temp",
            "humidity": "Rel_Humidity",
            "solar": "Solar_Irradiance"
        }, errors="ignore")
    else:
        print("  ⚠️ Weather file not found.")
        weather_df = pd.DataFrame()

    # ── 4. Merge All ──
    print("\nMerging into master timeline...")
    master = pd.concat([samui_df, phangan_df, tao_df, kmb_df, weather_df], axis=1)
    
    # Clean up
    master = master.sort_index().dropna(subset=["Island_Load_MW"])
    
    # Fill gaps in exogenous
    master = master.interpolate(method="time").ffill().bfill()
    
    # Frequency check
    print(f"  Final shape: {master.shape}")
    print(f"  Range: {master.index.min()} to {master.index.max()}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    master.to_parquet(args.output)
    print(f"\n✅ Aggregation complete: {args.output}")

if __name__ == "__main__":
    main()

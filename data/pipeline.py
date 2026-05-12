"""
GridTokenX: Data Pipeline Orchestrator
======================================
Unified entry point for data ingestion, integration, and preprocessing.
Supports both synthetic generation and real-world SCADA integration.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import yaml
import time
from datetime import datetime

# Add root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def run_step(name, script_path, args=None):
    print(f"\n>>> [STEP] {name}")
    start_time = time.time()
    
    # Use 'uv run' if possible, else fallback to sys.executable
    cmd = ["uv", "run", "python", script_path]
    if args:
        cmd.extend(args)
    
    print(f"    Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"!!! Error in {name}. Aborting.")
            sys.exit(1)
    except FileNotFoundError:
        # Fallback to sys.executable if uv is not in PATH
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"!!! Error in {name}. Aborting.")
            sys.exit(1)
            
    elapsed = time.time() - start_time
    print(f">>> {name} completed in {elapsed:.2f}s")

def validate_data(cfg):
    print("\n>>> [STEP] Data Quality Validation")
    splits = ["train", "val", "test"]
    all_ok = True
    
    for s in splits:
        path = f"data/processed/{s}.parquet"
        if not os.path.exists(path):
            print(f"!!! CRITICAL: Missing processed split: {path}")
            all_ok = False
            continue
            
        df = pd.read_parquet(path)
        rows, cols = df.shape
        print(f"    {s.upper()}: {rows:,} rows, {cols} columns")
        
        # Check for nulls
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            print(f"    [WARNING] Found {null_counts} null values in {s}!")
            
        # Check date range
        if not df.index.is_monotonic_increasing:
            print(f"    [WARNING] Timestamps are not monotonic in {s}!")
            
        # Check for cluster features (SoC check)
        cluster_cols = ["phangan_soc_pct", "samui_soc_pct"]
        for c in cluster_cols:
            if c in df.columns:
                if (df[c] == 0).all():
                    print(f"    [WARNING] Cluster column '{c}' is all zeros in {s}!")
                else:
                    print(f"    [OK] Cluster column '{c}' has data.")
            else:
                print(f"    [ERROR] Missing cluster column '{c}' in {s}!")
                all_ok = False
            
    if all_ok:
        print(">>> Validation: PASS")
    else:
        print(">>> Validation: FAIL")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="GridTokenX Data Pipeline")
    parser.add_argument("--real", action="store_true", help="Use real data integration (ERA5/Tourism) instead of synthetic")
    parser.add_argument("--pea", type=str, help="Integrate PEA AWS Sandbox data from specified directory")
    parser.add_argument("--fetch", action="store_true", help="Fetch public datasets before processing")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing raw data")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    args = parser.parse_args()

    cfg = load_cfg()
    out_path = cfg["data"]["output_path"]
    
    print("="*60)
    print(f"      GRIDTOKENX DATA PIPELINE | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # --- STAGE 0: FETCHING ---
    if args.fetch:
        run_step("Fetching Public Datasets", "data/fetch_public_datasets.py")
        run_step("Fetching NREL PERFORM", "data/fetch_nrel_perform.py")

    # --- STAGE 1: INGESTION / INTEGRATION ---
    if args.pea:
        print(f"\nMode: PEA AWS SANDBOX INTEGRATION ({args.pea})")
        run_step("PEA Sandbox Integration", "data/integrate_pea_raw.py", ["--dir", args.pea])
        # Update out_path to the one used by integrate_pea_raw if needed
        # But for now we assume we want to preprocess whatever is in data/processed/
    elif args.real:
        print("\nMode: REAL-WORLD INTEGRATION (ERA5/Tourism)")
        # Check if raw files exist
        raw_weather = "data/raw/raw_weather_thailand.parquet"
        raw_tourism = "data/raw/raw_tourism_samui.csv"
        
        if not (os.path.exists(raw_weather) and os.path.exists(raw_tourism)):
            print(f"!!! Missing raw data files: {raw_weather} or {raw_tourism}")
            print("    Falling back to synthetic generation...")
            run_step("Synthetic Generation", "data/generate_dataset.py")
        else:
            run_step("Real Data Integration", "data/integrate_raw.py")
    else:
        print("\nMode: SYNTHETIC GENERATION")
        if os.path.exists(out_path) and not args.force:
            print(f"    {out_path} already exists. Skipping generation.")
        else:
            run_step("Synthetic Generation", "data/generate_dataset.py")

    # --- STAGE 2: PREPROCESSING ---
    if not args.skip_preprocess:
        run_step("Preprocessing & Scaling", "data/preprocess.py")
    else:
        print("\nSkipping Preprocessing as requested.")

    # --- STAGE 3: VALIDATION ---
    validate_data(cfg)

    print("\n" + "="*60)
    print("      DATA PIPELINE COMPLETE: SUCCESS")
    print("="*60)

if __name__ == "__main__":
    main()

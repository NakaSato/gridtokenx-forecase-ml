"""
GridTokenX — Data Folder Cleanup Utility
=========================================
Aggressively removes generated artifacts, temporary files, and cache
to ensure a clean state for the research pipeline.

Usage:
    python data/cleanup.py [--all] [--models] [--logs]
"""
import os
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
LOGS = ROOT / "mlruns"

def cleanup(remove_models=False, remove_logs=False, remove_all=False):
    print("🧹 Starting Data Folder Cleanup...")
    
    # 1. Clean data/processed
    if DATA_PROCESSED.exists():
        print(f"  + Cleaning {DATA_PROCESSED}...")
        for f in DATA_PROCESSED.glob("*"):
            if f.name == "README.md": continue
            if remove_all or f.suffix in [".parquet", ".pkl", ".bak", ".csv"]:
                if f.is_file(): f.unlink()
                elif f.is_dir(): shutil.rmtree(f)

    # 2. Clean results
    if RESULTS.exists():
        print(f"  + Cleaning {RESULTS}...")
        for f in RESULTS.glob("*"):
            if f.suffix in [".json", ".png", ".pdf", ".txt"]:
                f.unlink()

    # 3. Clean databases
    for db in ["mlflow.db", "api_state.db"]:
        db_path = ROOT / db
        if db_path.exists():
            print(f"  + Removing {db}...")
            db_path.unlink()

    # 4. Clean models (Optional)
    if remove_models and MODELS.exists():
        print(f"  + Cleaning {MODELS}...")
        for f in MODELS.glob("*"):
            if f.suffix in [".pkl", ".pt"]:
                f.unlink()

    # 5. Clean logs/mlruns (Optional)
    if remove_logs and LOGS.exists():
        print(f"  + Removing MLflow runs {LOGS}...")
        shutil.rmtree(LOGS)

    # 6. Clean __pycache__
    print("  + Removing __pycache__ folders...")
    for p in ROOT.rglob("__pycache__"):
        shutil.rmtree(p)

    print("\n✅ Cleanup Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridTokenX Cleanup Utility")
    parser.add_argument("--all", action="store_true", help="Remove everything including logs and models")
    parser.add_argument("--models", action="store_true", help="Remove model artifacts (.pkl, .pt)")
    parser.add_argument("--logs", action="store_true", help="Remove MLflow logs")
    args = parser.parse_args()
    
    cleanup(remove_models=args.models or args.all, 
            remove_logs=args.logs or args.all,
            remove_all=args.all)

import os, sys, sqlite3, torch, pickle
import yaml
import numpy as np

def check_env():
    print("🔍 Checking Environment...")
    omp_num = os.environ.get("OMP_NUM_THREADS")
    kmp_ok = os.environ.get("KMP_DUPLICATE_LIB_OK")
    print(f"  OMP_NUM_THREADS: {omp_num}")
    print(f"  KMP_DUPLICATE_LIB_OK: {kmp_ok}")
    
    if kmp_ok != "TRUE":
        print("  ⚠️  Warning: KMP_DUPLICATE_LIB_OK not set to TRUE. Potential macOS/Linux segfaults.")

def check_models():
    print("🔍 Checking Models...")
    paths = [
        "models/lgbm.pkl",
        "models/tcn.pt",
        "models/meta_learner.pkl"
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"  ✅ {p} found.")
        else:
            print(f"  ❌ {p} MISSING!")
            sys.exit(1)
            
    # Load test
    try:
        with open("models/lgbm.pkl", "rb") as f: pickle.load(f)
        torch.load("models/tcn.pt", map_location="cpu")
        print("  ✅ Model integrity verified.")
    except Exception as e:
        print(f"  ❌ Model Load Error: {e}")
        sys.exit(1)

def check_db():
    print("🔍 Checking SQLite State...")
    db_path = "api_state.db"
    try:
        conn = sqlite3.connect(db_path)
        curr = conn.cursor()
        curr.execute("SELECT count(*) FROM telemetry")
        count = curr.fetchone()[0]
        print(f"  ✅ DB healthy. Telemetry count: {count}")
        
        # Check headroom column
        curr.execute("PRAGMA table_info(telemetry)")
        cols = [c[1] for c in curr.fetchall()]
        if "headroom_mw" in cols:
            print("  ✅ Schema includes headroom_mw.")
        else:
            print("  ❌ Schema OUTDATED (missing headroom_mw).")
            sys.exit(1)
        conn.close()
    except Exception as e:
        print(f"  ⚠️  DB status: {e} (Expected if first run)")

if __name__ == "__main__":
    print("="*40)
    print("  GRIDTOKENX EDGE STARTUP SANITY CHECK")
    print("="*40)
    check_env()
    check_models()
    check_db()
    print("="*40)
    print("  🚀 READY FOR DISPATCH")
    print("="*40)

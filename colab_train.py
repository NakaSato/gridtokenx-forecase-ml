"""
GridTokenX — Colab GPU Training Script
Runs full pipeline: generate → preprocess → lgbm → tcn → hybrid → evaluate
Uses calibrated per-island data (Ko Tao stable, Phangan/Samui volatile).
MLflow tracking local only; artifacts auto-downloaded at end.
"""
import os, subprocess, sys

os.chdir("/content/gridtokenx")
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////content/mlflow_colab.db"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["COLAB_TRAIN"] = "1"   # disables mlflow.sklearn.log_model in lgbm_model.py

steps = [
    ("Generate",          "python data/generate_dataset.py"),
    ("Preprocess",        "python data/preprocess.py"),
    ("LightGBM",          "python models/lgbm_model.py"),
    ("TCN",               "python models/tcn_model.py"),
    ("Hybrid+Backtest",   "python models/hybrid_pipeline.py"),
    ("Evaluate",          "python evaluate.py"),
]

for name, cmd in steps:
    print(f"\n{'='*50}\n▶ {name}\n{'='*50}")
    r = subprocess.run(cmd, shell=True, text=True)
    if r.returncode != 0:
        print(f"❌ {name} failed — stopping.")
        sys.exit(1)
    print(f"✅ {name} done")

print("\n🏁 All steps complete. Artifacts are ready in /content/gridtokenx/models and /content/gridtokenx/results.")

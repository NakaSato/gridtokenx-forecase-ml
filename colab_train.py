"""
GridTokenX — Colab GPU Training Script
Runs full pipeline: preprocess → lgbm → tcn → hybrid → evaluate
MLflow tracking disabled (no remote server); artifacts downloaded at the end.
"""
import os, subprocess, sys

os.chdir("/content/gridtokenx")
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////content/mlflow_colab.db"  # local-only, not synced back

steps = [
    ("Preprocess",      "python data/preprocess.py"),
    ("LightGBM",        "python models/lgbm_model.py"),
    ("TCN",             "python models/tcn_model.py"),
    ("Hybrid+Backtest", "python models/hybrid_pipeline.py"),
    ("Evaluate",        "python evaluate.py"),
]

for name, cmd in steps:
    print(f"\n{'='*50}\n▶ {name}\n{'='*50}")
    r = subprocess.run(cmd, shell=True, text=True)
    if r.returncode != 0:
        print(f"❌ {name} failed — stopping.")
        sys.exit(1)
    print(f"✅ {name} done")

print("\n🏁 All steps complete. Downloading artifacts...")

from google.colab import files
for path in [
    "models/tcn.pt",
    "models/meta_learner.pkl",
    "models/lgbm.pkl",
    "results/evaluation_report.json",
]:
    if os.path.exists(path):
        files.download(path)
        print(f"  ↓ {path}")
    else:
        print(f"  ⚠ missing: {path}")

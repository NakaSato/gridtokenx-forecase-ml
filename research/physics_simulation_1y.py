import os, sys, json, pickle, subprocess, tempfile
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.pandapower_model import PhysicsEngine
from models.tcn_model import TCN, WindowDataset
from models.device import get_device

ROOT = os.path.dirname(os.path.dirname(__file__))

def lgbm_predict_subprocess(parquet_path: str) -> np.ndarray:
    out = tempfile.mktemp(suffix=".npy")
    script = f"""
import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, {repr(ROOT)})
import numpy as np, pandas as pd
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl","rb") as f: model = pickle.load(f)
df = pd.read_parquet({repr(parquet_path)})
np.save({repr(out)}, model.predict(df[FEATURES]))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0: raise RuntimeError(r.stderr)
    preds = np.load(out)
    os.unlink(out)
    return preds

def main():
    print("🚀 Starting Physics-Aware 1-Year Simulation")
    with open("config.yaml") as f: cfg = yaml.safe_load(f)
    
    device = get_device()
    print(f"   Device: {device}")

    # 1. Load Data
    test = pd.read_parquet("data/processed/test.parquet")
    print(f"   Test set: {len(test)} rows ({test.index[0]} to {test.index[-1]})")

    # 2. Batch Predictions
    print("   Running LightGBM (subprocess)...")
    lgbm_preds = lgbm_predict_subprocess("data/processed/test.parquet")

    print("   Running TCN (dataloader)...")
    ckpt = torch.load("models/tcn.pt", map_location=device, weights_only=False)
    tc_cfg = ckpt["config"]
    window, horizon = tc_cfg["window_size"], tc_cfg["forecast_horizon"]
    
    tcn_net = TCN(ckpt["in_features"], tc_cfg["filters"], tc_cfg["kernel_size"],
                  tc_cfg["layers"], horizon, n_targets=ckpt.get("n_targets", 3)).to(device)
    tcn_net.load_state_dict(ckpt["state_dict"])
    tcn_net.eval()
    
    ds = WindowDataset(test, window, horizon)
    dl = DataLoader(ds, batch_size=512, num_workers=0)
    tcn_raw = []
    with torch.no_grad():
        for xb, _ in dl:
            tcn_raw.append(tcn_net(xb.to(device)).cpu().numpy())
    
    tcn_raw = np.concatenate(tcn_raw)[:, 0, :] # (N, 3)
    tcn_preds = np.full((len(test), 3), np.nan)
    tcn_preds[window: window + len(tcn_raw)] = tcn_raw

    # 3. Hybrid Blending
    print("   Blending Meta-Learner...")
    with open("models/meta_learner.pkl", "rb") as f: metas = pickle.load(f)
    
    mask = ~np.isnan(tcn_preds[:, 0])
    hybrid_preds = np.zeros_like(lgbm_preds[mask])
    for i, meta in enumerate(metas):
        X = np.column_stack([lgbm_preds[mask, i], tcn_preds[mask, i]])
        hybrid_preds[:, i] = meta.predict(X)

    # 4. Physics Loop (The Core)
    print("   Running Physics Engine Loop (1 Year)...")
    engine = PhysicsEngine()
    
    # We'll align the physics results to the masked (valid forecast) period
    test_masked = test.iloc[mask]
    results = []
    
    for i in tqdm(range(len(test_masked))):
        row = test_masked.iloc[i]
        
        # Run AC Power Flow
        phys = engine.run_step(
            tao_load_mw=float(row["tao_load_mw"]),
            phangan_load_mw=float(row["phangan_load_mw"]),
            samui_load_mw=float(row["samui_load_mw"])
        )
        
        results.append({
            "timestamp": test_masked.index[i],
            "actual": float(row["tao_load_mw"]),
            "forecast": hybrid_preds[i, 0],
            "line_loss_mw": phys.get("line_loss_mw", 0.0),
            "v_tao_pu": phys.get("v_tao_pu", 1.0),
            "hvdc_loading_pct": phys.get("bottleneck_loading_pct", 0.0)
        })

    # 5. Finalize
    res_df = pd.DataFrame(results)
    res_df["error"] = res_df["actual"] - res_df["forecast"]
    res_df["ape"] = (res_df["error"].abs() / (res_df["actual"] + 1e-8)) * 100
    
    mae = res_df["error"].abs().mean()
    mape = res_df["ape"].mean()
    
    print(f"\n{'═'*60}")
    print(f"  1-YEAR PHYSICS VALIDATION COMPLETE")
    print(f"  MAE  : {mae:.4f} MW")
    print(f"  MAPE : {mape:.4f} %")
    print(f"  Avg Line Loss : {res_df['line_loss_mw'].mean():.4f} MW")
    print(f"  Min Ko Tao Voltage : {res_df['v_tao_pu'].min():.4f} p.u.")
    print(f"  Max Bottleneck Load : {res_df['hvdc_loading_pct'].max():.2f} %")
    
    out_path = "results/physics_1year_report.csv"
    os.makedirs("results", exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"  Saved detailed report to: {out_path}")
    print(f"{'═'*60}")

if __name__ == "__main__":
    main()

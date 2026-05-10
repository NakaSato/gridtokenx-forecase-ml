"""
Hybrid pipeline: Multi-Target LightGBM + TCN → Parallel Ridge meta-learners.
Aligned to 3-island targets: tao_load_mw, cable_flow_mw, kmb_flow_mw.
"""
import os, sys, pickle, subprocess, tempfile
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from models.mlflow_utils import setup_mlflow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TARGETS = ["tao_load_mw", "cable_flow_mw", "kmb_flow_mw"]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def lgbm_predict_subprocess(parquet_path: str, out_path: str):
    """Run multi-target lgbm.predict in a clean subprocess."""
    script = f"""
import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, {repr(ROOT)})
import numpy as np, pandas as pd
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl", "rb") as f:
    model = pickle.load(f)
df = pd.read_parquet({repr(parquet_path)})

# Fill missing features with 0.0
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0

preds = model.predict(df[FEATURES])
np.save({repr(out_path)}, preds)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=ROOT
    )
    if result.returncode != 0:
        raise RuntimeError(f"lgbm subprocess failed:\n{result.stderr}")


def get_tcn_preds(ckpt, df, device):
    import torch
    from torch.utils.data import DataLoader
    from models.tcn_model import TCN, WindowDataset
    tc = ckpt["config"]
    window, horizon = tc["window_size"], tc["forecast_horizon"]
    n_targets = ckpt.get("n_targets", 3)
    net = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
              tc["layers"], horizon, n_targets=n_targets, dropout=tc.get("dropout", 0.2)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    ds = WindowDataset(df, window, horizon)
    dl = DataLoader(ds, batch_size=256, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(net(xb.to(device)).cpu().numpy())
    
    # preds: (N, horizon, n_targets)
    preds = np.concatenate(preds)[:, 0, :] # Focus on first step for backtest
    aligned = np.full((len(df), n_targets), np.nan)
    aligned[window: window + len(preds)] = preds
    return aligned


def train_meta_learner(val_df: pd.DataFrame, test_df: pd.DataFrame, ckpt: dict, device: str):
    """Core meta-learner training logic."""
    setup_mlflow()
    mlflow.set_experiment("GridTokenX_Hybrid")

    # LightGBM predictions
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f: val_npy = f.name
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f: test_npy = f.name

    print("Running LightGBM predictions (subprocess)...")
    # We write temporary parquets if dfs are not already on disk, 
    # but here we assume the paths provided in main are used or we write them.
    val_df.to_parquet(val_npy + ".parquet")
    test_df.to_parquet(test_npy + ".parquet")
    
    lgbm_predict_subprocess(val_npy + ".parquet",  val_npy)
    lgbm_predict_subprocess(test_npy + ".parquet", test_npy)
    
    lgbm_val  = np.load(val_npy)
    lgbm_test = np.load(test_npy)
    
    os.unlink(val_npy); os.unlink(test_npy)
    os.unlink(val_npy + ".parquet"); os.unlink(test_npy + ".parquet")

    # TCN predictions
    print("Running TCN predictions...")
    tcn_val  = get_tcn_preds(ckpt, val_df,  device)
    tcn_test = get_tcn_preds(ckpt, test_df, device)

    # Meta-learner: Parallel Ridge for each target
    metas = []
    
    with mlflow.start_run(run_name="hybrid_multi_target_aligned"):
        for i, target in enumerate(TARGETS):
            print(f"Training meta-learner for {target}...")
            
            # Align
            mask_v = ~np.isnan(tcn_val[:, i])
            mask_t = ~np.isnan(tcn_test[:, i])
            
            X_val = np.column_stack([lgbm_val[mask_v, i], tcn_val[mask_v, i]])
            X_test = np.column_stack([lgbm_test[mask_t, i], tcn_test[mask_t, i]])
            y_val = val_df[target].values[mask_v]
            y_test = test_df[target].values[mask_t]

            meta = Ridge(alpha=1.0)
            meta.fit(X_val, y_val)
            metas.append(meta)
            
            p_test = meta.predict(X_test)
            t_mape = mape(y_test, p_test)
            mlflow.log_metric(f"test_mape_{target}", t_mape)
            print(f"  {target:<25} | Test MAPE: {t_mape:.4f}%")

    return metas


def main():
    import torch
    from models.device import get_device

    device = get_device()
    print(f"Using device: {device}")

    ckpt_path = os.path.join(ROOT, "models/tcn.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    val  = pd.read_parquet(os.path.join(ROOT, "data/processed/val.parquet"))
    test = pd.read_parquet(os.path.join(ROOT, "data/processed/test.parquet"))

    metas = train_meta_learner(val, test, ckpt, device)

    save_path = os.path.join(ROOT, "models/meta_learner.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(metas, f)
    print(f"Saved → {save_path}")

if __name__ == "__main__":
    main()

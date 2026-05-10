"""
Hybrid pipeline: Multi-Target LightGBM + TCN → Parallel Ridge meta-learners.
"""
import os, sys, pickle, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from models.mlflow_utils import setup_mlflow

setup_mlflow()
mlflow.set_experiment("GridTokenX_Hybrid")

TARGETS = ["Samui_Load_MW", "Phangan_Load_MW", "Island_Load_MW", "Samui_Circuit_MW"]

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
    dropout = ckpt.get("dropout", 0.0)
    window, horizon = tc["window_size"], tc["forecast_horizon"]
    n_targets = ckpt.get("n_targets", 4)
    net = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
              tc["layers"], horizon, n_targets=n_targets, dropout=dropout).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    ds = WindowDataset(df, window, horizon)
    dl = DataLoader(ds, batch_size=256, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(net(xb.to(device)).cpu().numpy())
    
    preds = np.concatenate(preds)[:, 0, :] # → (N, n_targets)
    aligned = np.full((len(df), n_targets), np.nan)
    aligned[window: window + len(preds)] = preds
    return aligned


def main():
    import torch
    from models.device import get_device

    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load("models/tcn.pt", map_location=device)

    val  = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f: val_npy = f.name
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f: test_npy = f.name

    print("Running LightGBM predictions (subprocess)...")
    lgbm_predict_subprocess("data/processed/val.parquet",  val_npy)
    lgbm_predict_subprocess("data/processed/test.parquet", test_npy)
    lgbm_val  = np.load(val_npy)
    lgbm_test = np.load(test_npy)
    os.unlink(val_npy); os.unlink(test_npy)

    print("Running TCN predictions...")
    tcn_val  = get_tcn_preds(ckpt, val,  device)
    tcn_test = get_tcn_preds(ckpt, test, device)

    metas = []
    final_preds_test = np.zeros_like(lgbm_test)

    with mlflow.start_run(run_name="hybrid_multi_target"):
        for i, target in enumerate(TARGETS):
            print(f"Training meta-learner for {target}...")
            
            mask_v = ~np.isnan(tcn_val[:, i])
            mask_t = ~np.isnan(tcn_test[:, i])
            
            X_val = np.column_stack([lgbm_val[mask_v, i], tcn_val[mask_v, i]])
            X_test = np.column_stack([lgbm_test[mask_t, i], tcn_test[mask_t, i]])
            y_val = val[target].values[mask_v]
            y_test = test[target].values[mask_t]

            meta = Ridge(alpha=1.0)
            meta.fit(X_val, y_val)
            metas.append(meta)
            
            p_test = meta.predict(X_test)
            final_preds_test[mask_t, i] = p_test
            
            t_mape = mape(y_test, p_test)
            mlflow.log_metric(f"test_mape_{target}", t_mape)
            print(f"  {target:<20} | Test MAPE: {t_mape:.4f}%")

        tao_idx = TARGETS.index("Island_Load_MW")
        tao_mape = mape(test["Island_Load_MW"].values[~np.isnan(tcn_test[:, tao_idx])], 
                         final_preds_test[~np.isnan(tcn_test[:, tao_idx]), tao_idx])
        mlflow.log_metric("test_mape_tao", tao_mape)

    os.makedirs("models", exist_ok=True)
    with open("models/meta_learner.pkl", "wb") as f:
        pickle.dump(metas, f)
    print("Saved → models/meta_learner.pkl")

if __name__ == "__main__":
    main()

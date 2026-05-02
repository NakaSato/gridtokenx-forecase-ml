"""
12-Month Long-Term Backtest Script (May 2025 - April 2026)
Validates model stability across all seasons.
"""
import os, sys, pickle, tempfile, subprocess
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.tcn_model import TCN, WindowDataset
from models.device import get_device

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

def lgbm_predict_subprocess(parquet_path: str, out_path: str):
    """Run lgbm.predict in a clean subprocess to avoid OpenMP conflicts."""
    script = f"""
import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, {repr(ROOT)})
import numpy as np, pandas as pd
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl", "rb") as f:
    model = pickle.load(f)
df = pd.read_parquet({repr(parquet_path)})
# Handle missing features
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0
preds = model.predict(df[FEATURES])
np.save({repr(out_path)}, preds)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"lgbm subprocess failed:\n{result.stderr}")

def get_tcn_preds(ckpt, df, device):
    tc = ckpt["config"]
    window, horizon = tc["window_size"], tc["forecast_horizon"]
    net = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
              tc["layers"], horizon, dropout=tc.get("dropout", 0.2)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    ds = WindowDataset(df, window, horizon)
    dl = DataLoader(ds, batch_size=256, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(net(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds)[:, 0]
    aligned = np.full(len(df), np.nan)
    aligned[window: window + len(preds)] = preds
    return aligned

def run_backtest():
    device = get_device()
    print(f"--- 12-Month Backtest (Target: May 2025 - Apr 2026) ---")
    print(f"Device: {device}")

    # Load All Data
    df_train = pd.read_parquet("data/processed/train.parquet")
    df_val = pd.read_parquet("data/processed/val.parquet")
    df_test = pd.read_parquet("data/processed/test.parquet")
    full = pd.concat([df_train, df_val, df_test])
    
    # Slice 12 months
    mask = (full.index >= "2025-05-01") & (full.index <= "2026-04-30")
    bt_df = full[mask].copy()
    print(f"Backtest range: {bt_df.index.min()} to {bt_df.index.max()} ({len(bt_df)} steps)")

    # 1. TCN Predictions
    ckpt = torch.load("models/tcn.pt", map_location=device)
    tcn_preds = get_tcn_preds(ckpt, bt_df, device)

    # 2. LightGBM Predictions (via temp parquet to keep subprocess clean)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_parquet = f.name
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_npy = f.name
    
    bt_df.to_parquet(tmp_parquet)
    lgbm_predict_subprocess(tmp_parquet, tmp_npy)
    lgbm_preds = np.load(tmp_npy)
    os.unlink(tmp_parquet); os.unlink(tmp_npy)

    # 3. Meta-Learner Blend
    with open("models/meta_learner.pkl", "rb") as f:
        meta = pickle.load(f)
    
    # Align and Blend
    valid_mask = ~np.isnan(tcn_preds)
    y_true = bt_df["Island_Load_MW"].values[valid_mask]
    X_meta = np.column_stack([lgbm_preds[valid_mask], tcn_preds[valid_mask]])
    y_pred = meta.predict(X_meta)

    # Calculate Metrics
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    
    print(f"\nOverall 12-Month Results:")
    print(f"  MAPE: {mape:6.3f}%")
    print(f"  MAE:  {mae:6.3f} MW")

    # Seasonal Analysis
    bt_df_valid = bt_df.iloc[valid_mask].copy()
    bt_df_valid["pred"] = y_pred
    bt_df_valid["abs_err"] = np.abs(y_true - y_pred)
    bt_df_valid["mape"] = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100

    monthly = bt_df_valid.resample("ME").agg({"mape": "mean", "Island_Load_MW": "mean"})
    print("\nMonthly Performance:")
    print(monthly)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly.index, monthly["mape"], marker='o', lw=2, color='blue', label='Monthly MAPE')
    plt.axhline(10.0, color='red', ls='--', label='PEA Threshold (10%)')
    plt.title("12-Month Backtest: Seasonal Forecast Stability", fontsize=14)
    plt.ylabel("MAPE (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("results/backtest_12m.png")
    print(f"\n✅ 12-month backtest chart saved to: results/backtest_12m.png")

    # JSON report
    report = {
        "overall_mape": float(mape),
        "overall_mae": float(mae),
        "monthly_mape": {str(k): float(v) for k, v in monthly["mape"].to_dict().items()}
    }
    import json
    with open("results/backtest_12m_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run_backtest()

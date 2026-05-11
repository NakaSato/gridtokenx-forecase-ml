import os
import sys
import json
import pickle
import subprocess
import tempfile
import numpy as np
import pandas as pd
import torch
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research.pandapower_model import PhysicsEngine
from models.tcn_model import TCN, WindowDataset
from models.device import get_device
from optimizer.early_warning import check_warnings, format_warnings
from models.schema import SEQ_FEATURES, TARGETS, TAB_FEATURES

# Fix for OpenMP conflict if running on Mac/CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)

def lgbm_predict_subprocess(df_path: str) -> np.ndarray:
    out = tempfile.mktemp(suffix=".npy")
    script = f"""
import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, {repr(ROOT)})
import numpy as np, pandas as pd
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl","rb") as f: model = pickle.load(f)
df = pd.read_parquet({repr(df_path)})
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0
np.save({repr(out)}, model.predict(df[FEATURES]))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0: raise RuntimeError(r.stderr)
    preds = np.load(out)
    os.unlink(out)
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=1)
    parser.add_argument("--input", type=str, default="data/processed/test.parquet")
    parser.add_argument("--output", type=str, default="results/backtest_pea_report.csv")
    parser.add_argument("--fast", action="store_true", help="Skip physics check in lookahead for speed")
    args = parser.parse_args()

    print(f"🚀 Starting E2E Backtest ({args.months} month(s)) {'[FAST MODE]' if args.fast else ''}")
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"   Device: {device}")

    # 1. Load Data
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    df = pd.read_parquet(args.input)
    print(f"   Dataset: {len(df)} rows ({df.index[0]} to {df.index[-1]})")

    # 2. Load Models (except LGBM which we'll run in subprocess)
    print("   Loading models...")
    try:
        ckpt = torch.load("models/tcn.pt", map_location=device, weights_only=False)
        tc_cfg = ckpt["config"]
        tcn = TCN(
            ckpt["in_features"], tc_cfg["filters"], tc_cfg["kernel_size"],
            tc_cfg["layers"], tc_cfg["forecast_horizon"], n_targets=ckpt.get("n_targets", 3)
        ).to(device)
        tcn.load_state_dict(ckpt["state_dict"])
        tcn.eval()

        with open("models/meta_learner.pkl", "rb") as f:
            metas = pickle.load(f)
            
        import joblib
        tcn_scaler = joblib.load("data/processed/tcn_scaler.pkl") if os.path.exists("data/processed/tcn_scaler.pkl") else None
        target_scaler = joblib.load("data/processed/target_scaler.pkl") if os.path.exists("data/processed/target_scaler.pkl") else None
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 3. Slice Data for requested duration
    if args.months < 12:
        steps_per_month = 4 * 24 * 31
        df = df.iloc[:args.months * steps_per_month + tc_cfg["window_size"] + tc_cfg["forecast_horizon"]]
        print(f"   Sliced Dataset: {len(df)} rows for {args.months} month(s)")

    # 4. Batch Predictions
    print("   Running batch predictions...")
    
    # Save a temporary parquet of the sliced df for the subprocess
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_parquet = f.name
    df.to_parquet(tmp_parquet)
    
    lgbm_preds = lgbm_predict_subprocess(tmp_parquet)
    os.unlink(tmp_parquet)
    
    # TCN
    window = tc_cfg["window_size"]
    horizon = tc_cfg["forecast_horizon"]
    ds = WindowDataset(df, window, horizon, target_scaler=None, feature_scaler=tcn_scaler)
    dl = torch.utils.data.DataLoader(ds, batch_size=512)
    
    tcn_raw = []
    with torch.no_grad():
        for xb, _ in tqdm(dl, desc="TCN Batch"):
            tcn_raw.append(tcn.forward(xb.to(device)).cpu().numpy())
    
    tcn_all = np.concatenate(tcn_raw) # (N-win-hor+1, horizon, 3)
    
    # Inverse transform TCN if needed
    if target_scaler is not None:
        tcn_all_flat = tcn_all.reshape(-1, 3)
        tcn_all_flat = target_scaler.inverse_transform(tcn_all_flat)
        tcn_all = tcn_all_flat.reshape(tcn_all.shape)

    # 4. Simulation Loop
    print("   Running Physics + Warning Loop...")
    engine = PhysicsEngine()
    
    results = []
    warning_logs = []
    
    # We start from when TCN has enough window
    start_idx = window
    end_idx = len(df) - horizon
    
    for i in tqdm(range(start_idx, end_idx), desc="Simulation"):
        ts = df.index[i]
        actual_row = df.iloc[i]
        
        # 4.1 Actual Physics (Ground Truth)
        phys = engine.run_step(
            tao_load_mw=float(actual_row["tao_load_mw"]),
            phangan_load_mw=float(actual_row["phangan_load_mw"]),
            samui_load_mw=float(actual_row["samui_load_mw"])
        )
        
        # 4.2 Forecast Generation (24h lookahead)
        # tcn_idx corresponds to current step i in the original df
        # ds loop was range(len(vals) - window - horizon + 1)
        # i-window gives the correct index in tcn_all
        tcn_idx = i - window
        tcn_fc = tcn_all[tcn_idx] # (horizon, 3)
        
        # Hybrid Blending for the very first step (current point of interest)
        lgbm_p = lgbm_preds[i] # (3,)
        tcn_p_first = tcn_fc[0] # (3,)
        
        hybrid_p = np.zeros(3)
        for j in range(3):
            X_meta = np.array([[lgbm_p[j], tcn_p_first[j]]])
            hybrid_p[j] = metas[j].predict(X_meta)[0]
            
        # Construct full 24h forecast for Early Warning
        # We use Hybrid for step 0, and TCN for steps 1..horizon-1
        # Targets: 0:tao_load, 1:cable_flow (Phangan link), 2:kmb_flow (Mainland link)
        tao_fc = np.concatenate([[hybrid_p[0]], tcn_fc[1:, 0]])
        
        # circuit_forecast: capacity_mw is the limit. 
        # In early_warning, it uses this to check for bottleneck.
        circuit_fc = df["capacity_mw"].values[i : i + horizon]
        
        # Cluster forecasts for physics check
        # We don't have explicit models for phangan_load and samui_load,
        # but we can use their actuals as "perfect forecast" to test the system's 
        # ability to flag violations given correct secondary inputs.
        # Or we use persistence? Let's use actuals for this "Upper Bound" test.
        phangan_fc = df["phangan_load_mw"].values[i : i + horizon]
        samui_fc = df["samui_load_mw"].values[i : i + horizon]
        
        # 4.3 Early Warning Check
        soc = float(actual_row.get("bess_soc_pct", 0.5)) # Ko Tao dummy SOC
        
        # In fast mode, we skip phangan/samui forecasts to skip lookahead physics checks
        p_fc = phangan_fc if not args.fast else None
        s_fc = samui_fc if not args.fast else None

        warnings = check_warnings(
            load_forecast=tao_fc,
            circuit_forecast=circuit_fc,
            current_soc=soc,
            cfg=cfg,
            lookahead_hours=24,
            phangan_forecast=p_fc,
            samui_forecast=s_fc
        )
        
        # 4.4 Record results
        has_critical = any(w.level == "CRITICAL" for w in warnings)
        has_warning = any(w.level == "WARNING" for w in warnings)
        
        # Actual violations
        actual_violation = False
        if phys["status"] == "SUCCESS":
            if phys["bottleneck_loading_pct"] > 100.0 or phys["v_tao_pu"] < 0.95 or phys["v_tao_pu"] > 1.05:
                actual_violation = True
        
        results.append({
            "timestamp": ts,
            "actual_tao": float(actual_row["tao_load_mw"]),
            "forecast_tao": float(hybrid_p[0]),
            "actual_kmb": float(actual_row["kmb_flow_mw"]),
            "forecast_kmb": float(hybrid_p[2]),
            "hvdc_loading_pct": phys.get("bottleneck_loading_pct", 0.0),
            "v_tao_pu": phys.get("v_tao_pu", 1.0),
            "has_critical_warn": has_critical,
            "has_warning_warn": has_warning,
            "actual_violation": actual_violation
        })
        
        if warnings:
            warning_logs.append({
                "timestamp": ts.isoformat(),
                "warnings": [w.__dict__ for w in warnings]
            })

    # 5. Metrics & Reporting
    res_df = pd.DataFrame(results)
    res_df["error_tao"] = res_df["actual_tao"] - res_df["forecast_tao"]
    res_df["ape_tao"] = (res_df["error_tao"].abs() / (res_df["actual_tao"] + 1e-8)) * 100
    
    mape_tao = res_df["ape_tao"].mean()
    mae_tao = res_df["error_tao"].abs().mean()
    
    # Warning Performance (Binary Classification)
    # TP: Warning issued AND actual violation occurs (or will occur in window)
    # For simplicity: TP if (has_critical_warn or has_warning_warn) AND actual_violation is True at this step
    # Better: TP if Warning at t predicts violation in t..t+lookahead.
    # But let's start with simultaneous for a baseline.
    
    tp = len(res_df[(res_df["has_critical_warn"]) & (res_df["actual_violation"])])
    fp = len(res_df[(res_df["has_critical_warn"]) & (~res_df["actual_violation"])])
    fn = len(res_df[(~res_df["has_critical_warn"]) & (res_df["actual_violation"])])
    tn = len(res_df[(~res_df["has_critical_warn"]) & (~res_df["actual_violation"])])
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"\n{'═'*60}")
    print(f"  BACKTEST SUMMARY")
    print(f"  {'─'*60}")
    print(f"  Forecasting (Ko Tao):")
    print(f"    MAE  : {mae_tao:.4f} MW")
    print(f"    MAPE : {mape_tao:.4f} %")
    print(f"  {'─'*60}")
    print(f"  Early Warning (CRITICAL level):")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"  {'─'*60}")
    print(f"  Grid Reliability:")
    print(f"    Actual Violations : {res_df['actual_violation'].sum()}")
    print(f"    Max Loading       : {res_df['hvdc_loading_pct'].max():.2f} %")
    print(f"    Min Voltage       : {res_df['v_tao_pu'].min():.4f} p.u.")
    print(f"{'═'*60}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    res_df.to_csv(args.output, index=False)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "mape_tao": mape_tao,
            "mae_tao": mae_tao,
            "warning_precision": precision,
            "warning_recall": recall,
            "warning_f1": f1,
            "max_loading": res_df["hvdc_loading_pct"].max(),
            "min_voltage": res_df["v_tao_pu"].min()
        }
    }
    with open(args.output.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(args.output.replace(".csv", "_warnings.json"), "w") as f:
        json.dump(warning_logs, f, indent=2)

    print(f"✅ Reports saved to results/")

if __name__ == "__main__":
    main()

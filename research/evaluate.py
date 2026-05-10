"""
Evaluation: multi-target hybrid forecast (3-island) + coordinated dispatch.
Aligned to Project Schema.
"""
import os, sys, json, pickle, subprocess, tempfile
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import yaml
import mlflow
from models.mlflow_utils import setup_mlflow
from sklearn.metrics import mean_absolute_error, r2_score

setup_mlflow()
mlflow.set_experiment("GridTokenX_Eval")

from models.tcn_model import TCN, WindowDataset
from models.device import get_device
from optimizer.pea_dispatch_opt import cluster_optimize, ClusterStepResult

ROOT = os.path.dirname(__file__)
TARGETS = ["tao_load_mw", "cable_flow_mw", "kmb_flow_mw"]

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)


def lgbm_predict_subprocess(parquet_path: str) -> np.ndarray:
    out = tempfile.mktemp(suffix=".npy")
    script = f"""
import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname({repr(ROOT)}))
import numpy as np, pandas as pd
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl","rb") as f: model = pickle.load(f)
df = pd.read_parquet({repr(parquet_path)})

for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0

np.save({repr(out)}, model.predict(df[FEATURES]))
"""
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, cwd=os.path.dirname(ROOT))
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    preds = np.load(out)
    os.unlink(out)
    return preds


def get_tcn_preds(ckpt, df, device):
    from torch.utils.data import DataLoader
    tc = ckpt["config"]
    window, horizon = tc["window_size"], tc["forecast_horizon"]
    n_targets = ckpt.get("n_targets", 3)
    net = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
              tc["layers"], horizon, n_targets=n_targets, 
              dropout=ckpt.get("dropout", 0.2)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    ds = WindowDataset(df, window, horizon)
    dl = DataLoader(ds, batch_size=256, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(net(xb.to(device)).cpu().numpy())
    
    # preds: (N, horizon, n_targets)
    preds = np.concatenate(preds)[:, 0, :] # Focus on first step
    aligned = np.full((len(df), n_targets), np.nan)
    aligned[window: window + len(preds)] = preds
    return aligned


def main():
    with mlflow.start_run(run_name="eval_multi_target_aligned"):
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

        device = get_device()
        print(f"Device: {device}")

        # ── Load models ──────────────────────────────────────────────────────────
        with open("models/meta_learner.pkl", "rb") as f:
            metas = pickle.load(f)
        ckpt = torch.load("models/tcn.pt", map_location=device, weights_only=False)

        test = pd.read_parquet("data/processed/test.parquet")
        y_true_all = test[TARGETS].values

        # ── Forecast ─────────────────────────────────────────────────────────────
        print("LightGBM predictions...")
        lgbm_preds = lgbm_predict_subprocess("data/processed/test.parquet")
        print("TCN predictions...")
        tcn_preds  = get_tcn_preds(ckpt, test, device)

        mask = ~np.isnan(tcn_preds[:, 0])
        hybrid_preds_all = np.zeros_like(lgbm_preds[mask])
        
        for i, meta in enumerate(metas):
            X = np.column_stack([lgbm_preds[mask, i], tcn_preds[mask, i]])
            hybrid_preds_all[:, i] = meta.predict(X)
        
        # Metrics for each target
        report_metrics = {}
        for i, target in enumerate(TARGETS):
            yt = y_true_all[mask, i]
            yp = hybrid_preds_all[:, i]
            m = {
                f"mape_{target}": round(mape(yt, yp), 4),
                f"mae_{target}":  round(float(mean_absolute_error(yt, yp)), 4),
                f"r2_{target}":   round(float(r2_score(yt, yp)), 4),
            }
            report_metrics.update(m)
            print(f"{target:<20} → MAPE: {m[f'mape_{target}']}%  R²: {m[f'r2_{target}']}")
            
        mlflow.log_metrics(report_metrics)

        # ── Coordinated Dispatch on test set ──
        print("Running coordinated dispatch evaluation...")
        freq = cfg["data"].get("frequency", "h")
        sph = 4 if freq == "15min" else 1
        step_24h = 24 * sph
        
        n_days = len(hybrid_preds_all) // step_24h
        
        # Target mapping:
        # i=0: tao_load_mw
        # i=1: cable_flow_mw (Samui-Phangan capacity)
        # i=2: kmb_flow_mw (Mainland-Samui headroom)
        preds_tao = hybrid_preds_all[:, 0]
        preds_cable = hybrid_preds_all[:, 1]
        preds_kmb = hybrid_preds_all[:, 2]
        
        total_fuel = total_carbon = 0.0
        soc = 0.65
        
        test_masked = test.iloc[mask]
        
        for d in range(n_days):
            start, end = d*step_24h, (d+1)*step_24h
            
            day_loads = {
                "tao": preds_tao[start:end],
                "pha": test_masked["phangan_load_mw"].values[start:end],
                "sam": test_masked["samui_load_mw"].values[start:end]
            }
            day_kmb = preds_kmb[start:end]
            day_cable = preds_cable[start:end]
            # Use predicted capacity for Phangan-Tao link if available
            day_capacity = test_masked["capacity_mw"].values[start:end] 
            
            res = cluster_optimize(day_loads, day_kmb, initial_soc=soc, cfg=cfg,
                                   cable_cap=day_cable, distal_cap=day_capacity)
            if res["status"] == "SUCCESS":
                total_fuel   += res["total_fuel_kg"]
                total_carbon += res["total_carbon_kg"]
                soc = res["bess_soc_final"]

        dispatch_metrics = {
            "total_fuel_kg":   round(total_fuel, 2),
            "total_carbon_kg": round(total_carbon, 2),
        }
        print(f"Dispatch → Total Fuel: {total_fuel:,.1f} kg  Carbon: {total_carbon:,.1f} kg")
        mlflow.log_metrics(dispatch_metrics)

        # ── Report ───────────────────────────────────────────────────────────────
        report = {
            "forecast": report_metrics,
            "dispatch": dispatch_metrics,
            "days_evaluated": n_days,
            "targets_met": {
                "mape_ok": bool(report_metrics["mape_tao_load_mw"] <= cfg["targets"]["mape"]),
                "r2_ok":   bool(report_metrics["r2_tao_load_mw"]   >= cfg["targets"]["r2"])
            }
        }
        
        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Saved → results/evaluation_report.json")


if __name__ == "__main__":
    main()

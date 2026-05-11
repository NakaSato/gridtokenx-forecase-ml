import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from collections import deque

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.pandapower_model import PhysicsEngine
from models.tcn_model import TCN, SEQ_FEATURES
from models.lgbm_model import FEATURES as LGBM_FEATURES
from domain.entities import TelemetryRow

# Fix for OpenMP conflict if running on Mac/CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    print("🚀 Starting High-Performance 1-Year Physics Simulation")
    
    # 1. Load Config & Models
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    device = "cpu"
    print(f"   Using device: {device}")
    
    # Load TCN
    ckpt = torch.load("models/tcn.pt", map_location=device, weights_only=False)
    tc_cfg = ckpt["config"]
    tcn = TCN(
        ckpt["in_features"], tc_cfg["filters"], tc_cfg["kernel_size"],
        tc_cfg["layers"], tc_cfg["forecast_horizon"]
    ).to(device)
    tcn.load_state_dict(ckpt["state_dict"])
    tcn.eval()
    
    # Load LGBM
    with open("models/lgbm.pkl", "rb") as f:
        lgbm = pickle.load(f)
        
    # Load Meta-Learner
    with open("models/meta_learner.pkl", "rb") as f:
        meta = pickle.load(f)
        
    # Load Scaler (if needed for TCN)
    # Note: The current TCN implementation might expect scaled inputs.
    # If the model was trained on scaled data, we should scale here.
    # For now, I'll assume raw if that's what the predictor does.
    
    # 2. Load Dataset
    df = pd.read_parquet("data/processed/test.parquet")
    print(f"   Test set loaded: {len(df)} rows ({df.index[0]} to {df.index[-1]})")
    
    # 3. Batch Predict LGBM (Massive Speedup)
    print("   Calculating LGBM base forecasts in batch...")
    X_lgbm = df[LGBM_FEATURES].values
    lgbm_all_preds = lgbm.predict(X_lgbm) # (N, 3)
    
    # 4. Simulation Loop
    engine = PhysicsEngine()
    window_size = tc_cfg["window_size"]
    horizon = tc_cfg["forecast_horizon"]
    
    buffer = deque(maxlen=window_size)
    results = []
    
    # Mapping for TCN features
    seq_fields = [f.lower() for f in SEQ_FEATURES]

    print("   Running Sequential TCN + Physics Engine...")
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        ts = df.index[i]
        
        # 4.1 Physics
        phys = engine.run_step(
            tao_load_mw=float(row["tao_load_mw"]),
            phangan_load_mw=float(row["phangan_load_mw"]),
            samui_load_mw=float(row["samui_load_mw"])
        )
        
        # 4.2 Update Buffer
        tel_dict = {f: float(row[f]) for f in seq_fields if f in row.index}
        # In actual system, we use Pydantic, here we use dict for speed
        buffer.append(tel_dict)
        
        # 4.3 Forecast (Every hour or per step)
        forecast_mw = np.nan
        if len(buffer) == window_size:
            # Prepare TCN input
            x_arr = np.array([[b[f] for f in seq_fields] for b in buffer], dtype=np.float32)
            x_tensor = torch.tensor(x_arr).unsqueeze(0).to(device)
            
            with torch.no_grad():
                tcn_preds = tcn(x_tensor).cpu().numpy()[0] # (horizon, 3)
            
            # Hybrid Meta-Blending (Target 0: tao_load_mw)
            lgbm_p = lgbm_all_preds[i][0]
            tcn_p = tcn_preds[:, 0]
            
            X_meta = np.column_stack([np.full(horizon, lgbm_p), tcn_p])
            final_fc = meta[0].predict(X_meta)
            
            forecast_mw = final_fc[0] # Just take the first step of forecast for metrics
            
        # 4.4 Record
        results.append({
            "timestamp": ts,
            "actual": float(row["tao_load_mw"]),
            "forecast": forecast_mw,
            "line_loss_mw": phys.get("line_loss_mw", 0.0),
            "v_tao_pu": phys.get("v_tao_pu", 1.0),
            "hvdc_loading_pct": phys.get("bottleneck_loading_pct", 0.0)
        })

    # 5. Summary & Save
    res_df = pd.DataFrame(results)
    res_df["error"] = res_df["actual"] - res_df["forecast"]
    res_df["abs_error"] = res_df["error"].abs()
    res_df["ape"] = res_df["abs_error"] / res_df["actual"] * 100
    
    metrics = res_df.dropna()
    mae = metrics["abs_error"].mean()
    mape = metrics["ape"].mean()
    
    print(f"\n{'═'*60}")
    print(f"  1-YEAR SIMULATION RESULTS")
    print(f"  MAE  : {mae:.4f} MW")
    print(f"  MAPE : {mape:.4f} %")
    print(f"  Avg Loss : {res_df['line_loss_mw'].mean():.4f} MW")
    print(f"  Min Voltage : {res_df['v_tao_pu'].min():.4f} p.u.")
    print(f"  Max Loading : {res_df['hvdc_loading_pct'].max():.2f} %")
    
    output_path = "results/year_simulation_summary.csv"
    os.makedirs("results", exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"  Full report saved to: {output_path}")
    print(f"{'═'*60}")

if __name__ == "__main__":
    main()

import argparse
import os
import sys
import time
import requests
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from research.pandapower_model import PhysicsEngine
from domain.entities import TelemetryRow
from models.lgbm_model import FEATURES as LGBM_FEATURES

API = "http://localhost:8000"
ROOT = os.path.dirname(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description="Physics-Aware Real-Time Simulator")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows to simulate")
    parser.add_argument("--output", type=str, default="results/physics_sim_report.csv", help="CSV output path")
    parser.add_argument("--speed", type=float, default=0.0, help="Sleep between steps")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    
    window = cfg["model"]["tcn"]["window_size"]
    sph = 4 if cfg["data"].get("frequency") == "15min" else 1

    try:
        r = requests.get(f"{API}/health", timeout=5)
        print(f"✅ Connected to API ({r.json()['device']})")
    except Exception as e:
        print(f"❌ API not reachable: {e}")
        sys.exit(1)

    df = pd.read_parquet(os.path.join(ROOT, "data/processed/test.parquet"))
    n_rows = args.rows if args.rows is not None else len(df)
    n_rows = min(n_rows, len(df))
    
    print(f"🚀 Starting Physics Simulation ({n_rows} steps)...")
    
    # Initialize Persistent Physics Engine
    engine = PhysicsEngine()

    results = []
    last_forecast = []
    forecast_step = 0

    for i in tqdm(range(n_rows)):
        row = df.iloc[i]
        ts = df.index[i]
        
        # Physics Engine (Reusing object for speed)
        phys = engine.run_step(
            tao_load_mw=float(row["tao_load_mw"]),
            phangan_load_mw=float(row["phangan_load_mw"]),
            samui_load_mw=float(row["samui_load_mw"])
        )
        
        # Construct TelemetryRow dictionary
        tel_dict = {}
        for field_name in TelemetryRow.model_fields:
            if field_name in row.index:
                tel_dict[field_name] = float(row[field_name])
            else:
                tel_dict[field_name] = 0.0

        payload = {
            "row": tel_dict,
            "samui_load_mw": float(row["samui_load_mw"]),
            "phangan_load_mw": float(row["phangan_load_mw"])
        }
        
        if i >= window - 1:
            payload["lgbm_features"] = {k: float(row[k]) for k in LGBM_FEATURES if k in row.index}
            # Use capacity from physics or config
            payload["circuit_forecast"] = [16.0] * (24 * sph)

        try:
            resp = requests.post(f"{API}/stream/telemetry", json=payload, timeout=30)
            if resp.status_code != 200:
                continue
            
            data = resp.json()
            if data.get("status") == "forecast":
                last_forecast = data["forecast_mw"]
                forecast_step = 0
        except Exception:
            continue

        actual = float(row["tao_load_mw"])
        fc_val = 0.0
        
        if last_forecast and forecast_step < len(last_forecast):
            fc_val = last_forecast[forecast_step]
            # Record actual vs forecast
            # (In a year-long simulation, we might want to skip these calls to speed up)
            if i % sph == 0: # record once per hour for speed
                try:
                    requests.post(f"{API}/stream/actual", json={
                        "timestamp_iso": ts.isoformat(),
                        "actual_load_mw": actual,
                        "forecast_load_mw": fc_val,
                    }, timeout=1)
                except: pass
            forecast_step += 1

        if i % 100 == 0 or i == n_rows - 1:
            grid = data.get("grid_status", {}) or {}
            results.append({
                "timestamp": ts,
                "actual": actual,
                "forecast": fc_val if fc_val > 0 else np.nan,
                "line_loss_mw": phys.get("line_loss_mw", 0.0),
                "v_tao_pu": phys.get("v_tao_pu", 1.0),
                "hvdc_loading_pct": phys.get("bottleneck_loading_pct", 0.0),
                "fuel_kg": grid.get("diesel", {}).get("total_fuel_kg", 0)
            })

    if results:
        res_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        res_df.to_csv(args.output, index=False)
        final = requests.get(f"{API}/stream/metrics").json()
        print(f"\n{'═'*60}")
        print(f"  1-YEAR PHYSICS SIMULATION COMPLETE")
        print(f"  MAE  : {final['mae']:.4f} MW")
        print(f"  MAPE : {final['mape']:.4f} %")
        print(f"  Avg Loss : {res_df['line_loss_mw'].mean():.4f} MW")
        print(f"  Max HVDC Load : {res_df['hvdc_loading_pct'].max():.2f} %")
        print(f"  Report: {args.output}")
        print(f"{'═'*60}")

if __name__ == "__main__":
    main()

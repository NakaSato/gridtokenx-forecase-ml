import pandas as pd
import requests
import time
import json
import yaml
import os

def run_simulation(data_path, api_url="http://localhost:8000"):
    print("====================================================")
    print("      GRIDTOKENX: LIVE SCADA SIMULATOR (MOCK)       ")
    print("====================================================")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run main.py first.")
        return

    # Load the locked 2023 research dataset
    df = pd.read_parquet(data_path)
    
    # Take a 24-hour sample to demonstrate the live window
    sample = df.iloc[168:168+24] # Start after the first window
    
    for i, (ts, row) in enumerate(sample.iterrows()):
        print(f"\n[TIME: {ts}]")
        
        # 1. Prepare Telemetry Payload
        payload = {
            "row": {
                "island_load_mw": float(row["Island_Load_MW"]),
                "circuit_cap_mw": float(row["Circuit_Cap_MW"]),
                "bess_soc_pct":   65.0, # Mock SoC tracking
                "net_delta_mw":   float(row["Net_Delta_MW"]),
                "load_lag_1h":    float(row["Island_Load_MW"] * 0.98),
                "load_lag_24h":   float(row["Island_Load_MW"] * 1.02)
            },
            # Provide future circuit forecast (SCADA visibility)
            "circuit_forecast": [float(row["Circuit_Cap_MW"])] * 24,
            "lgbm_features": {
                "Dry_Bulb_Temp": float(row["Dry_Bulb_Temp"]),
                "Rel_Humidity":  78.0,
                "Tourist_Index": float(row["Tourist_Index"]),
                "Solar_Irradiance": float(row["Solar_Irradiance"])
            }
        }
        
        # 2. Push to Stream API
        try:
            resp = requests.post(f"{api_url}/stream/telemetry", json=payload)
            data = resp.json()
            
            if data["status"] == "success":
                # AI Predicted this!
                fc = data["forecast_mw"][0]
                summary = data["summary"]
                print(f"  AI Forecast:  {fc:6.2f} MW")
                print(f"  Grid Delta:   {row['Net_Delta_MW']:6.2f} MW")
                
                # Check for warnings
                if row["Circuit_Cap_MW"] < 5.0:
                    print("  ⚠️ [BOTTLE-NECK] Mainland link restricted.")
                
                if summary["total_fuel_kg"] > 0:
                    print(f"  ⛽ [DIESEL ON] Predicted usage: {summary['total_fuel_kg']:.2f} kg/day")
                else:
                    print("  🔋 [BESS PRIORITY] Battery covering deficit.")
            else:
                print(f"  Status: {data['status']} (Buffering window...)")
                
        except Exception as e:
            print(f"  API Connection Error: {e}")
            break
        
        time.sleep(1) # Simulate real-time (1 second = 1 hour)

if __name__ == "__main__":
    run_simulation("data/ko_tao_grid_2023_locked.parquet")

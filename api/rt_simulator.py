"""
Real-Time Simulator — streams test.parquet row-by-row to the API.

Every hour:
  1. POST /stream/telemetry  with the current row
     - Once buffer is full, also sends lgbm_features + circuit_forecast
       to trigger a 24h forecast
  2. POST /stream/actual     with actual vs forecast for the current step
  3. Prints a live error metrics table

Usage:
    # Terminal 1 — start API
    python -m uvicorn api.serve:app --host 0.0.0.0 --port 8000

    # Terminal 2 — run simulator
    python api/rt_simulator.py [--speed 0]   # 0 = no sleep (fast replay)
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np
import pandas as pd
import requests
import yaml

API = "http://localhost:8000"
ROOT = os.path.dirname(os.path.dirname(__file__))

# ── Load config & data ────────────────────────────────────────────────────────
with open(os.path.join(ROOT, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

from models.lgbm_model import FEATURES as LGBM_FEATURES

FREQ = CFG["data"].get("frequency", "h")
SPH  = 4 if FREQ == "15min" else 1
WINDOW = CFG["model"]["tcn"]["window_size"] 
CAP_MAX = CFG["data"]["circuit_cap_max"]
BOTTLENECK_HOURS = CFG["data"]["bottleneck_hours"]


def circuit_for_hour(h: int) -> float:
    return CAP_MAX * 0.30 if h in BOTTLENECK_HOURS else CAP_MAX


def row_to_telemetry(row: pd.Series) -> dict:
    return {
        "island_load_mw":  float(row["Island_Load_MW"]),
        "load_lag_1h":     float(row["Load_Lag_1h"]),
        "load_lag_24h":    float(row["Load_Lag_24h"]),
        "bess_soc_pct":    float(row["BESS_SoC_Pct"]),
        "headroom_mw":     float(row.get("Headroom_MW", 0.0)),
        "dry_bulb_temp":   float(row["Dry_Bulb_Temp"]),
        "heat_index":      float(row["Heat_Index"]),
        "rel_humidity":    float(row["Rel_Humidity"]),
        "hour_of_day":     float(row["Hour_of_Day"]),
        "is_high_season":  float(row["Is_High_Season"]),
        "is_thai_holiday": float(row.get("Is_Thai_Holiday", 0.0)),
    }


def lgbm_features_for(row: pd.Series) -> dict:
    return {k: float(row[k]) for k in LGBM_FEATURES if k in row.index}


def circuit_forecast_for(idx, df: pd.DataFrame, pos: int) -> list:
    """24h circuit capacity forecast starting from pos (96 steps for 15min)."""
    # 24 hours = 24 * SPH steps
    horiz_steps = 24 * SPH
    hours = [df.index[min(pos + s, len(df) - 1)].hour for s in range(horiz_steps)]
    return [circuit_for_hour(h) for h in hours]


def print_header():
    print(f"\n{'─'*120}")
    print(f"  {'Step':>5}  {'Timestamp':<20}  {'Actual':>7}  {'Forecast':>8}  "
          f"{'Error':>7}  {'MAPE%':>7}  {'SoC':>7}  {'Units':>6}  {'Fuel':>10}  {'Loss(MW)':>8}")
    print(f"{'─'*120}")


def print_row(step, ts, actual, forecast, metrics):
    err = actual - forecast
    mape = metrics.get("mape") or "—"
    
    grid = metrics.get("grid_status", {})
    soc = grid.get("bess", {}).get("soc_pct", 0)
    fuel = grid.get("diesel", {}).get("total_fuel_kg", 0)
    units = grid.get("diesel", {}).get("units_active", 0)
    loss = grid.get("line_losses_mw", 0)
    
    mape_s = f"{mape:.4f}" if isinstance(mape, float) else mape
    print(f"  {step:>5}  {str(ts):<20}  {actual:>7.3f}  {forecast:>8.3f}  "
          f"{err:>+7.3f}  {mape_s:>7}  {soc:>6.1f}%  {units:>6}  {fuel:>8.1f}kg  {loss:>8.4f}")


import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Seconds between rows (0 = fast replay)")
    parser.add_argument("--rows", type=int, default=200,
                        help="Number of rows to stream")
    parser.add_argument("--output", type=str, default="results/simulation_report.csv",
                        help="Path to save CSV report")
    args = parser.parse_args()

    # Create results dir if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # ... API check logic ...

    # Check API is up
    try:
        r = requests.get(f"{API}/health", timeout=3)
        r.raise_for_status()
        print(f"✅ API online — device: {r.json()['device']}")
    except Exception as e:
        print(f"❌ API not reachable at {API}: {e}")
        print("   Start it with:  python -m uvicorn api.serve:app --port 8000")
        sys.exit(1)

    df = pd.read_parquet(os.path.join(ROOT, "data/processed/test.parquet"))
    n_rows = min(args.rows, len(df))
    print(f"   Streaming {n_rows} rows from test.parquet (window={WINDOW})")

    last_forecast: list = []   # most recent 24h forecast
    forecast_step = 0          # which step of the forecast we're comparing

    # CSV Header
    csv_headers = [
        "timestamp", "actual_mw", "forecast_mw", "error_mw", "mape_pct", 
        "bess_soc_pct", "units_active", "fuel_kg", "line_loss_mw"
    ]
    
    with open(args.output, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_headers)

        print_header()

        for i in range(n_rows):
            row = df.iloc[i]
            ts  = df.index[i]
            
            # ... payload building ...
            tel = row_to_telemetry(row)
            payload = {"row": tel}
            if i >= WINDOW - 1:
                payload["lgbm_features"]    = lgbm_features_for(row)
                payload["circuit_forecast"] = circuit_forecast_for(df.index, df, i)

            resp = requests.post(f"{API}/stream/telemetry", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "forecast":
                last_forecast = data["forecast_mw"]
                forecast_step = 0

            actual = float(row["Island_Load_MW"])
            fc_val = 0.0
            metrics = {}

            if last_forecast and forecast_step < len(last_forecast):
                fc_val = last_forecast[forecast_step]
                act_payload = {
                    "timestamp_iso": ts.isoformat(),
                    "actual_load_mw": actual,
                    "forecast_load_mw": fc_val,
                }
                ar = requests.post(f"{API}/stream/actual", json=act_payload, timeout=5)
                ar.raise_for_status()
                metrics = ar.json()["metrics"]
                
                if args.speed > 0 or i % 100 == 0 or i == n_rows - 1:
                    print_row(i, ts, actual, fc_val, metrics)
                
                forecast_step += 1
            else:
                metrics = requests.get(f"{API}/stream/metrics").json()
                if i % 50 == 0:
                    grid = data.get("grid_status", {}) or {}
                    soc = grid.get("bess", {}).get("soc_pct", 0)
                    units = grid.get("diesel", {}).get("units_active", 0)
                    fuel = grid.get("diesel", {}).get("total_fuel_kg", 0)
                    loss = grid.get("line_losses_mw", 0.0)
                    print(f"  {i:>5}  {str(ts):<20}  {actual:>7.3f}  "
                          f"{'(buffering)':>8}  {'—':>7}  {'—':>7}  {soc:>6.1f}%  {units:>6}  {fuel:>8.1f}kg  {loss:>8.4f}")

            # Write to CSV
            grid = metrics.get("grid_status", {})
            writer.writerow([
                ts.isoformat(),
                actual,
                fc_val if fc_val > 0 else None,
                actual - fc_val if fc_val > 0 else None,
                metrics.get("mape"),
                grid.get("bess", {}).get("soc_pct"),
                grid.get("diesel", {}).get("units_active"),
                grid.get("diesel", {}).get("total_fuel_kg"),
                grid.get("line_losses_mw")
            ])
            f_csv.flush()

            if args.speed > 0:
                time.sleep(args.speed)

    # Final summary ...

    # Final summary
    final = requests.get(f"{API}/stream/metrics").json()
    print(f"\n{'═'*80}")
    print(f"  FINAL METRICS  (n={final['n']} observations)")
    print(f"  MAE  = {final['mae']:.4f} MW")
    print(f"  RMSE = {final['rmse']:.4f} MW   (RMSE = √(Σ(actual−forecast)²/N))")
    print(f"  MAPE = {final['mape']:.4f}%     (MAPE = (1/N)Σ|error/actual|×100)")
    pea_ok = final['mape'] <= 10.0
    print(f"  PEA ≤10% MAPE: {'PASS ✅' if pea_ok else 'FAIL ❌'}")
    print(f"{'═'*80}")

    # Generate GridCapturX Solution Report
    report_path = args.output.replace(".csv", "_report.txt")
    with open(report_path, "w") as f_rep:
        f_rep.write("====================================================\n")
        f_rep.write("       GRIDCAPTURX: DECISION INTELLIGENCE REPORT    \n")
        f_rep.write("====================================================\n\n")
        f_rep.write("Solution Summary:\n")
        f_rep.write("GridCapturX เสนอ Decision Intelligence Platform ที่รวม Hybrid AI/ML\n")
        f_rep.write("(TCN + LightGBM + Ridge Meta-Learner) สำหรับ Forecast 24 ชม. (MAPE ≤ 10%)\n")
        f_rep.write("+ MILP Optimizer สำหรับ Recommended Schedule + pandapower Physics Validator\n")
        f_rep.write("+ Gemma 4 Cognitive Layer (Apache 2.0, on-prem ready) สำหรับ explainable\n")
        f_rep.write("recommendation, SOP-aware validation, และ actionable Early Warning\n")
        f_rep.write("ที่ลด workload P1 และตอบ 'จุดคุ้มค่า' ของ P2 ผ่าน Cost Curve Visualizer\n")
        f_rep.write("+ Counterfactual Comparison\n\n")
        f_rep.write(f"Validation Result:\n")
        f_rep.write(f"- Total Observations: {final['n']}\n")
        f_rep.write(f"- Mean Absolute Error (MAE): {final['mae']:.4f} MW\n")
        f_rep.write(f"- Root Mean Squared Error (RMSE): {final['rmse']:.4f} MW\n")
        f_rep.write(f"- Mean Absolute Percentage Error (MAPE): {final['mape']:.4f}%\n")
        f_rep.write(f"- PEA Compliance (MAPE ≤ 10%): {'PASSED' if pea_ok else 'FAILED'}\n\n")
        f_rep.write(f"- Total Fuel Consumed: {final.get('grid_status', {}).get('diesel', {}).get('total_fuel_kg', 0):.2f} kg\n")
        f_rep.write(f"- BESS Final SoC: {final.get('grid_status', {}).get('bess', {}).get('soc_pct', 0):.1f}%\n")
        f_rep.write(f"\nReport Generated at: {time.ctime()}\n")
        f_rep.write(f"CSV Data stored at: {args.output}\n")
    
    print(f"✅ Simulation complete. Report saved to: {report_path}")


if __name__ == "__main__":
    main()

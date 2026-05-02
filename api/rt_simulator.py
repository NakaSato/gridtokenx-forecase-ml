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
    print(f"\n{'─'*80}")
    print(f"  {'Step':>5}  {'Timestamp':<20}  {'Actual':>7}  {'Forecast':>8}  "
          f"{'Error':>7}  {'MAE':>7}  {'RMSE':>7}  {'MAPE%':>7}")
    print(f"{'─'*80}")


def print_row(step, ts, actual, forecast, metrics):
    err = actual - forecast
    mae  = metrics.get("mae")  or "—"
    rmse = metrics.get("rmse") or "—"
    mape = metrics.get("mape") or "—"
    mae_s  = f"{mae:.4f}"  if isinstance(mae,  float) else mae
    rmse_s = f"{rmse:.4f}" if isinstance(rmse, float) else rmse
    mape_s = f"{mape:.4f}" if isinstance(mape, float) else mape
    print(f"  {step:>5}  {str(ts):<20}  {actual:>7.3f}  {forecast:>8.3f}  "
          f"{err:>+7.3f}  {mae_s:>7}  {rmse_s:>7}  {mape_s:>7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Seconds between rows (0 = fast replay)")
    parser.add_argument("--rows", type=int, default=200,
                        help="Number of rows to stream")
    args = parser.parse_args()

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

    print_header()

    for i in range(n_rows):
        row = df.iloc[i]
        ts  = df.index[i]
        h   = ts.hour

        # Build telemetry payload
        tel = row_to_telemetry(row)

        # Once buffer will be full, attach lgbm_features + circuit_forecast
        payload = {"row": tel}
        if i >= WINDOW - 1:
            payload["lgbm_features"]    = lgbm_features_for(row)
            payload["circuit_forecast"] = circuit_forecast_for(df.index, df, i)

        # POST telemetry
        resp = requests.post(f"{API}/stream/telemetry", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # If a new forecast was returned, reset comparison window
        if data.get("status") == "forecast":
            last_forecast = data["forecast_mw"]
            forecast_step = 0

        # POST actual vs forecast (once we have a forecast to compare)
        actual = float(row["Island_Load_MW"])
        if last_forecast and forecast_step < len(last_forecast):
            fc_val = last_forecast[forecast_step]
            act_payload = {
                "timestamp_iso":   ts.isoformat(),
                "actual_load_mw":  actual,
                "forecast_load_mw": fc_val,
            }
            ar = requests.post(f"{API}/stream/actual", json=act_payload, timeout=5)
            ar.raise_for_status()
            metrics = ar.json()["metrics"]
            
            # Reduce printing if speed is 0 (fast replay)
            if args.speed > 0 or i % 100 == 0 or i == n_rows - 1:
                print_row(i, ts, actual, fc_val, metrics)
            
            forecast_step += 1
        else:
            # Still filling buffer — just show actual, no forecast yet
            if i % 50 == 0:
                print(f"  {i:>5}  {str(ts):<20}  {actual:>7.3f}  "
                      f"{'(buffering)':>8}  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>7}")

        if args.speed > 0:
            time.sleep(args.speed)

    # Final summary
    final = requests.get(f"{API}/stream/metrics").json()
    print(f"\n{'═'*80}")
    print(f"  FINAL METRICS  (n={final['n']} observations)")
    print(f"  MAE  = {final['mae']:.4f} MW")
    print(f"  RMSE = {final['rmse']:.4f} MW   (RMSE = √(Σ(actual−forecast)²/N))")
    print(f"  MAPE = {final['mape']:.4f}%     (MAPE = (1/N)Σ|error/actual|×100)")
    pea_ok = final['mape'] <= 10.0
    print(f"  PEA ≤10% MAPE: {'PASS ✅' if pea_ok else 'FAIL ❌'}")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()

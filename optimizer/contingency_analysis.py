"""
N-1 Contingency Analysis — Stress Test

Simulates a total failure of the 115 kV submarine cable during the 
2026 Songkran Festival peak. Checks if local resources (Diesel + BESS)
can prevent a blackout.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yaml
from optimizer.pea_dispatch_opt import pea_optimize

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def run_analysis():
    cfg = load_cfg()
    df = pd.read_parquet("data/processed/test.parquet")
    
    # 1. Identify Songkran Peak (April 13, 2026)
    target_date = "2026-04-13"
    day_df = df[df.index.strftime("%Y-%m-%d") == target_date]
    if len(day_df) < 96:
        print(f"Error: Not enough data for {target_date}. Found {len(day_df)} steps.")
        return

    load_mw = day_df["Island_Load_MW"].values
    
    # 2. Simulate N-1 Cable Failure (Total loss of mainland supply)
    # Failure from 18:00 to 02:00 (next day simulation window covers 24h)
    # 18:00 is step 18*4 = 72
    # 02:00 (next day) is step (18+8)*4 = 104 -> since window is 96, we fail until end of day
    circuit_mw = np.full(96, cfg["data"]["circuit_cap_max"])
    failure_start = 72  # 18:00
    circuit_mw[failure_start:] = 0.0
    
    cap = cfg["bess"]["capacity_mwh"]
    print("=" * 70)
    print(f"  N-1 CONTINGENCY ANALYSIS: CABLE FAILURE @ {target_date}")
    print(f"  Incident: Total Failure of 115 kV NO.3 starting 18:00")
    print(f"  Local Resources: Diesel (10 MW) {'+ BESS (' + str(cap) + ' MWh)' if cap > 0 else '(No BESS)'}")
    print("=" * 70)

    # 3. Run Optimization
    initial_soc = 0.65 if cap > 0 else 0.0
    res = pea_optimize(load_mw, circuit_mw, initial_soc=initial_soc, cfg=cfg)

    # 4. Results
    print(f"\n  Survival Status: {'SUCCESS ✅' if res['diesel_hours'] <= 24 else 'FAILED ❌'}")
    print(f"  Max Load during Failure: {np.max(load_mw[failure_start:]):.2f} MW")
    print(f"  Local Diesel Usage: {res['diesel_hours']/4.0:.1f} hours")

    if cap > 0:
        print(f"  BESS Depletion: {initial_soc*100:.1f}% -> {res['bess_soc_final']*100:.1f}%")

    print(f"\n  {'h':>5} {'load':>6} {'circ':>6} {'diesel':>7} {'bess':>7} {'SoC%':>6} {'Status':>8}")
    print("  " + "─" * 60)

    for i, s in enumerate(res["schedule"]):
        # h:m format
        hour = i // 4
        minute = (i % 4) * 15
        ts = f"{hour:02d}:{minute:02d}"

        soc_pct = (s.soc_mwh / cap * 100) if cap > 0 else 0.0
        status = "CRITICAL" if s.circuit_mw == 0 else "NORMAL"
        if i >= failure_start - 4: # show a bit before failure
            print(f"  {ts:>5} {s.load_mw:>6.2f} {s.circuit_mw:>6.2f} "
                  f"{s.diesel_mw:>7.2f} {s.bess_mw:>+7.2f} {soc_pct:>6.1f} {status:>8}")

    # 5. Check for Load Shedding (Unmet load)
    # In pea_optimize, if constraints are violated it might not find a solution
    # but run_dispatch is more robust for checking coverage
    from optimizer.dispatch import run_dispatch
    rb_res = run_dispatch(load_mw, circuit_mw, initial_soc=initial_soc, cfg=cfg)
    
    unmet = [max(0, s.load_mw - (s.circuit_mw + s.diesel_mw + s.bess_mw)) for s in rb_res]
    total_unmet = sum(unmet) / 4.0
    
    print("\n" + "=" * 70)
    if total_unmet < 0.01:
        print("  ✅ VERDICT: Ko Tao survives the 115 kV failure.")
        print("  The ensemble of local Diesel + BESS successfully bridged the gap.")
    else:
        print(f"  ❌ VERDICT: Blackout Detected! Unmet load: {total_unmet:.2f} MWh")
        print("  Recommendation: Increase Diesel spinning reserve or BESS capacity.")
    print("=" * 70)

if __name__ == "__main__":
    run_analysis()

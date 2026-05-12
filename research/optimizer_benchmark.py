"""
Optimizer Benchmark: MILP vs ISCA vs Greedy Dispatch
=====================================================
Analyzes the optimality gap and computational performance of different dispatch solvers.
Focuses on non-linear fuel savings and grid stability.
"""
import os, sys
import time
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from typing import Dict

# Setup paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain.dispatch import run_dispatch, schedule_summary
from optimizer.pea_dispatch_opt import pea_optimize
from optimizer.isca import isca_optimize

# Setup paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

def run_benchmark(days: int = 7):
    print(f"🚀 Starting Optimizer Benchmark (Days={days})...")
    
    # Load test data
    df = pd.read_parquet(os.path.join(ROOT, "data/processed/test.parquet"))
    # We'll pick a slice that includes some bottleneck periods
    # Frequency is 15min (SPH=4), so 24h = 96 steps
    sph = 4 if CFG["data"].get("frequency") == "15min" else 1
    total_steps = days * 24 * sph
    
    # Check if we have enough data
    if len(df) < total_steps:
        total_steps = len(df)
        days = total_steps // (24 * sph)

    # We'll run day-by-day (sliding window) to simulate real operations
    results = []
    
    soc_greedy = 0.65
    soc_milp = 0.65
    soc_isca = 0.65

    for d in range(days):
        start = d * 24 * sph
        end = start + 24 * sph
        
        load_slice = df["tao_load_mw"].values[start:end]
        
        # Use realistic capacity logic instead of potentially bad data in parquet
        from api.rt_simulator import circuit_for_hour
        hours = [df.index[min(start + s, len(df) - 1)].hour for s in range(24 * sph)]
        circ_slice = np.array([circuit_for_hour(h) for h in hours])
        
        print(f"\n📅 Day {d+1}: Load Avg={load_slice.mean():.2f} MW, Min Capacity={circ_slice.min():.2f} MW")
        
        # 1. Greedy (Rule-based)
        t0 = time.time()
        sched_greedy = run_dispatch(load_slice, circ_slice, initial_soc=soc_greedy, cfg=CFG)
        t_greedy = time.time() - t0
        summary_greedy = schedule_summary(sched_greedy)
        soc_greedy = summary_greedy["bess_soc_final"]
        
        # 2. MILP (Linear Approximation)
        t0 = time.time()
        res_milp = pea_optimize(load_slice, circ_slice, initial_soc=soc_milp, cfg=CFG)
        t_milp = time.time() - t0
        if "total_fuel_kg" in res_milp:
            soc_milp = res_milp["bess_soc_final"]
        else:
            res_milp = {"total_fuel_kg": np.nan, "total_cost_thb": np.nan}

        # 3. ISCA (Metaheuristic - Non-linear)
        t0 = time.time()
        # ISCA currently expects 24 steps (hourly) in its internal loop if not adjusted
        # But we can pass the raw 96 steps if we adjust the ISCA script or just slice hourly
        # For this benchmark, let's assume hourly steps for simplicity or adjust ISCA
        res_isca = isca_optimize(load_slice, circ_slice, initial_soc=soc_isca, cfg=CFG)
        t_isca = time.time() - t0
        soc_isca = res_isca["bess_soc_final"]

        results.append({
            "day": d + 1,
            "greedy_fuel": summary_greedy["total_fuel_kg"],
            "greedy_time": t_greedy,
            "milp_fuel": res_milp["total_fuel_kg"],
            "milp_time": t_milp,
            "isca_fuel": res_isca["total_fuel_kg"],
            "isca_time": t_isca,
            "greedy_soc": summary_greedy["bess_soc_final"],
            "milp_soc": res_milp.get("bess_soc_final"),
            "isca_soc": res_isca["bess_soc_final"]
        })
        
        print(f"   Greedy: {summary_greedy['total_fuel_kg']:>8.2f} kg ({t_greedy:.4f}s)")
        print(f"   MILP:   {res_milp['total_fuel_kg']:>8.2f} kg ({t_milp:.4f}s)")
        print(f"   ISCA:   {res_isca['total_fuel_kg']:>8.2f} kg ({t_isca:.4f}s)")

    # Aggregation
    res_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("🔥 OPTIMIZER BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Fuel Consumption (kg):")
    print(f"  Greedy : {res_df['greedy_fuel'].sum():.2f}")
    print(f"  MILP   : {res_df['milp_fuel'].sum():.2f} (Gap vs Greedy: {((res_df['milp_fuel'].sum()/res_df['greedy_fuel'].sum())-1)*100:+.2f}%)")
    print(f"  ISCA   : {res_df['isca_fuel'].sum():.2f} (Gap vs Greedy: {((res_df['isca_fuel'].sum()/res_df['greedy_fuel'].sum())-1)*100:+.2f}%)")
    print("\nAvg Solving Time (s):")
    print(f"  Greedy : {res_df['greedy_time'].mean():.4f}")
    print(f"  MILP   : {res_df['milp_time'].mean():.4f}")
    print(f"  ISCA   : {res_df['isca_time'].mean():.4f}")
    print("="*60)

    # Plotting
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(res_df["day"], res_df["greedy_fuel"], marker='o', label="Greedy (Baseline)")
    plt.plot(res_df["day"], res_df["milp_fuel"], marker='s', label="MILP (Linear)")
    plt.plot(res_df["day"], res_df["isca_fuel"], marker='^', label="ISCA (Metaheuristic)")
    plt.title("Daily Fuel Consumption by Optimizer Type")
    plt.xlabel("Day")
    plt.ylabel("Fuel (kg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ROOT, "results/optimizer_comparison.png"))
    print(f"\n📊 Comparison plot saved to results/optimizer_comparison.png")

if __name__ == "__main__":
    run_benchmark(days=7)

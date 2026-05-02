"""
Monte Carlo Execution Engine — GridTokenX
Exhaustively evaluates grid resilience through randomized stochastic scenarios.
"""
import os, sys, yaml, random, multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from research.pandapower_model import create_ko_tao_network, verify_dispatch_stability

# Load config
def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def generate_random_scenario(rng):
    """Generates a randomized grid state based on historical/physics bounds."""
    # 1. Randomized Loads (Log-normal to avoid negative, biased towards peak)
    # Estimates based on audit: Tao (5-15), Phangan (15-35), Samui (40-110)
    tao_load = rng.lognormal(mean=np.log(8.0), sigma=0.3)
    phangan_load = rng.lognormal(mean=np.log(20.0), sigma=0.3)
    samui_load = rng.lognormal(mean=np.log(65.0), sigma=0.3)

    # 2. Randomized Contingencies (Failure Probabilities)
    # p_failure per scenario (not per hour)
    p_fail_hvdc = 0.02    # Mainland-Samui Circuit 3
    p_fail_inter = 0.01   # Samui-Phangan
    p_fail_distal = 0.01  # Phangan-Tao

    faults = {
        "hvdc_c3_fault": rng.random() < p_fail_hvdc,
        "inter_island_fault": rng.random() < p_fail_inter,
        "distal_fault": rng.random() < p_fail_distal
    }

    # 3. Asset Availability
    # Diesel generators can fail to start or trip
    p_diesel_fail = 0.05
    tao_diesel_avail = rng.random() > p_diesel_fail
    samui_diesel_avail = rng.random() > p_diesel_fail

    return {
        "loads": {"tao": tao_load, "phangan": phangan_load, "samui": samui_load},
        "faults": faults,
        "assets": {"tao_diesel": tao_diesel_avail, "samui_diesel": samui_diesel_avail}
    }

def simulate_scenario(scenario_idx, seed):
    """Executes a single physics simulation for a randomized scenario."""
    rng = np.random.default_rng(seed)
    sc = generate_random_scenario(rng)
    
    # Map scenario to pandapower parameters
    # Note: Currently create_ko_tao_network doesn't support easy line removal for N-1
    # We will simulate N-1 by reducing circuit capacity if fault occurs
    
    # If fault, we effectively lose that supply
    # (Simplified for now: faults trigger local diesel response)
    tao_diesel = 10.0 if sc["assets"]["tao_diesel"] and (sc["faults"]["distal_fault"] or sc["loads"]["tao"] > 8.0) else 0.0
    samui_diesel = 10.0 if sc["assets"]["samui_diesel"] and (sc["faults"]["hvdc_c3_fault"] or sc["loads"]["samui"] > 90.0) else 0.0

    res = verify_dispatch_stability(
        tao_load_mw=sc["loads"]["tao"],
        phangan_load_mw=sc["loads"]["phangan"],
        samui_load_mw=sc["loads"]["samui"],
        tao_diesel_mw=tao_diesel,
        samui_diesel_mw=samui_diesel
    )

    # Evaluation Criteria
    survival = res.get("stable", False) and res.get("voltage_ok", False)
    
    return {
        "id": scenario_idx,
        "tao_load": sc["loads"]["tao"],
        "ph_load": sc["loads"]["phangan"],
        "sm_load": sc["loads"]["samui"],
        "v_tao": res.get("v_tao_pu", 0.0),
        "hvdc_loading": res.get("bottleneck_loading_pct", 0.0),
        "survival": survival,
        "fault_hvdc": sc["faults"]["hvdc_c3_fault"],
        "fault_distal": sc["faults"]["distal_fault"]
    }

def run_monte_carlo(n_iterations=1000):
    print(f"🚀 Starting Monte Carlo Power Test (N={n_iterations})...")
    
    # Using ProcessPoolExecutor for parallel physics
    # Note: On macOS, avoid spawn/fork issues by keeping logic inside if __name__
    results = []
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(simulate_scenario, i, random.randint(0, 1000000)) for i in range(n_iterations)]
        for f in tqdm(futures, desc="Simulating"):
            results.append(f.result())

    df = pd.DataFrame(results)
    
    # ── Analysis ──
    lolp = (1 - df["survival"].mean()) * 100
    print("\n" + "="*80)
    print("  MONTE CARLO STOCHASTIC ANALYSIS RESULTS")
    print("="*80)
    print(f"  • Total Scenarios:   {len(df)}")
    print(f"  • Survival Rate:     {df['survival'].mean()*100:.2f}%")
    print(f"  • LOLP (Failure %):  {lolp:.2f}%")
    print(f"  • Mean Tao Voltage:  {df['v_tao'].mean():.4f} p.u.")
    print(f"  • Peak HVDC Load:    {df['hvdc_loading'].max():.1f}%")
    
    # Save Report
    df.to_csv("results/monte_carlo_results.csv", index=False)
    print(f"\n  ✅ Raw results saved to results/monte_carlo_results.csv")
    print("="*80)

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    # Small test run first
    n = 100
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_monte_carlo(n)

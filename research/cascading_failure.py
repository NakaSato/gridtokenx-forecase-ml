"""
Extreme Resilience Testing: N-1-1 and Cascading Failure Simulation.
Evaluates the grid's ability to survive multiple concurrent submarine cable failures.
"""
import os
import sys
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research.pypsa_model import create_pypsa_network

def simulate_cascading_failure(tao_load, phangan_load, samui_load, failed_lines=None):
    """
    Simulates a grid state with multiple failed lines and checks for load shedding.
    """
    n = create_pypsa_network()
    
    # Add Loads
    n.add("Load", "Tao_L",     bus="Tao",     p_set=tao_load)
    n.add("Load", "Phangan_L", bus="Phangan", p_set=phangan_load)
    n.add("Load", "Samui_L",   bus="Samui",   p_set=samui_load)

    # Simulate Failures
    if failed_lines:
        for line in failed_lines:
            if line in n.lines.index:
                n.lines.at[line, "s_nom"] = 0.0
                print(f"  [CRITICAL] Line Failure: {line}")

    # Add Load Shedding (as high-cost generators)
    # This allows the solver to find a solution even if the grid is unstable
    n.add("Generator", "Tao Shedding", bus="Tao", p_nom=1000, marginal_cost=1e6)
    n.add("Generator", "Phangan Shedding", bus="Phangan", p_nom=1000, marginal_cost=1e6)
    n.add("Generator", "Samui Shedding", bus="Samui", p_nom=1000, marginal_cost=1e6)

    try:
        n.optimize(solver_name='highs')
        
        shedding = {
            "Tao": n.generators_t.p.at[n.snapshots[0], "Tao Shedding"],
            "Phangan": n.generators_t.p.at[n.snapshots[0], "Phangan Shedding"],
            "Samui": n.generators_t.p.at[n.snapshots[0], "Samui Shedding"]
        }
        
        total_shedding = sum(shedding.values())
        is_blackout = total_shedding > 0.1
        
        return {
            "is_blackout": is_blackout,
            "total_shedding_mw": total_shedding,
            "island_shedding": shedding,
            "diesel_usage": {
                "Tao": n.generators_t.p.at[n.snapshots[0], "Tao Diesel"],
                "Phangan": n.generators_t.p.at[n.snapshots[0], "Phangan Diesel"],
                "Samui": n.generators_t.p.at[n.snapshots[0], "Samui Diesel"]
            }
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def run_stress_test_cycle():
    print(f"\n{'='*60}")
    print(f"  GRIDTOKENX: N-1-1 CASCADING FAILURE ANALYSIS")
    print(f"{'='*60}")
    
    # 1. Base Case (Normal)
    print("\n[SCENARIO 1] Baseline (All Lines Healthy, Peak Load)")
    res = simulate_cascading_failure(8.5, 25.0, 85.0)
    print(f"  Status: {'Stable' if not res['is_blackout'] else 'Blackout'}")
    print(f"  Diesel Support: {res['diesel_usage']}")

    # 2. N-1 (Mainland Cable Failure)
    print("\n[SCENARIO 2] N-1 (Mainland-Samui Failure)")
    res = simulate_cascading_failure(8.5, 25.0, 85.0, failed_lines=["Mainland-Samui"])
    print(f"  Status: {'Stable' if not res['is_blackout'] else 'CRITICAL BLACKOUT'}")
    if res['is_blackout']:
        print(f"  Load Shedding: {res['total_shedding_mw']:.2f} MW")

    # 3. N-1-1 (Mainland + Samui-Phangan Failure)
    print("\n[SCENARIO 3] N-1-1 (Mainland-Samui AND Samui-Phangan Failure)")
    res = simulate_cascading_failure(8.5, 25.0, 85.0, failed_lines=["Mainland-Samui", "Samui-Phangan"])
    print(f"  Status: {'Stable' if not res['is_blackout'] else 'EXTREME BLACKOUT'}")
    if res['is_blackout']:
        print(f"  Impact: Ko Tao and Phangan totally isolated.")
        print(f"  Load Shedding: {res['total_shedding_mw']:.2f} MW")

    # 4. Critical Node Isolation (Phangan-Tao Failure)
    print("\n[SCENARIO 4] Ko Tao Isolation (Phangan-Tao Failure)")
    res = simulate_cascading_failure(8.5, 25.0, 85.0, failed_lines=["Phangan-Tao"])
    print(f"  Status: {'Stable' if not res['is_blackout'] else 'KO TAO BLACKOUT'}")
    if res['is_blackout']:
        print(f"  Tao Shedding: {res['island_shedding']['Tao']:.2f} MW")
        print(f"  Note: Tao Diesel (10MW) vs Tao Load (8.5MW). Should survive if Diesel is ready.")

    print(f"\n{'='*60}")
    print(f"  RESILIENCE REPORT GENERATED")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_stress_test_cycle()

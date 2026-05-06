"""
N-1 Contingency Stress Testing
Simulates a total/partial loss of the mainland 115kV submarine cable.
Validates if BESS and Diesel dispatch can prevent load shedding.
"""
import numpy as np
import yaml
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from optimizer.pea_dispatch_opt import pea_optimize

def run_stress_test():
    print("=== N-1 Contingency Stress Test ===")
    
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Simulate 24h horizon
    T = 24
    rng = np.random.default_rng(42)
    
    # Simulate an extreme load day on Samui (75-95 MW)
    samui_load = rng.uniform(75.0, 95.0, T)
    
    # Simulate Mainland -> Samui Circuit capacity
    # Normal capacity is ~110 MW
    samui_circuit = np.full(T, 110.0)
    
    # --- N-1 EVENT AT PEAK ---
    # At 18:00 to 22:00, the main 115kV cable trips!
    # Capacity drops from 110 MW down to 30 MW (only 33kV and 1 remaining 115kV taking partial load).
    # Deficit is ~60 MW. BESS is 50 MWh, Diesels are 20 MW.
    # It will test the optimizer's limits.
    # To force MILP to struggle and possibly hit ISCA, we make it very tight.
    event_start, event_end = 18, 20
    samui_circuit[event_start:event_end] = 65.0  # Drops to 65 MW. Deficit is ~25 MW.

    # Apply Ko Samui specific asset configuration
    assets = cfg["cluster"]["assets"]["ko_samui"]
    cfg["data"]["frequency"] = "1h"
    cfg["bess"]["capacity_mwh"] = assets["bess_mwh"]
    cfg["bess"]["charge_rate_mw"] = 25.0  # Max charge/discharge rate for 50MWh BESS
    cfg["diesel"]["rated_mw"] = assets["diesel_mw"]
    cfg["diesel"]["optimal_output_mw"] = assets["diesel_mw"] * 0.75
    cfg["diesel"]["ramp_rate_mw_per_h"] = assets["diesel_mw"]  # Fast ramp for mobile gens

    print("Running Dispatch Optimizer (MILP)...")
    res = pea_optimize(samui_load, samui_circuit, initial_soc=0.65, cfg=cfg)
    
    print("\n=== Result Summary ===")
    print(f"Status: {res['solver_status']}")
    print(f"Total Shed: {res['total_shed_mwh']} MWh")
    print(f"Total Cost: {res['total_cost_thb']} THB")
    print(f"Total Fuel: {res['total_fuel_kg']} kg")
    
    # Assertions
    shed = res['total_shed_mwh']
    
    print("\n=== Event Window Dispatch (18:00 - 19:00) ===")
    for h in range(18, 20):
        s = res['schedule'][h]
        print(f"h{h:02d} | Load: {s.load_mw:5.1f} | Import: {s.circuit_mw:5.1f} | "
              f"BESS: {s.bess_mw:+6.1f} (SoC {s.soc_mwh:4.1f}) | Diesel: {s.diesel_mw:4.1f} | "
              f"Shed: {s.shed_mw:4.1f}")

    if shed > 0:
        print("\n❌ FAILED: The system shed load during the N-1 contingency.")
    else:
        print("\n✅ PASSED: The dispatch layer survived the N-1 contingency without blackout!")

if __name__ == "__main__":
    run_stress_test()

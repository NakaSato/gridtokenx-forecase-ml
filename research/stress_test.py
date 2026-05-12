import os
import sys
import numpy as np
import pandas as pd
import yaml
import json
from tqdm import tqdm

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research.pypsa_model import create_pypsa_network

def run_n1_contingency(tao_load, phangan_load, samui_load):
    """
    Simulates an N-1 event: Total failure of the Mainland-Samui 115kV cable.
    Checks if island generators can prevent a blackout.
    """
    n = create_pypsa_network()
    
    # 1. Set loads
    n.add("Load", "Tao_L",     bus="Tao",     p_set=tao_load)
    n.add("Load", "Phangan_L", bus="Phangan", p_set=phangan_load)
    n.add("Load", "Samui_L",   bus="Samui",   p_set=samui_load)

    # 2. Trigger N-1 Contingency: Disable the mainland link
    # Line 0 is Mainland-Samui in create_pypsa_network
    n.lines.at["Mainland-Samui", "s_nom"] = 0.0
    # Also disable the Mainland generator to be safe
    n.generators.at["Mainland Import", "p_nom"] = 0.0

    # 3. Run LOPF
    try:
        n.optimize(solver_name='highs')
        status = n.model.status
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

    # 4. Analyze results
    total_load = tao_load + phangan_load + samui_load
    served_load = n.generators_t.p.sum(axis=1).iloc[0]
    ens = max(0.0, total_load - served_load)
    
    dispatch = n.generators_t.p.iloc[0].to_dict()
    
    return {
        "status": status,
        "total_load_mw": total_load,
        "served_load_mw": served_load,
        "energy_not_served_mw": ens,
        "survival": "SUCCESS" if ens < 0.1 and status == "ok" else "FAILURE",
        "dispatch": dispatch
    }

def main():
    print("🚨 Starting N-1 Contingency Stress Test")
    print("   Scenario: Total failure of Mainland-Samui 115kV Submarine Cable")
    
    # Test cases: Low, Medium, High load
    scenarios = [
        {"name": "Low Load (Night)", "tao": 5.0, "pha": 15.0, "sam": 45.0},
        {"name": "Medium Load (Day)", "tao": 7.5, "pha": 22.0, "sam": 70.0},
        {"name": "Peak Load (Songkran)", "tao": 10.0, "pha": 30.0, "sam": 95.0},
    ]
    
    results = []
    for sc in scenarios:
        print(f"\n   Running scenario: {sc['name']}...")
        res = run_n1_contingency(sc['tao'], sc['pha'], sc['sam'])
        res["scenario"] = sc["name"]
        results.append(res)
        
        print(f"   Status: {res['status']} | Survival: {res['survival']}")
        print(f"   ENS: {res['energy_not_served_mw']:.2f} MW")
        if res['survival'] == "FAILURE":
            print(f"   ⚠️ WARNING: Blackout risk in this scenario!")

    # Final Report
    report_path = "results/stress_test_report.json"
    os.makedirs("results", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Stress test complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()

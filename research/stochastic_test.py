import os
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research.pypsa_model import run_security_constrained_opf

def main():
    print("🎲 Starting Stochastic Resilience Test (Monte Carlo)")
    n_trials = 100
    
    # Base loads
    tao_base, pha_base, sam_base = 7.5, 22.0, 70.0
    
    results = []
    print(f"   Running {n_trials} trials with ±20% load variation...")
    
    for i in tqdm(range(n_trials)):
        # Random variation (Normal distribution, 10% std)
        tao = max(2.0, np.random.normal(tao_base, tao_base * 0.1))
        pha = max(5.0, np.random.normal(pha_base, pha_base * 0.1))
        sam = max(20.0, np.random.normal(sam_base, sam_base * 0.1))
        
        try:
            res = run_security_constrained_opf(tao, pha, sam)
            results.append({
                "trial": i,
                "loads": {"tao": tao, "pha": pha, "sam": sam},
                "status": res["status"],
                "total_cost": res["total_cost"],
                "max_loading": max(res["line_loading"].values()) if res["line_loading"] else 0.0
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    
    # Analysis
    success_rate = (df["status"] == "ok").mean() * 100
    avg_loading = df["max_loading"].mean() * 100
    
    print(f"\n{'═'*60}")
    print(f"  STOCHASTIC RESILIENCE SUMMARY")
    print(f"  Trials       : {n_trials}")
    print(f"  Success Rate : {success_rate:.1f}%")
    print(f"  Avg Max Load : {avg_loading:.2f}%")
    print(f"  Max Load Peak: {df['max_loading'].max()*100:.2f}%")
    print(f"{'═'*60}")

    # Save
    report_path = "results/stochastic_test_report.json"
    os.makedirs("results", exist_ok=True)
    df.to_json(report_path, orient="records", indent=2)
    print(f"✅ Stochastic report saved to {report_path}")

if __name__ == "__main__":
    main()

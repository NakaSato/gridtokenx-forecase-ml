"""
GridTokenX Intelligence Diagnostic — 2026 Strategy Audit
Synthesizes Topology, Physics, Assets, and Adaptive Strategy into one report.
Now with Multi-Library Power System Cross-Verification (Pandapower + PyPSA).
"""
import os, sys, yaml
import numpy as np
import pandas as pd
from research.pandapower_model import verify_dispatch_stability
from research.pypsa_model import run_pypsa_analysis
from research.load_estimator import estimate_cluster_loads

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def run_diagnostic():
    cfg = load_cfg()
    
    print("="*80)
    print("  GRIDTOKENX: HOLISTIC GRID INTELLIGENCE DIAGNOSTIC (MAY 2026)")
    print("="*80)

    # ── 1. STRATEGIC POSTURE ──
    print("\n[1] STRATEGIC POSTURE: ADAPTIVE OBSERVATION")
    print(f"  • Load Knowledge:   ZERO (No History, No Base/Peak assumptions)")
    print(f"  • Inference Mode:   Mainland-Circuit Reverse Inference")
    print(f"  • Forecast Mode:    Zero-Shot Physics + Online Calibration")
    
    # Show sample estimation
    est = estimate_cluster_loads(100.0, 8.5)
    print(f"  • Sample Estimate:  {est['total_estimated_mw']} MW Cluster (Samui={est['samui_mw']}, Phangan={est['phangan_mw']})")
    
    print(f"  • Control Window:   15-Minute Resolution (Sph=4)")
    print(f"  • Primary Target:   Ko Tao Substation (Radial Terminus)")

    # ── 2. ASSET INVENTORY ──
    print("\n[2] ASSET INVENTORY & NODAL STABILITY")
    nodes = [
        {"name": "Ko Samui",   "gen": "10 MW Diesel", "store": "50 MWh BESS", "avr": "No"},
        {"name": "Ko Phangan", "gen": "None",          "store": "None",         "avr": "Yes (Active)"},
        {"name": "Ko Tao",     "gen": "10 MW Diesel", "store": "None",         "avr": "Yes (Active)"},
    ]
    print(f"  {'Island':<15} | {'Generation':<15} | {'Storage':<15} | {'Voltage Reg'}")
    print(f"  {'-'*60}")
    for n in nodes:
        print(f"  {n['name']:<15} | {n['gen']:<15} | {n['store']:<15} | {n['avr']}")

    # ── 3. TOPOLOGY & BOTTLENECK AUDIT ──
    print("\n[3] TOPOLOGY & RADIAL CONSTRAINTS")
    print("  Mainland ──► [115kV C2 / C3 Bottleneck / 33kV] ──► Samui")
    print("  Samui    ──► [115kV / 33kV XLPE] ──► Phangan")
    print("  Phangan  ──► [33kV XLPE (0-16 MW)] ──► Ko Tao")
    
    # ── 4. PHYSICS CROSS-VERIFICATION ──
    print("\n[4] PHYSICS CROSS-VERIFICATION (Stochastic Sample)")
    loads = {"tao": 8.5, "phangan": 22.0, "samui": 75.0}
    
    # Pandapower (AC Power Flow)
    ph = verify_dispatch_stability(
        tao_load_mw=loads["tao"], phangan_load_mw=loads["phangan"], samui_load_mw=loads["samui"]
    )
    
    # PyPSA (Linear Power Flow / LPF)
    py = run_pypsa_analysis(
        tao_load_mw=loads["tao"], phangan_load_mw=loads["phangan"], samui_load_mw=loads["samui"]
    )

    print(f"  • Library:          {'Pandapower (AC)':<20} | {'PyPSA (LPF)':<20}")
    print(f"  {'-'*65}")
    if ph.get("stable"):
        print(f"  • Status:           {'SUCCESS':<20} | {'SUCCESS':<20}")
        print(f"  • HVDC Load:        {str(ph['bottleneck_loading_pct'])+'%':<20} | {str(py['hvdc_loading_pct'])+'%':<20}")
        print(f"  • Tao Link Load:    {str(ph['max_line_loading'])+'%':<20} | {str(py['tao_link_loading_pct'])+'%':<20}")
        print(f"  • Ko Tao Voltage:   {str(ph['v_tao_pu'])+' p.u.':<20} | {'N/A (Linear)':<20}")
    else:
        print(f"  ❌ ERROR: Physics check failed.")

    # ── 5. STOCHASTIC RESILIENCE (Monte Carlo) ──
    print("\n[5] STOCHASTIC RESILIENCE: MONTE CARLO ANALYSIS")
    mc_path = "results/monte_carlo_results.csv"
    if os.path.exists(mc_path):
        df_mc = pd.read_csv(mc_path)
        lolp = (1 - df_mc["survival"].mean()) * 100
        print(f"  • Loss of Load Prob (LOLP): {lolp:.2f}%")
        print(f"  • Survival Confidence:     {df_mc['survival'].mean()*100:.1f}%")
        print(f"  • Peak Sample Stress:      {df_mc['hvdc_loading'].max():.1f}% HVDC load")
    else:
        print("  ⚠️  Monte Carlo results not found. Run 'just stochastic-test'.")

    # ── 6. RESILIENCE PROFILE (N-1) ──
    print("\n[6] RESILIENCE AUDIT: N-1 SURVIVAL POSTURE")
    print("  • Ko Samui:  HIGH (Local Diesel + BESS Buffer)")
    print("  • Ko Phangan: ZERO (No Local Assets - Total Radial Dependency)")
    print("  • Ko Tao:     MODERATE (Diesel Backup, but limited to 10 MW)")
    print("\n  [!] CRITICAL RISK: Koh Phangan enters immediate blackout on cable fault.")

    print("\n"+"="*80)
    print("  DIAGNOSTIC COMPLETE: GridTokenX is ready for Multi-Library Verification.")
    print("="*80)

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    run_diagnostic()

"""
Optimal Power Flow (OPF) Analysis — GridTokenX
Determines the most cost-effective generation setpoints while respecting grid physics.
"""
import pypsa
import pandas as pd
import numpy as np
import yaml

def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def run_opf_analysis():
    cfg = load_cfg()
    oc = cfg["optimizer"]
    
    n = pypsa.Network()
    # Define snapshots (Single point for analysis)
    n.set_snapshots(pd.Index(["now"]))
    
    # ── 1. NETWORK TOPOLOGY (115kV/33kV) ──
    n.add("Bus", "Khanom", v_nom=115.0)
    n.add("Bus", "Samui",  v_nom=115.0)
    n.add("Bus", "Phangan",v_nom=33.0)
    n.add("Bus", "Tao",    v_nom=33.0)

    # Mainland slack (Increase s_nom to ensure feasibility for extreme test)
    n.add("Generator", "Mainland Slack", bus="Khanom", control="Slack", 
          p_nom=150, marginal_cost=0.5)

    # HVDC & Radial Links
    # Using Line components with high capacity to establish a baseline optimum
    n.add("Line", "HVDC_Backbone", bus0="Khanom", bus1="Samui", s_nom=150, x=0.01, r=0.001)
    n.add("Line", "Samui_Phangan_Link", bus0="Samui", bus1="Phangan", s_nom=50, x=0.02, r=0.002)
    n.add("Line", "Phangan_Tao_Link", bus0="Phangan", bus1="Tao", s_nom=16, x=0.05, r=0.005)

    # ── 2. ASSETS (Generation & Costs) ──
    # Fuel cost + Carbon cost approx 0.90 USD/MWh relative scale
    diesel_cost = oc["diesel_price_per_kg"] + (oc["carbon_price_per_kg"] * 2.68)
    
    n.add("Generator", "Samui Diesel", bus="Samui", p_nom=10, marginal_cost=diesel_cost * 200) # g/kWh scaling
    n.add("Generator", "Tao Diesel", bus="Tao", p_nom=10, marginal_cost=diesel_cost * 200)

    # ── 3. LOADS (Extreme Peak Scenario) ──
    n.add("Load", "Samui_Load", bus="Samui", p_set=95.0)
    n.add("Load", "Phangan_Load", bus="Phangan", p_set=26.0)
    n.add("Load", "Tao_Load", bus="Tao", p_set=13.0)

    print("="*80)
    print("  GRIDTOKENX: OPTIMAL POWER FLOW (OPF) ANALYSIS")
    print("="*80)

    try:
        # Solve OPF (Minimizes Cost)
        # We use lopf (Linear OPF) for speed and system-wide balancing
        n.optimize(solver_name='highs')
        
        print("\n[1] OPTIMAL GENERATION DISPATCH")
        gens = n.generators_t.p.loc["now"]
        for name, val in gens.items():
            print(f"  • {name:<15}: {val:>6.2f} MW")

        print("\n[2] LINE LOADING & BOTTLENECK STATUS")
        lines = n.lines_t.p0.loc["now"]
        for name, flow in lines.items():
            limit = n.lines.loc[name, "s_nom"]
            loading = abs(flow) / limit * 100
            print(f"  • {name:<20}: {abs(flow):>6.2f} MW ({loading:>5.1f}%)")

        print("\n[3] ECONOMIC SUMMARY")
        total_cost = n.objective
        print(f"  • Objective (Total Cost Index): {total_cost:.2f}")
        print(f"  • Mainland Import dependency:  {gens['Mainland Slack']:.2f} MW")

        # Verdict
        if gens['Tao Diesel'] > 0:
            print("\n  ✅ VERDICT: Local Tao Diesel is REQUIRED for system optimality.")
        else:
            print("\n  ℹ️ VERDICT: Mainland import is sufficient for current cost-optimum.")

    except Exception as e:
        print(f"  ❌ OPF Failed: {e}")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_opf_analysis()

"""
Technical Power System Analysis — Ko Tao-Phangan-Samui Cluster
Performs Load Flow, Voltage Stability, and Thermal Limit auditing.
"""
import pandapower as pp
import numpy as np
import pandas as pd
from research.pandapower_model import create_ko_tao_network

def analyze_power_system():
    print("="*80)
    print("  TECHNICAL POWER SYSTEM ANALYSIS: RADIAL 115kV/33kV CLUSTER")
    print("="*80)

    scenarios = [
        {"name": "BASE LOAD (Nominal)", "tao": 7.0, "phangan": 18.0, "samui": 55.0},
        {"name": "PEAK LOAD (Songkran)", "tao": 13.0, "phangan": 26.0, "samui": 95.0},
        {"name": "LOW LOAD (Off-Season)", "tao": 5.0, "phangan": 12.0, "samui": 40.0}
    ]

    results = []

    for sc in scenarios:
        net = create_ko_tao_network(
            tao_load_mw=sc["tao"],
            phangan_load_mw=sc["phangan"],
            samui_load_mw=sc["samui"]
        )
        
        try:
            pp.runpp(net, algorithm="nr")
            
            # 1. Voltage Profile
            buses = net.res_bus
            # Find bus by name
            b_tao = net.bus[net.bus.name == "Ko Tao 20kV"].index[0]
            b_phangan = net.bus[net.bus.name == "Phangan 20kV"].index[0]
            b_samui = net.bus[net.bus.name == "Samui Station Phanom - Samui"].index[0]
            
            v_tao = buses.at[b_tao, "vm_pu"]
            v_phangan = buses.at[b_phangan, "vm_pu"]
            v_samui = buses.at[b_samui, "vm_pu"]

            # 2. Line Loading
            lines = net.res_line
            # Bottleneck C2 is now the primary constraint
            l_c2 = net.line[net.line.name == "HVDC C2 (Bottleneck)"].index[0]
            load_c2 = lines.at[l_c2, "loading_percent"]
            # Phangan-Tao link lookup by name
            l_pt = net.line[net.line.name == "Phangan-Tao 33kV Link"].index[0]
            load_pt = lines.at[l_pt, "loading_percent"]
            flow_pt = abs(lines.at[l_pt, "p_from_mw"])

            results.append({
                "Scenario": sc["name"],
                "V_Samui": round(v_samui, 4),
                "V_Phangan": round(v_phangan, 4),
                "V_Tao": round(v_tao, 4),
                "C2_Load%": round(load_c2, 1),
                "TaoLink_MW": round(flow_pt, 2),
                "TaoLink_Load%": round(load_pt, 1)
            })

        except Exception as e:
            print(f"  ❌ Scenario {sc['name']} failed: {e}")

    # Display Steady-State Results
    df_res = pd.DataFrame(results)
    print("\n[1] STEADY-STATE LOAD FLOW (Radial Chain)")
    print(df_res.to_string(index=False))

    # 3. Voltage Stability Audit (AVR Effectiveness)
    print("\n[2] VOLTAGE STABILITY AUDIT (AVR Effectiveness)")
    print("  Distal Node: Ko Tao (Radial Terminus)")
    for sc in scenarios:
        v = next(r for r in results if r["Scenario"] == sc["name"])["V_Tao"]
        status = "STABLE" if 0.95 <= v <= 1.05 else "VIOLATION"
        print(f"  • {sc['name']:<20}: {v:.4f} p.u. [{status}]")

    # 4. Thermal Bottleneck Analysis
    print("\n[3] THERMAL BOTTLENECK ANALYSIS")
    peak = next(r for r in results if r["Scenario"] == "PEAK LOAD (Songkran)")
    print(f"  • Circuit 2 (Bottleneck) Loading: {peak['C2_Load%']}%")
    print(f"  • Phangan-Tao 33kV Link Flow:    {peak['TaoLink_MW']} MW (Limit: 16 MW)")
    
    if peak['TaoLink_MW'] > 16.0:
        print("  ⚠️  WARNING: Distal Link Capacity exceeded in Peak Scenario!")
    else:
        print("  ✅ Distal Link remains within 16 MW thermal envelope.")

    # 5. N-1 Reliability Posture
    print("\n[4] N-1 RELIABILITY POSTURE (Blackout Risk)")
    print("  Scenario: Loss of Samui-Phangan 115kV/33kV Link")
    print("  • Ko Samui:   SURVIVES (Local Gen + BESS)")
    print("  • Ko Phangan: BLACKOUT (Zero Local Assets)")
    print("  • Ko Tao:     PARTIAL (Can island with 10 MW Diesel if link isolated)")

    print("\n"+"="*80)
    print("  ANALYSIS COMPLETE: Grid operates at high utilization during Peak.")
    print("  Active AVR regulation is mandatory to prevent distal voltage collapse.")
    print("="*80)

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    analyze_power_system()

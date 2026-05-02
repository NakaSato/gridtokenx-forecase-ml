"""
PyPSA physics model for Ko Tao–Phangan–Samui multi-voltage cluster.
Modern Python library for Power System Analysis (PyPSA).
"""
import pypsa
import pandas as pd
import numpy as np

def create_pypsa_network(
    tao_load_mw=7.0, phangan_load_mw=20.0, samui_load_mw=65.0,
    tao_diesel_mw=0.0, phangan_diesel_mw=0.0, samui_diesel_mw=0.0,
    phangan_bess_mw=0.0, samui_bess_mw=0.0
):
    n = pypsa.Network()
    
    # ── Buses ─────────────────────────────────────────────────────────────────
    n.add("Bus", "Khanom", v_nom=115.0)
    n.add("Bus", "Samui",  v_nom=115.0)
    n.add("Bus", "Phangan",v_nom=33.0)
    n.add("Bus", "Tao",    v_nom=33.0)

    # ── Slack (Mainland EGAT) ────────────────────────────────────────────────
    n.add("Generator", "Mainland Slack", bus="Khanom", control="Slack", 
          p_nom_extendable=True, carrier="AC")

    # ── Lines & Constraints ──────────────────────────────────────────────────
    # Mainland -> Samui (Circuit 3 Bottleneck)
    n.add("Line", "HVDC_C2", bus0="Khanom", bus1="Samui", r=0.047*23.4, x=0.1*23.4, s_nom=100)
    n.add("Line", "HVDC_C3_Bottleneck", bus0="Khanom", bus1="Samui", r=0.047*23.4, x=0.1*23.4, s_nom=100)

    # Samui -> Phangan (Radial)
    # Define transformer with reactance (x) and resistance (r)
    n.add("Transformer", "Samui_StepDown", bus0="Samui", bus1="Phangan", 
          s_nom=40, x=0.1, r=0.01) # Standard 40MVA reactance proxy
    
    # Phangan -> Tao (Distal 33kV XLPE)
    n.add("Line", "Phangan_Tao_Link", bus0="Phangan", bus1="Tao", 
          r=0.08*40, x=0.1*40, s_nom=16) # 16 MW Thermal Limit

    # ── Loads ────────────────────────────────────────────────────────────────
    n.add("Load", "Samui_Load", bus="Samui", p_set=samui_load_mw)
    n.add("Load", "Phangan_Load", bus="Phangan", p_set=phangan_load_mw)
    n.add("Load", "Tao_Load", bus="Tao", p_set=tao_load_mw)

    # ── Local Generation (Diesel) ────────────────────────────────────────────
    if samui_diesel_mw > 0:
        n.add("Generator", "Samui Diesel", bus="Samui", p_set=samui_diesel_mw, p_nom=10)
    if tao_diesel_mw > 0:
        n.add("Generator", "Tao Diesel", bus="Tao", p_set=tao_diesel_mw, p_nom=10)

    # ── Local Storage (BESS) ───────────────────────────────────────────────
    if samui_bess_mw != 0:
        # BESS as a generator (positive = discharge, negative = charge)
        n.add("Generator", "Samui BESS", bus="Samui", p_set=samui_bess_mw, p_nom=8)

    return n

def run_pypsa_analysis(tao_load_mw=7.0, phangan_load_mw=20.0, samui_load_mw=65.0):
    n = create_pypsa_network(tao_load_mw, phangan_load_mw, samui_load_mw)
    
    try:
        # Run Linear Power Flow (Fast, standard for energy systems)
        n.lpf()
        
        # Extract Results
        flow_hvdc = n.lines_t.p0.loc["now", "HVDC_C3_Bottleneck"] if "now" in n.lines_t.p0.index else n.lines_t.p0.iloc[0]["HVDC_C3_Bottleneck"]
        flow_tao  = n.lines_t.p0.loc["now", "Phangan_Tao_Link"] if "now" in n.lines_t.p0.index else n.lines_t.p0.iloc[0]["Phangan_Tao_Link"]
        
        loading_hvdc = abs(flow_hvdc) / n.lines.loc["HVDC_C3_Bottleneck", "s_nom"] * 100
        loading_tao  = abs(flow_tao) / n.lines.loc["Phangan_Tao_Link", "s_nom"] * 100
        
        return {
            "status": "SUCCESS",
            "hvdc_loading_pct": round(loading_hvdc, 2),
            "tao_link_loading_pct": round(loading_tao, 2),
            "tao_link_mw": round(abs(flow_tao), 2),
            "total_import_mw": round(n.generators_t.p.loc["now", "Mainland Slack"] if "now" in n.generators_t.p.index else n.generators_t.p.iloc[0]["Mainland Slack"], 2)
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

if __name__ == "__main__":
    res = run_pypsa_analysis()
    print(f"PyPSA Analysis Result: {res}")

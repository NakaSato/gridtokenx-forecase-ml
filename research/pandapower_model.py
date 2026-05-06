"""
Pandapower physics model for Ko Tao–Phangan–Samui multi-voltage cluster.

Topology (Radial):
  Mainland (Slack) ──► 115 kV C2/C3 + 33 kV Oil/XLPE ──► Samui
  Samui ──► 115 kV + 33 kV ──► Phangan
  Phangan ──► 33 kV XLPE (0-16 MW cap) ──► Ko Tao (w/ AVR)

Cable parameters (IEC 60287 / PEA standards):
  115 kV XLPE: r=0.047 Ω/km, x=0.100 Ω/km, max_i=0.530 kA
  33 kV XLPE:  r=0.125 Ω/km, x=0.120 Ω/km, max_i=0.350 kA
"""
import pandapower as pp
import os

def _add_cable_types(net):
    # 115 kV XLPE Submarine
    pp.create_std_type(net, {
        "r_ohm_per_km": 0.0470, "x_ohm_per_km": 0.1000, "c_nf_per_km": 150.0,
        "max_i_ka": 0.530, "type": "cs"
    }, name="115kV_XLPE", element="line")
    
    # 33 kV XLPE Submarine
    pp.create_std_type(net, {
        "r_ohm_per_km": 0.0800, "x_ohm_per_km": 0.1000, "c_nf_per_km": 200.0,
        "max_i_ka": 0.450, "type": "cs"
    }, name="33kV_XLPE", element="line")

    # 33 kV Oil-Filled Submarine (Older, higher impedance)
    pp.create_std_type(net, {
        "r_ohm_per_km": 0.1250, "x_ohm_per_km": 0.1200, "c_nf_per_km": 180.0,
        "max_i_ka": 0.350, "type": "cs"
    }, name="33kV_Oil", element="line")

def create_ko_tao_network(
    tao_load_mw=7.0, phangan_load_mw=20.0, samui_load_mw=65.0,
    tao_diesel_mw=0.0, phangan_diesel_mw=0.0, samui_diesel_mw=0.0,
    phangan_bess_mw=0.0, samui_bess_mw=0.0
):
    net = pp.create_empty_network()
    _add_cable_types(net)

    # ── Buses ─────────────────────────────────────────────────────────────────
    b_khanom_115 = pp.create_bus(net, vn_kv=110, name="Khanom 110kV", geodata=(99.860229, 9.234960))
    b_khanom_33  = pp.create_bus(net, vn_kv=20,  name="Khanom 20kV",  geodata=(99.860229, 9.244960))
    b_samui_115  = pp.create_bus(net, vn_kv=110, name="Samui Station Phanom - Samui", geodata=(99.946747, 9.431112))
    b_samui_33   = pp.create_bus(net, vn_kv=20,  name="Samui 20kV",  geodata=(99.946747, 9.441112))
    b_phangan_115= pp.create_bus(net, vn_kv=110, name="Phangan 110kV", geodata=(99.994854, 9.706584))
    b_phangan_33 = pp.create_bus(net, vn_kv=20,  name="Phangan 20kV",  geodata=(99.994854, 9.716584))
    b_tao_33     = pp.create_bus(net, vn_kv=20,  name="Ko Tao 20kV",   geodata=(99.826585, 10.078363))

    pp.create_ext_grid(net, bus=b_khanom_115, vm_pu=1.02, name="Mainland Slack")

    # ── Transformers ──────────────────────────────────────────────────────────
    # Khanom: 115/33kV
    pp.create_transformer(net, b_khanom_115, b_khanom_33, std_type="25 MVA 110/20 kV")
    # Samui: 115/33kV
    pp.create_transformer(net, b_samui_115, b_samui_33, std_type="40 MVA 110/20 kV", tap_pos=-4)
    # Phangan: 115/33kV
    pp.create_transformer(net, b_phangan_115, b_phangan_33, std_type="25 MVA 110/20 kV", tap_pos=-4)

    # ── Lines (Topology-Aware) ────────────────────────────────────────────────
    # Mainland -> Samui (115kV + 33kV)
    pp.create_line(net, b_khanom_115, b_samui_115, 23.25, "115kV_XLPE", name="HVDC C3")
    pp.create_line(net, b_khanom_115, b_samui_115, 23.25, "115kV_XLPE", name="HVDC C2 (Bottleneck)")
    pp.create_line(net, b_khanom_33,  b_samui_33,  23.25, "33kV_Oil",    name="Mainland-Samui 33kV Oil")
    pp.create_line(net, b_khanom_33,  b_samui_33,  23.25, "33kV_XLPE",   name="Mainland-Samui 33kV XLPE")
    
    # Samui -> Phangan
    pp.create_line(net, b_samui_115, b_phangan_115, 30.0, "115kV_XLPE", name="Samui-Phangan 115kV")
    pp.create_line(net, b_samui_33, b_phangan_33, 30.0, "33kV_XLPE", name="Samui-Phangan 33kV")

    # Phangan -> Ko Tao (Distal 33kV XLPE)
    # The 'Excess Power 0-16 MW' link
    pp.create_line(net, b_phangan_33, b_tao_33, 45.25, "33kV_XLPE", name="Phangan-Tao 33kV Link")

    # ── AVRs (Voltage Regulation) ────────────────────────────────────────────
    # Modeled as generators (PV bus) to strictly enforce voltage setpoints
    pp.create_gen(net, b_phangan_33, p_mw=0.0, vm_pu=1.0, name="Ko Phangan AVR", min_q_mvar=-10.0, max_q_mvar=10.0)
    pp.create_gen(net, b_tao_33,     p_mw=0.0, vm_pu=1.0, name="Ko Tao AVR",     min_q_mvar=-5.0, max_q_mvar=5.0)

    # ── Loads & Generation ────────────────────────────────────────────────────
    pp.create_load(net, b_samui_115, samui_load_mw, samui_load_mw*0.329, name="Samui Load")
    pp.create_load(net, b_phangan_115, phangan_load_mw, phangan_load_mw*0.329, name="Phangan Load")
    pp.create_load(net, b_tao_33, tao_load_mw, tao_load_mw*0.329, name="Ko Tao Load")

    if tao_diesel_mw > 0:
        pp.create_sgen(net, b_tao_33, p_mw=tao_diesel_mw, name="Tao Diesel")
    if phangan_diesel_mw > 0:
        pp.create_sgen(net, b_phangan_115, p_mw=phangan_diesel_mw, name="Phangan Diesel")
    if samui_diesel_mw > 0:
        pp.create_sgen(net, b_samui_115, p_mw=samui_diesel_mw, name="Samui Diesel")
    
    if phangan_bess_mw != 0:
        pp.create_sgen(net, b_phangan_115, p_mw=phangan_bess_mw, name="Phangan BESS")
    if samui_bess_mw != 0:
        pp.create_sgen(net, b_samui_115, p_mw=samui_bess_mw, name="Samui BESS")

    return net

def verify_dispatch_stability(
    tao_load_mw=7.0, phangan_load_mw=20.0, samui_load_mw=65.0,
    tao_diesel_mw=0.0, phangan_diesel_mw=0.0, samui_diesel_mw=0.0,
    phangan_bess_mw=0.0, samui_bess_mw=0.0
):
    net = create_ko_tao_network(tao_load_mw, phangan_load_mw, samui_load_mw,
                                tao_diesel_mw, phangan_diesel_mw, samui_diesel_mw,
                                phangan_bess_mw, samui_bess_mw)
    try:
        pp.runpp(net, algorithm="nr")
        
        # Helper to get bus voltage by name
        def get_v(name):
            idx = net.bus[net.bus.name == name].index[0]
            return net.res_bus.at[idx, "vm_pu"]

        results = {
            "stable": True,
            "v_khanom_pu": round(get_v("Khanom 110kV"), 4),
            "v_samui_pu": round(get_v("Samui Station Phanom - Samui"), 4),
            "v_phangan_pu": round(get_v("Phangan 110kV"), 4),
            "v_tao_pu": round(get_v("Ko Tao 20kV"), 4),
            "bottleneck_loading_pct": round(net.res_line.at[1, "loading_percent"], 2),
            "max_line_loading": round(net.res_line["loading_percent"].max(), 2),
            "voltage_ok": 0.95 <= get_v("Ko Tao 20kV") <= 1.05,
            "lines": []
        }
        
        for idx, row in net.line.iterrows():
            results["lines"].append({
                "name": row["name"],
                "loading_percent": net.res_line.at[idx, "loading_percent"],
                "p_from_mw": net.res_line.at[idx, "p_from_mw"]
            })
            
        return results
    except Exception as e:
        return {"stable": False, "error": str(e)}


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")

    def _print(label, r):
        if "error" in r:
            print(f"  ❌ ERROR: {r['error']}"); return
        status = ("✅ FEASIBLE" if r["stable"] and r["voltage_ok"] else
                  "⚠️  LINE OVERLOAD" if r["stable"] and r["max_line_loading"] > 100 else "❌ VOLTAGE VIOLATION")
        print(f"  {status}")
        print(f"  Voltages (p.u.):")
        print(f"    Khanom={r['v_khanom_pu']}  Samui={r['v_samui_pu']}"
              f"  Phangan={r['v_phangan_pu']}  Tao={r['v_tao_pu']}")
        print(f"  Bottleneck (HVDC Connector): {r['bottleneck_loading_pct']}%")
        for ln in r["lines"]:
            flag = " ⚡" if ln["loading_percent"] > 100 else ""
            print(f"    {ln['name']:<35} {ln['loading_percent']:>6.1f}%  {ln['p_from_mw']:>7.2f} MW{flag}")

    print("=== Normal operation (Samui 65 MW) ===")
    _print("normal", verify_dispatch_stability())

    print("\n=== Samui peak (95 MW) + Samui diesel 10 MW ===")
    _print("peak", verify_dispatch_stability(
        samui_load_mw=95.0, tao_load_mw=7.5, phangan_load_mw=25.0,
        tao_diesel_mw=7.5, samui_diesel_mw=10.0,
    ))

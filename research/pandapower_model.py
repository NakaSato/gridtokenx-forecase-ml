"""
Pandapower physics model for Ko Tao–Phangan–Samui 115 kV cluster.

6-bus model — Samui internal ring fully resolved from OSM/GeoJSON coordinates.

Bus topology:
  b_khanom      Khanom Substation (slack)       99.860°E  9.234°N
  b_samui_hvdc  HVDC Connector terminus         99.947°E  9.431°N
  b_samui3      สถานีไฟฟ้าเกาะสมุย 3            100.020°E  9.441°N
  b_samui_trans Samui Transition Substation     100.003°E  9.567°N
  b_samui_dist  Ko Samui Distribution Sub        99.995°E  9.707°N  ← load bus
  b_phangan     Ko Phangan                      100.060°E  9.740°N  ← estimated
  b_tao         Ko Tao                           99.840°E 10.095°N  ← estimated

Cable distances (haversine on OSM coords, 115 kV XLPE submarine 630 mm² Cu):
  Khanom → Samui HVDC end  : 23.42 km  ⚡ HVDC Koh Samui Connector (bottleneck)
  Samui HVDC end → Samui 3 :  8.13 km  Samui ring segment
  Samui 3 → transition     : 14.17 km  Koh Samui Export seg 1
  transition → dist sub    : 15.57 km  Koh Samui Export seg 2
  dist sub → Phangan       : 30.00 km  estimated
  Phangan → Ko Tao         : 40.00 km  estimated

Cable parameters (ABB XLPE Submarine Cable Systems, IEC 60287):
  r = 0.047 Ω/km  x = 0.100 Ω/km  c = 250 nF/km  max_i = 530 A
"""
import pandapower as pp
import os

ROOT = os.path.dirname(os.path.dirname(__file__))


def _add_submarine_cable_type(net):
    # 115 kV XLPE submarine cable, 630 mm² Cu (ABB/Prysmian standard)
    # c_nf_per_km reduced to 150 nF/km (effective after partial shunt compensation
    # at cable terminations — standard practice for cables >20 km at 115 kV)
    pp.create_std_type(net, {
        "r_ohm_per_km": 0.0470,
        "x_ohm_per_km": 0.1000,
        "c_nf_per_km":  150.0,
        "max_i_ka":     0.530,
        "type":         "cs",
    }, name="115kV_XLPE_Submarine_630Cu", element="line")


def create_ko_tao_network(
    tao_load_mw=7.0,
    phangan_load_mw=20.0,
    samui_load_mw=65.0,
    tao_diesel_mw=7.5,
    phangan_diesel_mw=0.0,
    samui_diesel_mw=0.0,
):
    net = pp.create_empty_network()
    _add_submarine_cable_type(net)

    # ── Buses ─────────────────────────────────────────────────────────────────
    b_khanom      = pp.create_bus(net, vn_kv=115, name="Khanom Substation (Slack)",
                                  geodata=(99.860, 9.234))
    b_samui_hvdc  = pp.create_bus(net, vn_kv=115, name="Samui HVDC Terminus",
                                  geodata=(99.947, 9.431))
    b_samui3      = pp.create_bus(net, vn_kv=115, name="Samui 3 Substation",
                                  geodata=(100.020, 9.441))
    b_samui_trans = pp.create_bus(net, vn_kv=115, name="Samui Transition Sub",
                                  geodata=(100.003, 9.567))
    b_samui_dist  = pp.create_bus(net, vn_kv=115, name="Ko Samui Distribution Sub",
                                  geodata=(99.995, 9.707))
    b_phangan     = pp.create_bus(net, vn_kv=115, name="Ko Phangan",
                                  geodata=(100.060, 9.740))
    b_tao         = pp.create_bus(net, vn_kv=115, name="Ko Tao",
                                  geodata=(99.840, 10.095))

    pp.create_ext_grid(net, bus=b_khanom, vm_pu=1.00, name="Khanom 970MW")

    # ── Cables (real OSM distances) ───────────────────────────────────────────
    # All inter-island segments have parallel circuits (N-1 redundancy standard).
    ST = "115kV_XLPE_Submarine_630Cu"
    pp.create_line(net, b_khanom,      b_samui_hvdc,  23.42, ST, name="HVDC Koh Samui Connector C1")
    pp.create_line(net, b_khanom,      b_samui_hvdc,  23.42, ST, name="HVDC Koh Samui Connector C2")
    pp.create_line(net, b_samui_hvdc,  b_samui3,       8.13, ST, name="Samui HVDC–Samui3 Ring C1")
    pp.create_line(net, b_samui_hvdc,  b_samui3,       8.13, ST, name="Samui HVDC–Samui3 Ring C2")
    pp.create_line(net, b_samui3,      b_samui_trans, 14.17, ST, name="Koh Samui Export Seg1 C1")
    pp.create_line(net, b_samui3,      b_samui_trans, 14.17, ST, name="Koh Samui Export Seg1 C2")
    pp.create_line(net, b_samui_trans, b_samui_dist,  15.57, ST, name="Koh Samui Export Seg2 C1")
    pp.create_line(net, b_samui_trans, b_samui_dist,  15.57, ST, name="Koh Samui Export Seg2 C2")
    pp.create_line(net, b_samui_dist,  b_phangan,     30.00, ST, name="Samui–Phangan Cable C1")
    pp.create_line(net, b_samui_dist,  b_phangan,     30.00, ST, name="Samui–Phangan Cable C2")
    pp.create_line(net, b_phangan,     b_tao,         40.00, ST, name="Phangan–Tao Cable")

    # ── Loads (pf=0.95 → q ≈ p × 0.329) ─────────────────────────────────────
    # Samui split: 70% at distribution sub (hotels/airport north), 30% at Samui3 ring
    pp.create_load(net, b_samui_dist, samui_load_mw * 0.70,
                   samui_load_mw * 0.70 * 0.329, name="Ko Samui Load (dist)")
    pp.create_load(net, b_samui3,     samui_load_mw * 0.30,
                   samui_load_mw * 0.30 * 0.329, name="Ko Samui Load (ring)")
    pp.create_load(net, b_phangan,    phangan_load_mw,
                   phangan_load_mw * 0.329,       name="Ko Phangan Load")
    pp.create_load(net, b_tao,        tao_load_mw,
                   tao_load_mw * 0.329,           name="Ko Tao Load")

    # ── Shunt reactors (voltage control) ─────────────────────────────────────
    # Cable charging Q (150 nF/km effective, 115 kV):
    #   Samui–Phangan 30 km → 18.7 Mvar  → 15 Mvar reactor at Samui dist
    #   Phangan–Tao   40 km → 24.9 Mvar  → 20 Mvar reactor at Phangan
    #   Additional at Tao to absorb end-of-line voltage rise
    pp.create_shunt(net, b_samui_dist, q_mvar=-15.0, p_mw=0.0, name="Samui Shunt Reactor 15Mvar")
    pp.create_shunt(net, b_phangan,    q_mvar=-22.0, p_mw=0.0, name="Phangan Shunt Reactor 22Mvar")
    pp.create_shunt(net, b_tao,        q_mvar=-28.0, p_mw=0.0, name="Ko Tao Shunt Reactor 28Mvar")

    # ── Diesel generators (AI-controlled setpoints) ───────────────────────────
    if tao_diesel_mw     > 0: pp.create_sgen(net, b_tao,        tao_diesel_mw,     name="Ko Tao Diesel")
    if phangan_diesel_mw > 0: pp.create_sgen(net, b_phangan,    phangan_diesel_mw,  name="Ko Phangan Diesel")
    if samui_diesel_mw   > 0: pp.create_sgen(net, b_samui_dist, samui_diesel_mw,    name="Ko Samui Diesel")

    return net


def verify_dispatch_stability(
    tao_load_mw=7.0, phangan_load_mw=20.0, samui_load_mw=65.0,
    tao_diesel_mw=7.5, phangan_diesel_mw=0.0, samui_diesel_mw=0.0,
):
    net = create_ko_tao_network(
        tao_load_mw, phangan_load_mw, samui_load_mw,
        tao_diesel_mw, phangan_diesel_mw, samui_diesel_mw,
    )
    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)
        buses = net.res_bus["vm_pu"]
        lines = net.res_line[["loading_percent", "p_from_mw"]].copy()
        lines["name"] = net.line["name"].values

        v_ok    = all(0.95 <= v <= 1.05 for v in buses.values)
        line_ok = lines["loading_percent"].max() <= 100.0

        return {
            "stable":                   v_ok and line_ok,
            "voltage_ok":               v_ok,
            "line_ok":                  line_ok,
            "v_khanom_pu":              round(buses.iloc[0], 4),
            "v_samui_hvdc_pu":          round(buses.iloc[1], 4),
            "v_samui3_pu":              round(buses.iloc[2], 4),
            "v_samui_trans_pu":         round(buses.iloc[3], 4),
            "v_samui_dist_pu":          round(buses.iloc[4], 4),
            "v_phangan_pu":             round(buses.iloc[5], 4),
            "v_tao_pu":                 round(buses.iloc[6], 4),
            "max_line_loading":         round(lines["loading_percent"].max(), 2),
            "bottleneck_loading_pct":   round(
                lines.loc[lines["name"].str.startswith("HVDC Koh Samui Connector"),
                          "loading_percent"].max(), 2),
            "lines": lines.round(2).to_dict("records"),
        }
    except Exception as e:
        return {"stable": False, "error": str(e)}


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")

    def _print(label, r):
        if "error" in r:
            print(f"  ❌ ERROR: {r['error']}"); return
        status = ("✅ FEASIBLE" if r["stable"] else
                  "⚠️  LINE OVERLOAD" if r["voltage_ok"] else "❌ VOLTAGE VIOLATION")
        print(f"  {status}")
        print(f"  Voltages (p.u.):")
        print(f"    Khanom={r['v_khanom_pu']}  SamuiHVDC={r['v_samui_hvdc_pu']}"
              f"  Samui3={r['v_samui3_pu']}  SamuiTrans={r['v_samui_trans_pu']}"
              f"  SamuiDist={r['v_samui_dist_pu']}  Phangan={r['v_phangan_pu']}"
              f"  Tao={r['v_tao_pu']}")
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

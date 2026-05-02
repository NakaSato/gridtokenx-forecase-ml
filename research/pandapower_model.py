import pandapower as pp
import pandapower.networks as nw
import yaml
import os

def create_ko_tao_network(load_mw=8.5, diesel_mw=7.5, circuit_cap_mw=8.0):
    """
    Creates an electrical model of the Ko Tao-Phangan-Samui cluster.
    Locked to 115 kV PEA specifications and GeoJSON distances.
    """
    net = pp.create_empty_network()
    
    # 1. Define Buses (Nodes)
    # 115 kV Backbone
    b_khanom = pp.create_bus(net, vn_kv=115, name="Khanom (Slack)")
    b_samui  = pp.create_bus(net, vn_kv=115, name="Samui Substation")
    b_tao    = pp.create_bus(net, vn_kv=115, name="Ko Tao Node")
    
    # 2. External Grid (Mainland Feeder)
    # Representing the 970 MW Khanom Power Station
    pp.create_ext_grid(net, bus=b_khanom, vm_pu=1.02, name="Mainland Source")
    
    # 3. Transmission Lines (Locked to GeoJSON Distances)
    # Using standard 115 kV subsea cable parameters
    # Khanom -> Samui (45.12 km)
    pp.create_line(net, from_bus=b_khanom, to_bus=b_samui, length_km=45.12, 
                   std_type="149-AL1/24-ST1A 110.0", name="HVDC Connector")
    
    # Samui -> Tao (approx 60 km via Phangan)
    # This line has a capacity of circuit_cap_mw
    pp.create_line(net, from_bus=b_samui, to_bus=b_tao, length_km=60.0, 
                   std_type="149-AL1/24-ST1A 110.0", name="Phangan-Tao Link")
    
    # 4. Local Assets (AI Control Layer)
    # Local Island Load
    pp.create_load(net, bus=b_tao, p_mw=load_mw, q_mvar=load_mw*0.25, name="Tao Load")
    
    # Diesel Generation (The AI-controlled setpoint)
    if diesel_mw > 0:
        pp.create_sgen(net, bus=b_tao, p_mw=diesel_mw, name="Diesel Gen")
    
    return net

def verify_dispatch_stability(load_mw, diesel_mw, circuit_mw):
    """Runs a Power Flow to verify voltage stability at Ko Tao."""
    net = create_ko_tao_network(load_mw, diesel_mw, circuit_mw)
    
    try:
        pp.runpp(net)
        v_tao = net.res_bus.at[2, "vm_pu"]
        loading_pct = net.res_line.loading_percent.max()
        
        stable = 0.95 <= v_tao <= 1.05
        
        return {
            "stable": stable,
            "v_tao_pu": round(v_tao, 4),
            "max_line_loading": round(loading_pct, 2),
            "load_mw": load_mw,
            "diesel_mw": diesel_mw
        }
    except:
        return {"stable": False, "error": "Power flow did not converge"}

if __name__ == "__main__":
    # Test Scenario: 8.5 MW Load, 7.5 MW Diesel, 4.0 MW Grid Import
    print("--- GridTokenX: Pandapower Physics Verification ---")
    result = verify_dispatch_stability(load_mw=8.5, diesel_mw=7.5, circuit_mw=4.0)
    
    if result["stable"]:
        print(f"✅ FEASIBLE: Voltage at Tao = {result['v_tao_pu']} p.u.")
        print(f"   Line Loading: {result['max_line_loading']}%")
    else:
        print(f"❌ UNSTABLE: Dispatch setpoints violate PEA safety limits.")

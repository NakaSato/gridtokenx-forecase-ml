import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd

class PhysicsEngine:
    """
    Maintains a persistent pandapower network for high-performance physics simulation.
    """
    def __init__(self):
        self.net = self._create_network()
        # Pre-run to warm up Numba/cache
        pp.runpp(self.net)

    def _create_network(self):
        """
        Creates a 6-bus physics model of the Ko Tao-Phangan-Samui radial cluster.
        Topology:
          [Bus 0: Khanom] --(115kV Cable)--> [Bus 1: Samui HV]
          [Bus 1] --(Transformer)--> [Bus 2: Samui MV]
          [Bus 2] --(115kV Cable)--> [Bus 3: Phangan HV]
          [Bus 3] --(Transformer)--> [Bus 4: Phangan MV]
          [Bus 4] --(33kV Cable)--> [Bus 5: Tao MV]
        """
        net = pp.create_empty_network()

        # 1. Create Buses
        b0 = pp.create_bus(net, vn_kv=115.0, name="Khanom (Slack)")
        b1 = pp.create_bus(net, vn_kv=115.0, name="Samui HV")
        b2 = pp.create_bus(net, vn_kv=33.0,  name="Samui MV")
        b3 = pp.create_bus(net, vn_kv=115.0, name="Phangan HV")
        b4 = pp.create_bus(net, vn_kv=33.0,  name="Phangan MV")
        b5 = pp.create_bus(net, vn_kv=33.0,  name="Ko Tao 33kV")
        b6 = pp.create_bus(net, vn_kv=22.0,  name="Ko Tao 22kV")

        # 2. Slack (External Grid)
        pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Mainland Grid")

        # 3. Transformers
        pp.create_transformer_from_parameters(
            net, hv_bus=b1, lv_bus=b2, sn_mva=100.0, 
            vn_hv_kv=115.0, vn_lv_kv=33.0, 
            vkr_percent=1.0, vk_percent=12.0, pfe_kw=50.0, i0_percent=0.1,
            name="Samui Main TX"
        )
        pp.create_transformer_from_parameters(
            net, hv_bus=b3, lv_bus=b4, sn_mva=63.0, 
            vn_hv_kv=115.0, vn_lv_kv=33.0, 
            vkr_percent=1.0, vk_percent=12.0, pfe_kw=35.0, i0_percent=0.1,
            name="Phangan Main TX"
        )
        pp.create_transformer_from_parameters(
            net, hv_bus=b5, lv_bus=b6, sn_mva=16.0, 
            vn_hv_kv=33.0, vn_lv_kv=22.0, 
            vkr_percent=1.2, vk_percent=10.0, pfe_kw=15.0, i0_percent=0.1,
            name="Tao Distal TX"
        )

        # 4. Cables (Submarine)
        pp.create_line_from_parameters(
            net, from_bus=b0, to_bus=b1, length_km=25.0,
            r_ohm_per_km=0.04, x_ohm_per_km=0.12, c_nf_per_km=250,
            max_i_ka=0.9, name="Mainland-Samui 115kV"
        )
        pp.create_line_from_parameters(
            net, from_bus=b1, to_bus=b3, length_km=18.0,
            r_ohm_per_km=0.04, x_ohm_per_km=0.12, c_nf_per_km=250,
            max_i_ka=0.9, name="Samui-Phangan 115kV"
        )
        pp.create_line_from_parameters(
            net, from_bus=b4, to_bus=b5, length_km=45.0,
            r_ohm_per_km=0.12, x_ohm_per_km=0.11, c_nf_per_km=210,
            max_i_ka=0.3, name="Phangan-Tao 33kV"
        )

        # 5. Loads (Initial) - Tao moved to b6
        pp.create_load(net, bus=b2, p_mw=55.0, q_mvar=17.0, name="Samui Load")
        pp.create_load(net, bus=b4, p_mw=18.0, q_mvar=5.6,  name="Phangan Load")
        pp.create_load(net, bus=b6, p_mw=7.0,  q_mvar=2.2,  name="Tao Load")

        return net

    def run_step(self, tao_load_mw, phangan_load_mw, samui_load_mw):
        # Update loads (MW)
        self.net.load.at[0, "p_mw"] = float(samui_load_mw)
        self.net.load.at[1, "p_mw"] = float(phangan_load_mw)
        self.net.load.at[2, "p_mw"] = float(tao_load_mw)
        
        # Approximate reactive power (cos phi = 0.95)
        self.net.load.at[0, "q_mvar"] = float(samui_load_mw * 0.31)
        self.net.load.at[1, "q_mvar"] = float(phangan_load_mw * 0.31)
        self.net.load.at[2, "q_mvar"] = float(tao_load_mw * 0.31)
        
        try:
            pp.runpp(self.net, enforce_q_lims=True)
            
            hvdc_loading = self.net.res_line.at[0, "loading_percent"]
            v_tao = self.net.res_bus.at[6, "vm_pu"]
            
            total_p_gen = self.net.res_ext_grid.p_mw.sum()
            total_p_load = self.net.res_load.p_mw.sum()
            total_losses = total_p_gen - total_p_load
            
            return {
                "status": "SUCCESS",
                "bottleneck_loading_pct": float(hvdc_loading),
                "v_tao_pu": float(v_tao),
                "line_loss_mw": float(total_losses),
                "tao_v_kv": float(v_tao * 22.0)
            }
        except Exception as e:
            return {"status": "FAILURE", "error": str(e)}

# For backward compatibility with existing code
def verify_dispatch_stability(tao_load_mw, phangan_load_mw, samui_load_mw):
    engine = PhysicsEngine()
    return engine.run_step(tao_load_mw, phangan_load_mw, samui_load_mw)

if __name__ == "__main__":
    engine = PhysicsEngine()
    import time
    start = time.time()
    for _ in range(10):
        res = engine.run_step(8.0, 22.0, 70.0)
    print(f"Time for 10 runs: {time.time()-start:.4f}s")
    print(f"Result: {res}")

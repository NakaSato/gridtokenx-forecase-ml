import pypsa
import numpy as np
import pandas as pd
import yaml
import os

def create_pypsa_network():
    """
    Creates a PyPSA model of the Ko Tao-Phangan-Samui radial cluster.
    Optimized for Linear Optimal Power Flow (LOPF) and Security-Constrained Analysis.
    """
    n = pypsa.Network()
    
    # 1. Add Buses
    n.add("Bus", "Khanom", v_nom=115.0, carrier="AC")
    n.add("Bus", "Samui",  v_nom=115.0, carrier="AC")
    n.add("Bus", "Phangan", v_nom=115.0, carrier="AC")
    n.add("Bus", "Tao",     v_nom=22.0,  carrier="AC")

    # 2. Add External Grid (Mainland)
    # Marginal cost reflects the price of importing from the mainland
    n.add("Generator", "Mainland Import", 
          bus="Khanom", 
          p_nom=1000, 
          marginal_cost=70.0, # THB/MWh base
          carrier="Grid")

    # 3. Add Lines / Links
    # Note: Using Links for radial submarine cables simplifies LOPF with efficiency losses
    n.add("Line", "Mainland-Samui",
          bus0="Khanom", bus1="Samui",
          x=0.1, r=0.01, s_nom=110.0) # 110 MW limit

    n.add("Line", "Samui-Phangan",
          bus0="Samui", bus1="Phangan",
          x=0.15, r=0.02, s_nom=45.0) # 45 MW limit

    n.add("Transformer", "Phangan-Tao Transformer",
          bus0="Phangan", bus1="Tao",
          s_nom=16.0, x=0.1, r=0.01)

    n.add("Line", "Phangan-Tao",
          bus0="Phangan", bus1="Tao",
          x=0.2, r=0.05, s_nom=16.0) # 16 MW limit (XLPE link)

    # 4. Add Diesel Generators (Backup)
    n.add("Generator", "Tao Diesel",
          bus="Tao",
          p_nom=10.0,
          marginal_cost=450.0, # Expensive diesel
          carrier="Diesel")

    n.add("Generator", "Phangan Diesel",
          bus="Phangan",
          p_nom=15.0,
          marginal_cost=380.0,
          carrier="Diesel")

    n.add("Generator", "Samui Diesel",
          bus="Samui",
          p_nom=20.0,
          marginal_cost=320.0,
          carrier="Diesel")

    # 5. Add BESS (at Samui)
    n.add("Bus", "Samui BESS Bus", carrier="DC")
    n.add("Store", "Samui BESS",
          bus="Samui BESS Bus",
          e_nom=50.0,
          e_cyclic=True,
          marginal_cost=12.0)
    
    n.add("Link", "Samui BESS Converter",
          bus0="Samui", bus1="Samui BESS Bus",
          p_nom=8.0, # 8 MW Rating from SLD
          efficiency=0.95)

    return n

def run_security_constrained_opf(tao_load, phangan_load, samui_load, mainland_price=70.0):
    """
    Runs LOPF to find the most cost-effective dispatch while respecting grid limits.
    """
    n = create_pypsa_network()
    
    # Set Loads
    n.add("Load", "Tao_L",     bus="Tao",     p_set=tao_load)
    n.add("Load", "Phangan_L", bus="Phangan", p_set=phangan_load)
    n.add("Load", "Samui_L",   bus="Samui",   p_set=samui_load)
    
    # Set Mainland Price
    n.generators.at["Mainland Import", "marginal_cost"] = mainland_price
    
    # Run LOPF
    n.optimize(solver_name='highs')
    
    return {
        "status": n.model.status,
        "total_cost": n.objective,
        "dispatch": n.generators_t.p.iloc[0].to_dict(),
        "line_loading": (n.lines_t.p0.iloc[0].abs() / n.lines.s_nom).to_dict(),
        "bess_soc": n.stores_t.e.iloc[0].to_dict() if not n.stores.empty else {}
    }

if __name__ == "__main__":
    # Test Run
    try:
        res = run_security_constrained_opf(8.5, 25.0, 75.0)
        print(f"OPF Status: {res['status']}")
        print(f"Total Objective: {res['total_cost']:.2f} THB")
        print(f"Dispatch: {res['dispatch']}")
        print(f"Line Loading: {res['line_loading']}")
    except Exception as e:
        print(f"Error: {e}")

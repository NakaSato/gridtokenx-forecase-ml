import numpy as np
import yaml
import pandas as pd

class IslandAgent:
    def __init__(self, name, load, gen, max_diesel, diesel_cost):
        self.name = name
        self.load = load
        self.gen = gen
        self.max_diesel = max_diesel
        self.diesel_cost = diesel_cost # THB per MW
        
        # Internal state for ADMM
        self.diesel_output = 0.0
        self.u = 0.0 # Dual variable

    def update_local(self, x_avg, u, rho):
        """
        Local step for Exchange Problem:
        Minimize C_i * d_i + (rho/2) * || d_i - d_i_old + x_avg + u ||^2
        """
        # Analytical solution: d_i = d_i_old - x_avg - u - C_i/rho
        v = self.diesel_output - x_avg - u
        target = v - (self.diesel_cost / rho)
        self.diesel_output = np.clip(target, 0, self.max_diesel)
        return self.diesel_output

def run_cluster_admm(agents, target_diesel_total, max_iter=500, rho=50.0, tolerance=0.01):
    """
    ADMM for Cluster-wide Diesel Coordination (Exchange Problem).
    Minimize sum(C_i * d_i) s.t. sum(d_i) = target_diesel_total.
    """
    n = len(agents)
    history = []
    
    # Global dual variable for the sum constraint
    u = 0.0 
    
    # Initialize diesel outputs (fair share)
    for a in agents:
        a.diesel_output = target_diesel_total / n
    
    for i in range(max_iter):
        # 1. Calculate average mismatch (x_avg)
        # We want sum(d_i) = target -> sum(d_i - target/n) = 0
        d_vals = np.array([a.diesel_output for a in agents])
        x_avg = np.mean(d_vals) - (target_diesel_total / n)
        
        # 2. Local Update
        for a in agents:
            a.update_local(x_avg, u, rho)
            
        # 3. Dual Update
        # Update u with the new average mismatch
        new_d_vals = np.array([a.diesel_output for a in agents])
        new_x_avg = np.mean(new_d_vals) - (target_diesel_total / n)
        u = u + new_x_avg
            
        # Check convergence
        current_sum = np.sum(new_d_vals)
        residual = np.abs(current_sum - target_diesel_total)
        history.append(residual)
        
        if residual < tolerance and i > 20:
            break
            
    return history, [a.diesel_output for a in agents]

def get_cluster_dispatch(samui_load: float, phangan_load: float, tao_load: float):
    """Entry point for API/Dashboard coordination."""
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    cc = cfg["cluster"]
    ca = cc["assets"]
    n1_limit = cc["admm"]["mainland_n1_limit_mw"]
    
    net_load = samui_load + phangan_load + tao_load
    deficit = max(0, net_load - n1_limit)
    
    # Real-world costs and capacities from coordinated config
    agents = [
        IslandAgent("Ko Tao",      load=tao_load,    gen=ca["ko_tao"]["renewable_mw"],    max_diesel=ca["ko_tao"]["diesel_mw"],    diesel_cost=ca["ko_tao"]["diesel_cost"]),
        IslandAgent("Ko Phangan",  load=phangan_load, gen=ca["ko_phangan"]["renewable_mw"], max_diesel=ca["ko_phangan"]["diesel_mw"], diesel_cost=ca["ko_phangan"]["diesel_cost"]),
        IslandAgent("Ko Samui",    load=samui_load,   gen=ca["ko_samui"]["renewable_mw"],   max_diesel=ca["ko_samui"]["diesel_mw"],   diesel_cost=ca["ko_samui"]["diesel_cost"])
    ]

    if deficit <= 0:
        return {
            "deficit_mw": 0.0,
            "total_dispatch_mw": 0.0,
            "n1_limit_mw": n1_limit,
            "agents": [{"name": a.name, "diesel_output_mw": 0.0} for a in agents]
        }

    history, results = run_cluster_admm(agents, deficit, **cc["admm"])
    
    return {
        "deficit_mw": round(deficit, 3),
        "total_dispatch_mw": round(sum(results), 3),
        "n1_limit_mw": n1_limit,
        "agents": [
            {"name": a.name, "diesel_output_mw": round(res, 3), "cost_per_mw": a.diesel_cost}
            for a, res in zip(agents, results)
        ],
        "iterations": len(history)
    }

def simulate_cluster_dispatch(samui_load, phangan_load, tao_load):
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          GridTokenX — Multi-Island ADMM Diesel Coordinator           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cc = cfg["cluster"]
    ca = cc["assets"]
    n1_limit = cc["admm"]["mainland_n1_limit_mw"]

    # Current Grid State
    net_load = samui_load + phangan_load + tao_load
    deficit = max(0, net_load - n1_limit)
    
    print(f"Cluster Total Load: {net_load:.2f} MW")
    print(f"N-1 Mainland Limit: {n1_limit:.2f} MW")
    print(f"Required Diesel Dispatch: {deficit:.2f} MW\n")

    # Define Island Agents from coordinated config
    agents = [
        IslandAgent("Ko Tao",      load=tao_load,    gen=ca["ko_tao"]["renewable_mw"],    max_diesel=ca["ko_tao"]["diesel_mw"],    diesel_cost=ca["ko_tao"]["diesel_cost"]),
        IslandAgent("Ko Phangan",  load=phangan_load, gen=ca["ko_phangan"]["renewable_mw"], max_diesel=ca["ko_phangan"]["diesel_mw"], diesel_cost=ca["ko_phangan"]["diesel_cost"]),
        IslandAgent("Ko Samui",    load=samui_load,   gen=ca["ko_samui"]["renewable_mw"],   max_diesel=ca["ko_samui"]["diesel_mw"],   diesel_cost=ca["ko_samui"]["diesel_cost"])
    ]

    if deficit <= 0:
        print("✅ Grid stable. No diesel dispatch required.")
        return

    history, results = run_cluster_admm(agents, deficit)
    
    summary = []
    for a, output in zip(agents, results):
        summary.append({
            "Island": a.name,
            "Load": a.load,
            "Diesel Output": round(output, 2),
            "Cost/MW": a.diesel_cost,
            "Utilization": f"{(output/a.max_diesel)*100:.1f}%"
        })
    
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    print(f"\nFinal Cluster Balance: {sum(results):.3f} MW (Target: {deficit:.2f} MW)")
    print(f"ADMM Convergence: {len(history)} iterations, residual {history[-1]:.6f}")

if __name__ == "__main__":
    # Test Scenario: Peak load exceeds N-1 limit
    # Samui=120, Phangan=35, Tao=10 -> Total=165 (Deficit 4.2 MW)
    simulate_cluster_dispatch(samui_load=120, phangan_load=35, tao_load=10)
    
    print("\n" + "="*80 + "\n")
    
    # Test Scenario: Severe overload
    # Samui=130, Phangan=40, Tao=12 -> Total=182 (Deficit 21.2 MW)
    simulate_cluster_dispatch(samui_load=130, phangan_load=40, tao_load=12)

import numpy as np
import yaml

class IslandNode:
    def __init__(self, name, local_load, renewable_gen, bess_cap, diesel_cap):
        self.name = name
        self.load = local_load
        self.gen = renewable_gen
        self.bess_cap = bess_cap
        self.diesel_cap = diesel_cap
        # Initial mismatch
        self.p_export = renewable_gen - local_load 
        self.u = 0.0         # Dual variable

    def optimize_local(self, p_avg, u, rho):
        # Target export to help balance global mismatch
        # Local mismatch is self.gen - self.load
        local_mismatch = self.gen - self.load
        
        # Target: maximize help to cluster while staying within limits
        target = local_mismatch - (p_avg - u)
        
        # Physical constraints (MW export/import limits)
        max_export = max(0, self.gen + self.bess_cap - self.load)
        max_import = max(0, self.load + self.diesel_cap - self.gen)
        
        self.p_export = np.clip(target, -max_import, max_export)
        return self.p_export

def run_admm_consensus(nodes, max_iter=50, rho=0.1, tolerance=0.001):
    """
    Alternating Direction Method of Multipliers (ADMM)
    Ensures sum(p_export) = 0 (Cluster Balance)
    """
    history = []
    for i in range(max_iter):
        # 1. Local Optimization
        p_avg = np.mean([node.p_export for node in nodes])
        p_vals = [node.optimize_local(p_avg, node.u, rho) for node in nodes]
        
        # 2. Update Global Consensus (Avg export must be 0)
        p_avg_new = np.mean(p_vals)
        
        # 3. Update Dual Variables
        for node in nodes:
            node.u += (node.p_export - p_avg_new)
            
        # Check convergence
        residual = np.abs(p_avg_new)
        history.append(residual)
        if residual < tolerance:
            break
            
    return history, p_vals

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    cc = cfg["cluster"]
    ca = cc["assets"]

    # Scenario: Ko Tao has deficit, Samui has surplus
    # Initializing IslandNodes from coordinated config
    tao = IslandNode("Ko Tao",      local_load=8.5,  renewable_gen=ca["ko_tao"]["renewable_mw"],    bess_cap=ca["ko_tao"]["bess_mwh"],    diesel_cap=ca["ko_tao"]["diesel_mw"])
    phangan = IslandNode("Ko Phangan", local_load=12.0, renewable_gen=ca["ko_phangan"]["renewable_mw"], bess_cap=ca["ko_phangan"]["bess_mwh"], diesel_cap=ca["ko_phangan"]["diesel_mw"])
    samui = IslandNode("Ko Samui",    local_load=30.0, renewable_gen=ca["ko_samui"]["renewable_mw"],   bess_cap=ca["ko_samui"]["bess_mwh"],   diesel_cap=ca["ko_samui"]["diesel_mw"])
    
    nodes = [tao, phangan, samui]
    print(f"--- Multi-Island Resilience ADMM (Cluster: {cc['islands']}) ---")
    history, results = run_admm_consensus(nodes, **cc['admm'])
    
    for node, res in zip(nodes, results):
        status = "EXPORTING" if res > 0 else "IMPORTING"
        print(f"Island: {node.name:12} | Power: {res:6.2f} MW | Status: {status}")
    
    print(f"Convergence Residual: {history[-1]:.6f}")

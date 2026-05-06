import numpy as np
import yaml

class IslandNode:
    def __init__(self, name, local_load, renewable_gen, bess_cap, diesel_cap, tx_import_limit=float('inf'), tx_export_limit=float('inf')):
        self.name = name
        self.load = local_load
        self.gen = renewable_gen
        self.bess_cap = bess_cap
        self.diesel_cap = diesel_cap
        self.tx_import_limit = tx_import_limit
        self.tx_export_limit = tx_export_limit
        # Initial mismatch
        self.p_export = renewable_gen - local_load 
        self.u = 0.0         # Dual variable

    def optimize_local(self, p_avg, u, rho):
        # Target export to help balance global mismatch
        # Local mismatch is self.gen - self.load
        local_mismatch = self.gen - self.load
        
        # Target: maximize help to cluster while staying within limits
        target = local_mismatch - (p_avg - u)
        
        # Physical constraints (Net Export/Import limits)
        # Net Export = (Gen + Diesel + BESS_disc) - Load
        # Net Import = Load + BESS_char - (Gen + Diesel_min)
        bess_rate = self.bess_cap / 2.0
        max_net_export = (self.gen + self.diesel_cap + bess_rate) - self.load
        max_net_import = (self.load + bess_rate) - self.gen
        
        # Apply transmission limits (net flow limits)
        max_net_export = min(max_net_export, self.tx_export_limit)
        max_net_import = min(max_net_import, self.tx_import_limit)
        
        self.p_export = np.clip(target, -max_net_import, max_net_export)
        return self.p_export

def run_admm_consensus(nodes, max_iter=50, rho=0.1, tolerance=0.001, **kwargs):
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

    # Scenario: Extreme cluster shortage
    # Let's say Samui load is very high (90 MW)
    tao = IslandNode("Ko Tao",      local_load=8.5,  renewable_gen=ca["ko_tao"]["renewable_mw"],    bess_cap=ca["ko_tao"]["bess_mwh"],    diesel_cap=ca["ko_tao"]["diesel_mw"], tx_import_limit=16.0)
    phangan = IslandNode("Ko Phangan", local_load=25.0, renewable_gen=ca["ko_phangan"]["renewable_mw"], bess_cap=ca["ko_phangan"]["bess_mwh"], diesel_cap=ca["ko_phangan"]["diesel_mw"], tx_import_limit=45.0)
    samui = IslandNode("Ko Samui",    local_load=90.0, renewable_gen=ca["ko_samui"]["renewable_mw"],   bess_cap=ca["ko_samui"]["bess_mwh"],   diesel_cap=ca["ko_samui"]["diesel_mw"])
    
    mainland_limit = cc["admm"].get("mainland_n1_limit_mw", 65.0)
    # Simulate an N-1 event where mainland is severely bottlenecked (65 MW)
    mainland = IslandNode("Mainland", local_load=0.0, renewable_gen=65.0, bess_cap=0.0, diesel_cap=0.0, tx_export_limit=65.0)
    
    nodes = [tao, phangan, samui, mainland]
    print(f"--- Multi-Island Resilience ADMM (Cluster: {cc['islands']} + Mainland) ---")
    for node in nodes:
        bess_rate = node.bess_cap / 2.0
        max_ne = (node.gen + node.diesel_cap + bess_rate) - node.load
        max_ni = (node.load + bess_rate) - node.gen
        print(f"Node: {node.name:12} | Load: {node.load:5.1f} | Gen: {node.gen:4.1f} | Diesel: {node.diesel_cap:4.1f} | BESS Rate: {bess_rate:4.1f} | MaxNetExp: {max_ne:6.1f} | MaxNetImp: {max_ni:6.1f}")
    
    history, results = run_admm_consensus(nodes, **cc['admm'])
    
    for node, res in zip(nodes, results):
        status = "EXPORTING" if res > 0 else "IMPORTING"
        print(f"Island: {node.name:12} | Power: {res:6.2f} MW | Status: {status}")
    
    print(f"Convergence Residual: {history[-1]:.6f}")

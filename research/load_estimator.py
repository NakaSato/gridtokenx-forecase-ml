"""
Adaptive Observation Utility — GridTokenX
Estimates unobservable island loads (Samui, Phangan) based on total circuit flow.
"""
import numpy as np

def estimate_cluster_loads(total_import_mw: float, tao_observed_mw: float, 
                           tao_diesel_mw: float = 0.0):
    """
    Infers the unobservable Phangan and Samui loads.
    Logic: Net_Import = Samui + Phangan + Tao - Local_Gen
    Therefore: Samui + Phangan = Net_Import - Tao + Local_Gen
    We use a fixed ratio (75/25) for initial estimation until more data is available.
    """
    cluster_net = total_import_mw - tao_observed_mw + tao_diesel_mw
    
    # 75% Samui, 25% Phangan (Based on topology relative sizes)
    estimated_samui = max(0, cluster_net * 0.75)
    estimated_phangan = max(0, cluster_net * 0.25)
    
    return {
        "samui_mw": round(estimated_samui, 2),
        "phangan_mw": round(estimated_phangan, 2),
        "total_estimated_mw": round(cluster_net, 2)
    }

if __name__ == "__main__":
    # Test case: 100 MW Import, 8.5 MW Tao, 0 MW Diesel
    res = estimate_cluster_loads(100.0, 8.5)
    print(f"Estimation: {res}")

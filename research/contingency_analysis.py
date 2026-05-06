import numpy as np
import pandas as pd

# Constants for 115 kV Submarine Connector (N=2 Circuits)
VOLTAGE_KV = 115.0
LENGTH_KM = 24.14
R_OHM_PER_KM = 0.05
AMPACITY_AMPS = 850.0  # Thermal limit per circuit
PF = 0.95

# System Loads (Peak)
LOADS = {
    "Ko Tao": 7.7,
    "Ko Phangan": 26.0,
    "Ko Samui": 95.0,
}
TOTAL_LOAD_MW = sum(LOADS.values())

def run_contingency():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║            GridTokenX — N-1 Contingency & Resilience Analysis        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"Scenario: Mainland-Samui 115kV Connector (Double Circuit, {LENGTH_KM} km)")
    print(f"Total Cluster Load: {TOTAL_LOAD_MW:.2f} MW\n")

    results = []
    for mode in ["N (Normal)", "N-1 (Contingency)"]:
        n_circuits = 2 if "Normal" in mode else 1
        
        # Current per remaining circuit
        # I_total = P / (sqrt(3) * V * PF)
        total_current = (TOTAL_LOAD_MW * 1e6) / (np.sqrt(3) * VOLTAGE_KV * 1e3 * PF)
        current_per_circuit = total_current / n_circuits
        
        # Thermal Loading
        loading_pct = (current_per_circuit / AMPACITY_AMPS) * 100
        
        # Losses: P_loss = N * 3 * I_circuit^2 * R_total
        r_total = R_OHM_PER_KM * LENGTH_KM
        total_loss_kw = n_circuits * (3 * (current_per_circuit**2) * r_total) / 1000.0
        
        results.append({
            "Mode": mode,
            "Active Circuits": n_circuits,
            "Amps / Circuit": round(current_per_circuit, 1),
            "Thermal Load %": round(loading_pct, 1),
            "Total Loss (kW)": round(total_loss_kw, 2),
            "Loss Increase": "0%" if n_circuits == 2 else f"{(total_loss_kw / results[0]['Total Loss (kW)'] - 1)*100:.0f}%"
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Calculate Critical Failure Threshold
    # At what load does N-1 exceed 100% ampacity?
    # P_max = sqrt(3) * V * I_limit * PF
    p_max_n1 = (np.sqrt(3) * VOLTAGE_KV * AMPACITY_AMPS * PF) / 1e3
    print(f"\n[Resilience Metrics]")
    print(f"Critical N-1 Failure Threshold: {p_max_n1:.2f} MW")
    print(f"Current System Headroom (N-1): {p_max_n1 - TOTAL_LOAD_MW:.2f} MW")
    
    if TOTAL_LOAD_MW > p_max_n1:
        print("🚨 CRITICAL: System cannot survive N-1 failure at current peak load!")
    else:
        print("✅ PASS: System survives N-1 failure with current thermal ratings.")

if __name__ == "__main__":
    run_contingency()

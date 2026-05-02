import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

def run_sensitivity_study(base_df_path, temp_shifts=[-2, 0, 2, 4]):
    """
    Analyzes how temperature shifts impact the total Island Load and 
    subsequent BESS/Diesel requirements.
    """
    df = pd.read_parquet(base_df_path)
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    results = []
    
    for shift in temp_shifts:
        # Simulate shift in Dry Bulb Temp
        shifted_temp = df["Dry_Bulb_Temp"] + shift
        
        # Calculate new A/C driven load
        # Load = Base + (Temp - Threshold) * Coeff
        ac_load_new = np.maximum(0, (shifted_temp - cfg["data"]["ac_threshold_temp"]) * cfg["data"]["ac_coefficient"])
        
        # We assume the 'base' part of the load remains constant
        # Total_Load = Base_Load + AC_Load
        # So, New_Total = Original_Total - Original_AC + New_AC
        ac_load_orig = np.maximum(0, (df["Dry_Bulb_Temp"] - cfg["data"]["ac_threshold_temp"]) * cfg["data"]["ac_coefficient"])
        shifted_load = df["Island_Load_MW"] - ac_load_orig + ac_load_new
        shifted_load = shifted_load.clip(5, 12) # 12MW absolute peak in stress scenarios
        
        peak_load = shifted_load.max()
        avg_load = shifted_load.mean()
        total_mwh = shifted_load.sum()
        
        results.append({
            "temp_shift": shift,
            "peak_mw": round(peak_load, 2),
            "avg_mw": round(avg_load, 2),
            "total_gwh": round(total_mwh / 1000, 2)
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    os.makedirs("research", exist_ok=True)
    report = run_sensitivity_study("data/processed/island_grid.parquet")
    print("--- Climate Sensitivity Analysis (Temperature Shift) ---")
    print(report)
    
    # Export for researcher review
    report.to_csv("research/sensitivity_report.csv", index=False)
    print("\nReport saved to research/sensitivity_report.csv")

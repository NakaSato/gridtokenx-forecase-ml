"""
GridTokenX: Master Training & Backtest Pipeline
===============================================
Core Research Suite for Ko Tao-Phangan-Samui AI Forecasting.
Focus: Desirability (MAPE < 2.65%) and Viability (Savings > 22%).
"""
import subprocess
import sys
import os
import json

def run_step(name, command):
    print(f"\n>>> [STEP] {name}...")
    result = subprocess.run(command, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"!!! Error in {name}. Aborting.")
        sys.exit(1)

def main():
    print("====================================================")
    print("      GRIDTOKENX RESEARCH REFINEMENT PIPELINE       ")
    print("====================================================")

    # --- STAGE 1: PRE-TRAINING (Grid Physics Mastery) ---
    print("\n--- [STAGE 1] PRE-TRAINING ON 5-YEAR SYNTHETIC DATA ---")
    run_step("Synthetic Data Generation", ".venv/bin/python data/generate_dataset.py")
    
    # Force use of synthetic data
    if os.path.exists("data/ko_tao_grid_2023_locked.parquet"):
        os.rename("data/ko_tao_grid_2023_locked.parquet", "data/ko_tao_grid_2023_locked.bak")
    
    run_step("Synthetic Preprocessing", ".venv/bin/python data/preprocess.py")
    run_step("Pre-training Base Models", ".venv/bin/python models/hybrid_pipeline.py")

    # --- STAGE 2: FINE-TUNING (Climate Nuance Specialization) ---
    print("\n--- [STAGE 2] FINE-TUNING ON REAL ERA5 GROUND-TRUTH ---")
    if os.path.exists("data/ko_tao_grid_2023_locked.bak"):
        os.rename("data/ko_tao_grid_2023_locked.bak", "data/ko_tao_grid_2023_locked.parquet")
    else:
        run_step("Real Data Integration", ".venv/bin/python data/integrate_raw.py")
        
    # preprocess.py will now find the locked file and use it
    run_step("Real-World Preprocessing", ".venv/bin/python data/preprocess.py")
    
    # Run specialized Optuna Tuning on the Real data distribution
    print("\n--- [STAGE 3] HYPERPARAMETER RE-CALIBRATION (OPTUNA) ---")
    run_step("Recalibrating for Noise", ".venv/bin/python optimizer/tune.py --n-trials 10")
    
    # Final Specialized Training with calibrated params
    print("\n--- [STAGE 4] FINAL SPECIALIZED TRAINING ---")
    run_step("Training Final LGBM", ".venv/bin/python models/lgbm_model.py")
    run_step("Training Final TCN", ".venv/bin/python models/tcn_model.py")
    run_step("Fitting Meta-Learner", ".venv/bin/python models/hybrid_pipeline.py")
    
    # --- STAGE 5: VALIDATION ---
    print("\n--- [STAGE 5] VALIDATION ---")
    run_step("Final Backtest & Electrical Check", ".venv/bin/python evaluate.py")
    run_step("Generating Commissioning Report", ".venv/bin/python research/generate_report.py")
    
    print("\n" + "="*52)
    print("      RESEARCH PIPELINE COMPLETE: SUCCESS           ")
    print("="*52)
    
    # Display final report
    try:
        with open("results/evaluation_report.json") as f:
            r = json.load(f)
            print(f"\nFinal Forecast MAPE: {r['forecast']['mape']:.2f}%")
            print(f"Final Fuel Savings:  {r['dispatch']['fuel_savings_pct']:.2f}%")
            print(f"Asset Health (SoH):  {r['dispatch']['bess_soh_estimate']:.4f}")
    except:
        pass

if __name__ == "__main__":
    main()

import os
import sys
import subprocess
from pathlib import Path

def run_step(command: list, description: str):
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    # Run using the python executable of the virtual environment to ensure libraries are available
    python_exe = str(Path("venv/Scripts/python.exe").resolve())
    if not Path(python_exe).exists():
        python_exe = sys.executable # Fallback
        
    full_cmd = [python_exe, "-u"] + command
    print(f"Running command: {' '.join(full_cmd)}")
    
    result = subprocess.run(full_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Step failed with return code {result.returncode}!")
        sys.exit(result.returncode)
    print(f"[SUCCESS] Step completed successfully.")

def main():
    print("="*80)
    print("LAUNCHING ADENOSINE SELECTIVITY MODEL PIPELINE")
    print("="*80)
    
    # Step 1: Retrain Production Models with conformal prediction
    run_step(["-m", "src.retrain_production"], "Production Model Training & Conformal Prediction (MAPIE)")
    
    # Step 2: Build Direct Selectivity Models
    run_step(["-m", "src.selectivity_models"], "Pairwise Affinity Difference Selectivity Models")
    
    # Step 3: Run Y-Randomization on A2A subtype
    run_step(["-m", "src.y_randomization", "--subtype", "A2A", "--iterations", "15"], "Y-Randomization Robustness Check (A2A)")
    
    # Step 4: Run SHAP Explainability analysis on A2A subtype
    run_step(["-m", "src.shap_analysis", "--subtype", "A2A"], "SHAP Tree Explainability & Chemical Sanity (A2A)")
    
    # Step 5: Run A1 Quality Diagnostics
    run_step(["-m", "src.diagnostics.a1_diagnosis"], "A1 Receptor Dataset Bottleneck Diagnostics")
    
    # Step 6: Run Conformal Model Evaluator (precise mode)
    run_step(["-m", "src.evaluator"], "Conformal Model Metrics Evaluator (Precise Mode)")
    
    # Step 7: Run Streamlit Example Predictions Generator (precise mode)
    run_step(["results.py"], "Streamlit Example Predictions Generator")
    
    print("\n" + "="*80)
    print("ALL PIPELINE COMPONENTS COMPLETED COMPREHENSIVELY!")
    print("Calibrated conformal models, direct selectivity estimators, and SHAP explanations are ready.")
    print("="*80)

if __name__ == "__main__":
    main()

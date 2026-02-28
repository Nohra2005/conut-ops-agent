import subprocess
import os
import sys

def run_pipeline():
    # Since retrain.py is IN services/forecasting, 
    # its directory is the same as the scripts it needs to run.
    forecast_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the scripts in order
    pipeline_scripts = [
        "preprocessing.py",
        "train_model.py",
        "forecast_service.py"
    ]
    
    print("="*30)
    print("STARTING MODULAR RETRAINING PIPELINE")
    print(f"Current working directory: {forecast_dir}")
    print("="*30)
    
    for script in pipeline_scripts:
        script_path = os.path.join(forecast_dir, script)
        
        # Verify file existence
        if not os.path.exists(script_path):
            print(f"\n!!! ERROR: Cannot find {script} at {script_path}")
            continue

        print(f"\n>>> Executing: {script}")
        
        try:
            # Run the script using the current python executable
            subprocess.run(
                [sys.executable, script_path],
                check=True,
                text=True,
                capture_output=False 
            )
            print(f"--- {script} completed successfully ---")
        except subprocess.CalledProcessError as e:
            print(f"\n!!! ERROR: {script} failed.")
            print(f"Reason: {e}")
            return 

    print("\n" + "="*30)
    print("PIPELINE SUCCESSFUL")
    print("MLflow tracking and local assets updated.")
    print("="*30)

if __name__ == "__main__":
    run_pipeline()
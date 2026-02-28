import subprocess
import os
import sys

def run_pipeline():
    # Dynamic path resolution to find the forecasting folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    forecast_dir = os.path.join(base_dir, "services", "forecasting")
    
    # Define the scripts in order
    pipeline_scripts = [
        "preprocessing.py",
        "train_model.py",
        "forecast_service.py"
    ]
    
    print("="*30)
    print("STARTING RETRAINING PIPELINE")
    print("="*30)
    
    for script in pipeline_scripts:
        script_path = os.path.join(forecast_dir, script)
        print(f"\n>>> Executing: {script}")
        
        try:
            # Run the script using the current python executable
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                text=True,
                capture_output=False # Set to True if you want to hide internal logs
            )
            print(f"--- {script} completed successfully ---")
        except subprocess.CalledProcessError as e:
            print(f"\n!!! ERROR: {script} failed.")
            print(f"Reason: {e}")
            return # Stop the pipeline if one step fails

    print("\n" + "="*30)
    print("PIPELINE SUCCESSFUL")
    print("All models and charts updated.")
    print("="*30)

if __name__ == "__main__":
    run_pipeline()
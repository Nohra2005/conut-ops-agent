import subprocess
import os
import sys

def run_pipeline():
    # Dynamic path resolution
    # BASE_DIR is .../conut-ops-agent/pipeline
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ROOT_DIR is .../conut-ops-agent
    root_dir = os.path.normpath(os.path.join(current_dir, ".."))
    
    # Path to the forecasting scripts
    forecast_dir = os.path.join(root_dir, "services", "forecasting")
    
    # Define the scripts in order
    pipeline_scripts = [
        "preprocessing.py",
        "train_model.py",
        "forecast_service.py"
    ]
    
    print("="*40)
    print("STARTING RETRAINING PIPELINE")
    print(f"Target Directory: {forecast_dir}")
    print("="*40)
    
    for script in pipeline_scripts:
        script_path = os.path.join(forecast_dir, script)
        
        # Check if script exists before running
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

    print("\n" + "="*40)
    print("PIPELINE SUCCESSFUL")
    print("Forecasting model and assets updated.")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()
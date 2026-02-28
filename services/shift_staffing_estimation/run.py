import subprocess
import sys
import os

# ensure script runs in its own directory
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# dynamically find the root of the project (two levels up)
root_dir = os.path.normpath(os.path.join(current_dir, "..", ".."))

# define the default file locations
default_attendance = os.path.join(root_dir, "data", "raw_data", "REP_S_00461.csv")
default_inventory = os.path.join(root_dir, "services", "forecasting", "master_daily_inventory.csv")

if len(sys.argv) > 1 and sys.argv[1] == "update":
    print("running staffing mlflow pipeline...")
    
    pipeline_args = sys.argv[2:] 
    
    # if you didn't provide custom paths, use the defaults automatically!
    if not pipeline_args:
        print("no custom paths provided. using default project files...")
        pipeline_args = [
            "--attendance", default_attendance,
            "--inventory", default_inventory
        ]
    
    try:
        subprocess.run([sys.executable, "pipeline.py"] + pipeline_args, check=True)
        print("\n✅ pipeline complete. if the server is running, hit the /reload-model endpoint!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ pipeline failed: {e}")
        
elif len(sys.argv) > 1 and sys.argv[1] == "mlflow":
    print("starting mlflow ui on http://localhost:5000...")
    subprocess.run([sys.executable, "-m", "mlflow", "ui"])
    
else:
    print("starting fastapi server on http://127.0.0.1:8005...")
    subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", "8005"])
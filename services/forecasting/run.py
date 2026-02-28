import subprocess
import sys
import os

# ensure script runs in its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1 and sys.argv[1] == "update":
    print("="*40)
    print("STARTING DETERMINISTIC FORECAST PIPELINE")
    print("="*40)
    
    try:
        print("\n>>> 1. Executing: preprocessing.py (Recalculating weights and quantities)")
        subprocess.run([sys.executable, "preprocessing.py"], check=True)
        
        print("\n>>> 2. Executing: forecast_service.py (Running sanity tests)")
        subprocess.run([sys.executable, "forecast_service.py"], check=True)
        
        print("\n✅ Data pipeline complete. (Note: If the API is currently running, you will need to restart it so it loads the fresh CSV into memory).")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed: {e}")
else:
    print("Starting FastAPI server on http://127.0.0.1:8001...")
    subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", "8001"])
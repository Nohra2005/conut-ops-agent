import subprocess
import sys
import os

# ensure script runs in its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1 and sys.argv[1] == "update":
    print("="*40)
    print("STARTING GROWTH STRATEGY PIPELINE")
    print("="*40)
    
    try:
        print("\n>>> 1. Executing: preprocessing.py (Calculating attachment rates and gaps)")
        subprocess.run([sys.executable, "preprocessing.py"], check=True)
        
        print("\n>>> 2. Executing: growth_service.py (Running opportunity tests)")
        subprocess.run([sys.executable, "growth_service.py"], check=True)
        
        print("\n✅ Growth data refreshed. (Restart the API if it is currently running to load the new metrics).")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed: {e}")
else:
    print("Starting FastAPI server on http://127.0.0.1:8002...")
    subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", "8002"])
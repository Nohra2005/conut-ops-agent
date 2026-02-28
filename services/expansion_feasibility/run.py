import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1 and sys.argv[1] == "update":
    print("Running expansion data refresh pipeline...")
    subprocess.run([sys.executable, "update_pipeline.py"], check=True)
    print("\nDone. Restart api to reload.")
else:
    subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", "8004"])
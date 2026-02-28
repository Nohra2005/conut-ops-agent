import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) > 1 and sys.argv[1] == "update":
    print("Running full combo pipeline...")
    for script in ["price_parser.py", "basket_parser.py", "combo_analysis.py", "generate_combos.py"]:
        print(f"\n>>> {script}")
        result = subprocess.run([sys.executable, script], check=True)
    print("\nPipeline complete.")
else:
    subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", "8003"])
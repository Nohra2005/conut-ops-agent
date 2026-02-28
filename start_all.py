"""
start_all.py — Launches all 4 Conut microservices simultaneously
================================================================
Place this file at the ROOT of your project (same level as the services/ folder).

Usage:
    python start_all.py          # start all services
    python start_all.py --stop   # (manual: just close the terminal window)

Ports:
    8001 → Forecasting
    8002 → Growth
    8003 → Combo Optimization
    8004 → Expansion Feasibility
"""

import subprocess
import sys
import os
import time
import signal
import threading

# ── Service definitions ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVICES = [
    {
        "name": "Forecasting",
        "port": 8001,
        "run_py": os.path.join(BASE_DIR, "services", "forecasting", "run.py"),
    },
    {
        "name": "Growth",
        "port": 8002,
        "run_py": os.path.join(BASE_DIR, "services", "growth", "run.py"),
    },
    {
        "name": "Combo Optimization",
        "port": 8003,
        "run_py": os.path.join(BASE_DIR, "services", "combo_optimization", "run.py"),
    },
    {
        "name": "Expansion Feasibility",
        "port": 8004,
        "run_py": os.path.join(BASE_DIR, "services", "expansion_feasibility", "run.py"),
    },
]

processes = []


def stream_output(proc, service_name, color_code):
    """Stream a service's stdout/stderr to console with a colored prefix."""
    prefix = f"\033[{color_code}m[{service_name}]\033[0m "
    for line in proc.stdout:
        print(prefix + line, end="", flush=True)


def start_services():
    print("\n" + "═" * 60)
    print("  🟡  CONUT OPS — Starting all services")
    print("═" * 60)

    colors = ["33", "36", "35", "32"]  # yellow, cyan, magenta, green

    for i, svc in enumerate(SERVICES):
        if not os.path.exists(svc["run_py"]):
            print(f"  ⚠  Skipping {svc['name']} — run.py not found at {svc['run_py']}")
            continue

        proc = subprocess.Popen(
            [sys.executable, svc["run_py"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(proc)

        # Stream output in a background thread so all services log concurrently
        t = threading.Thread(
            target=stream_output,
            args=(proc, svc["name"], colors[i % len(colors)]),
            daemon=True,
        )
        t.start()

        print(f"  ✓  {svc['name']} started → http://127.0.0.1:{svc['port']} (PID {proc.pid})")
        time.sleep(0.5)  # stagger slightly to avoid port conflicts

    print("\n" + "═" * 60)
    print("  All services running. Open dashboard.html in your browser.")
    print("  Press Ctrl+C to stop all services.")
    print("═" * 60 + "\n")


def stop_all(sig=None, frame=None):
    print("\n\n  Stopping all services...")
    for proc in processes:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    print("  All services stopped. Goodbye.\n")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    start_services()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
            # Check if any process died unexpectedly
            for proc in processes:
                if proc.poll() is not None:
                    pass  # silently handled — logs already printed
    except KeyboardInterrupt:
        stop_all()
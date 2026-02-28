"""
pipeline.py
───────────
Full pipeline: parse data → update CSV → train multiple models → track with MLflow → auto-select best

SETUP (first time only)
────────────────────────
    pip install scikit-learn pandas mlflow numpy

USAGE
─────
Add new data + retrain:
    python pipeline.py --attendance REP_S_00461.csv --inventory master_daily_inventory.csv

Multiple attendance files at once:
    python pipeline.py --attendance REP_S_00461.csv REP_S_00462.csv --inventory master_daily_inventory.csv

Retrain only (no new files):
    python pipeline.py

Predict using the best registered model:
    python pipeline.py --predict --branch "Main Street Coffee" --demand 420
    python pipeline.py --predict --branch "Main Street Coffee" --demand 420 --date 2025-12-15

Open MLflow UI to compare runs, metrics, models:
    mlflow ui
    → open http://localhost:5000 in your browser

HOW MLFLOW WORKS HERE
─────────────────────
Every retrain run:
  - Tries 3 models (RandomForest, GradientBoosting, Ridge)
  - Logs metrics (MAE, RMSE, R², CV-MAE), params, and the model artifact for each
  - Registers the best model in the MLflow Model Registry as "employee_demand_model"
  - Tags it with the "champion" alias — predictions always use the champion

In the UI you can:
  - Compare all runs side by side
  - See which model type wins each time
  - Manually promote any older version back to champion if needed
  - View how metrics improve as more data is added over time

FILES CREATED
─────────────
  training_data.csv   — master dataset, grows every run
  mlruns/             — MLflow tracking store (auto-created, keep this folder)
  model_info.txt      — plain-text summary of the last run
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_CSV   = "training_data.csv"
MODEL_INFO     = "model_info.txt"
MODEL_NAME     = "employee_demand_model"
CHAMPION_ALIAS = "champion"
EXPERIMENT     = "branch_staffing"

# ── Branch aliases (edit if the same branch appears under different names) ────
BRANCH_ALIASES: dict = {
    # "old spelling": "canonical name",
}

def normalize_branch(name: str) -> str:
    return BRANCH_ALIASES.get(name.strip(), name.strip())


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSE ATTENDANCE  →  {(date, branch): {emp_ids}}
# ─────────────────────────────────────────────────────────────────────────────
def parse_attendance(filepaths: list) -> dict:
    combined = defaultdict(set)
    date_re  = re.compile(r"^\d{2}-[A-Za-z]{3}-\d{2}$")

    for filepath in filepaths:
        current_emp, current_branch = None, None
        with open(filepath, encoding="utf-8-sig", newline="") as f:
            for line in f:
                row = [c.strip() for c in line.rstrip("\r\n").split(",")]

                if len(row) >= 2 and "EMP ID" in row[1]:
                    current_emp, current_branch = row[1], None
                    continue

                if current_emp is not None and current_branch is None:
                    candidate = row[1] if len(row) > 1 else row[0]
                    if (candidate
                            and "PUNCH"     not in candidate
                            and "From Date" not in candidate
                            and re.search(r"[A-Za-z]", candidate)):
                        current_branch = normalize_branch(candidate)
                        continue

                if current_emp and current_branch and row[0] and date_re.match(row[0]):
                    try:
                        d = datetime.strptime(row[0], "%d-%b-%y").strftime("%Y-%m-%d")
                        combined[(d, current_branch)].add(current_emp)
                    except ValueError:
                        pass

        print(f"    ✓ {filepath}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARSE INVENTORY  →  {(date, branch): total_qty}
#    Dates in source files are 1 year ahead — shift back automatically
# ─────────────────────────────────────────────────────────────────────────────
def parse_inventory(filepath: str) -> dict:
    inventory = defaultdict(float)
    with open(filepath, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            date   = row.get("date", "").strip()
            branch = normalize_branch(row.get("branch", "").strip())
            qty    = row.get("daily_qty_rounded", "0").strip()
            if date and branch:
                try:
                    d    = datetime.strptime(date, "%Y-%m-%d")
                    date = d.replace(year=d.year - 1).strftime("%Y-%m-%d")
                    inventory[(date, branch)] += float(qty)
                except ValueError:
                    pass
    print(f"    ✓ {filepath}")
    return inventory


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD TRAINING ROWS  (only rows with both employees AND items)
# ─────────────────────────────────────────────────────────────────────────────
def build_rows(attendance: dict, inventory: dict) -> list:
    rows = []
    for (date, branch) in sorted(set(attendance) | set(inventory)):
        n_emp   = len(attendance.get((date, branch), set()))
        n_items = round(inventory.get((date, branch), 0.0), 4)
        if n_emp > 0 and n_items > 0:
            rows.append({"branch": branch, "date": date,
                         "num_employees": n_emp, "total_items": n_items})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 4. CSV MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def load_existing_csv(filepath: str) -> list:
    rows = []
    if Path(filepath).exists():
        with open(filepath, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append({"branch":        row["branch"].strip(),
                             "date":          row["date"].strip(),
                             "num_employees": int(row["num_employees"]),
                             "total_items":   float(row["total_items"])})
        print(f"  Loaded {len(rows)} existing rows from '{filepath}'")
    else:
        print(f"  No existing '{filepath}' — starting fresh")
    return rows


def merge_and_save(existing: list, new_rows: list, filepath: str) -> list:
    merged = {(r["branch"], r["date"]): r for r in existing}
    for r in new_rows:
        merged[(r["branch"], r["date"])] = r          # new data wins
    all_rows = [merged[k] for k in sorted(merged)]
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["branch", "date", "num_employees", "total_items"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved {len(all_rows)} rows (+{len(merged) - len(existing)} new) → '{filepath}'")
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    df = df.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["branch_enc"] = le.transform(df["branch"])
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([4, 5]).astype(int)   # Fri/Sat
    return df[["branch_enc", "total_items", "dayofweek", "month", "is_weekend"]]


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN + MLFLOW TRACKING
# ─────────────────────────────────────────────────────────────────────────────
def train(all_rows: list):
    df = pd.DataFrame(all_rows)
    le = LabelEncoder()
    le.fit(df["branch"])
    X = make_features(df, le)
    y = df["num_employees"]

    # Candidate models — add more here anytime
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
        "Ridge": Ridge(alpha=1.0),
    }

    mlflow.set_experiment(EXPERIMENT)

    has_split = len(df) >= 10
    if has_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        print("  ⚠  Small dataset — evaluating on full data")

    best_mae, best_run_id, best_name, best_version_num = float("inf"), None, None, None
    print()

    for model_name, model in candidates.items():
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):

            # Log hyperparams
            mlflow.log_params({
                "model_type":    model_name,
                "training_rows": len(df),
                "num_branches":  df["branch"].nunique(),
                "branches":      str(sorted(df["branch"].unique().tolist())),
                **{k: str(v) for k, v in model.get_params().items()},
            })

            # Train
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Metrics
            mae  = mean_absolute_error(y_test, preds)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2   = r2_score(y_test, preds)
            cv   = cross_val_score(model, X, y,
                                   cv=min(5, len(df)),
                                   scoring="neg_mean_absolute_error")
            cv_mae = float(-cv.mean())

            mlflow.log_metrics({
                "mae":    round(mae,    4),
                "rmse":   round(rmse,   4),
                "r2":     round(r2,     4),
                "cv_mae": round(cv_mae, 4),
            })

            # Log model to registry
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                metadata={"model_type": model_name,
                          "le_classes": le.classes_.tolist()},
            )
            # Also save label encoder as a JSON artifact so predict() can reload it
            mlflow.log_dict({"classes": le.classes_.tolist()}, "label_encoder.json")

            run_id = mlflow.active_run().info.run_id
            print(f"  {model_name:20s}  MAE={mae:.3f}  RMSE={rmse:.3f}"
                  f"  R²={r2:.3f}  CV-MAE={cv_mae:.3f}  [{run_id[:8]}]")

            if mae < best_mae:
                best_mae    = mae
                best_run_id = run_id
                best_name   = model_name

    # ── Promote best model as "champion" ─────────────────────────────────────
    client = mlflow.MlflowClient()
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in sorted(all_versions, key=lambda x: int(x.version), reverse=True):
        if v.run_id == best_run_id:
            best_version_num = v.version
            break

    if best_version_num:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, best_version_num)
        print(f"\n  🏆 Champion → {best_name}  MAE={best_mae:.3f}"
              f"  (version {best_version_num})")

    # Plain-text summary
    with open(MODEL_INFO, "w") as f:
        f.write(f"Last trained : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Training rows: {len(df)}\n")
        f.write(f"Branches     : {sorted(df['branch'].unique().tolist())}\n")
        f.write(f"Best model   : {best_name}\n")
        f.write(f"Best MAE     : {best_mae:.3f}\n")
        f.write(f"Champion ver : {best_version_num}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PREDICT — always loads the champion from the registry
# ─────────────────────────────────────────────────────────────────────────────
def predict(branch: str, demand: float, date_str: str = None):
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    except Exception as e:
        print(f"  Could not load champion model: {e}")
        print("  Run the pipeline with data first.")
        return

    client  = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    le_path = client.download_artifacts(version.run_id, "label_encoder.json")
    with open(le_path) as f:
        classes = json.load(f)["classes"]
    le = LabelEncoder()
    le.fit(classes)

    if branch not in le.classes_:
        print(f"  Unknown branch '{branch}'.")
        print(f"  Known: {list(le.classes_)}")
        return

    d          = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.today()
    branch_enc = le.transform([branch])[0]
    X = pd.DataFrame([{
        "branch_enc":  branch_enc,
        "total_items": demand,
        "dayofweek":   d.weekday(),
        "month":       d.month,
        "is_weekend":  int(d.weekday() in [4, 5]),
    }])

    result = model.predict(X)[0]
    print(f"\n  Branch   : {branch}")
    print(f"  Demand   : {demand} items")
    print(f"  Date     : {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")
    print(f"  Model    : champion  (v{version.version})")
    print(f"  → Predicted employees needed: {round(result, 1)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attendance", nargs="+", default=[])
    ap.add_argument("--inventory",  default=None)
    ap.add_argument("--predict",    action="store_true")
    ap.add_argument("--branch",     default=None)
    ap.add_argument("--demand",     type=float, default=None)
    ap.add_argument("--date",       default=None, help="YYYY-MM-DD for prediction")
    args = ap.parse_args()

    if args.predict:
        if not args.branch or args.demand is None:
            print("Usage: python pipeline.py --predict --branch 'Main Street Coffee' --demand 420")
            return
        predict(args.branch, args.demand, args.date)
        return

    new_rows = []
    if args.attendance and args.inventory:
        print("\n[1] Parsing attendance …")
        attendance = parse_attendance(args.attendance)
        print("\n[2] Parsing inventory …")
        inventory  = parse_inventory(args.inventory)
        print("\n[3] Building rows …")
        new_rows   = build_rows(attendance, inventory)
        print(f"  {len(new_rows)} complete rows")
        if not new_rows:
            print("  ⚠  No overlapping dates — check that both files cover the same month.")
    else:
        print("\n[!] No new files provided — retraining on existing data.\n")

    print("\n[4] Updating training CSV …")
    existing = load_existing_csv(TRAINING_CSV)
    all_rows = merge_and_save(existing, new_rows, TRAINING_CSV)

    print("\n[5] Training & tracking with MLflow …")
    train(all_rows)

    print("\n✅ Done!")
    print("  View runs  : mlflow ui  →  http://localhost:5000")
    print("  Predict    : python pipeline.py --predict --branch 'Main Street Coffee' --demand 420\n")


if __name__ == "__main__":
    main()

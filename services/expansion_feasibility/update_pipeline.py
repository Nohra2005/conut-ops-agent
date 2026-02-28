"""
retrain.py — Expansion Feasibility: Data Refresh + MLflow Model Training

Phases:
  1. Ingest CSV sales/customers data → write JSON files (same as before)
  2. Build a feature matrix from the JSON data
  3. Label each branch using the deterministic rule-based scoring logic
  4. Train RandomForest, GradientBoosting, LogisticRegression, and SVM
     across multiple hyperparameter configurations under the MLflow
     experiment named "expansion"
  5. Select the best model by weighted F1 (LOO cross-val) and tag it

Usage:
    python retrain.py
    python retrain.py --dry-run          # simulate only, no writes / no MLflow
    python retrain.py --skip-refresh     # skip CSV ingest, retrain from existing JSON
"""

import csv
import json
import os
import warnings
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.base import clone as sklearn_clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data", "raw_data"))

DEFAULT_SALES_CSV      = os.path.join(DATA_DIR, "rep_s_00334_1_SMRY.csv")
DEFAULT_CUSTOMERS_CSV  = os.path.join(DATA_DIR, "rep_s_00150.csv")
DEFAULT_SALES_JSON     = os.path.join(BASE_DIR, "monthly_sales_by_branch.json")
DEFAULT_CUSTOMERS_JSON = os.path.join(BASE_DIR, "customers.json")
BEST_MODEL_PATH        = os.path.join(BASE_DIR, "best_expansion_model.json")

MLFLOW_EXPERIMENT = "expansion"

# ── Constants (mirror expansion_model.py) ─────────────────────────────────────
MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
MONTH_ORDER = {m: i for i, m in enumerate(MONTHS)}

CUSTOMER_SECTION_RANGES = [
    (0,   99,  "Conut - Tyre"),
    (99,  302, "Conut"),
    (302, 563, "Conut Jnah"),
    (563, None, "Main Street Coffee"),
]

P1_SATURATION_THRESHOLD = 0.80
P1_AMBER_THRESHOLD      = 0.60
P1_MIN_CONSECUTIVE      = 2
P2_R2_MIN_GREEN         = 0.70
P2_R2_MIN_AMBER         = 0.40
P2_MIN_MONTHS           = 3
P3_MIN_CUSTOMERS_GREEN  = 150
P3_MIN_CUSTOMERS_AMBER  = 50
P3_REPEAT_RATE_GREEN    = 0.15
P3_REPEAT_RATE_AMBER    = 0.05
CLOSURE_MAX_LAST_UTIL   = 0.20
CLOSURE_MAX_MOM_CHANGE  = -0.50

# Label mapping: 0=Do Not Expand / Close, 1=Monitor, 2=Expand
VERDICT_MAP = {
    "EXPAND":           2,
    "MONITOR CLOSELY":  1,
    "MONITOR":          1,
    "DO NOT EXPAND":    0,
    "CONSIDER CLOSURE": 0,
}
LABEL_NAMES = {0: "DO_NOT_EXPAND", 1: "MONITOR", 2: "EXPAND"}

# ── Model grid: 4 algorithms × 4 hyperparameter sets = 16 runs ────────────────
MODEL_GRID = [
    {
        "name": "RandomForest",
        "model_cls": RandomForestClassifier,
        "needs_scaling": False,
        "param_grid": [
            {"n_estimators": 50,  "max_depth": 3,    "min_samples_split": 2, "random_state": 42},
            {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 2, "random_state": 42},
            {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "random_state": 42},
            {"n_estimators": 100, "max_depth": 3,    "min_samples_split": 5, "random_state": 42},
        ],
    },
    {
        "name": "GradientBoosting",
        "model_cls": GradientBoostingClassifier,
        "needs_scaling": False,
        "param_grid": [
            {"n_estimators": 50,  "learning_rate": 0.10, "max_depth": 2, "random_state": 42},
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
            {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 3, "random_state": 42},
            {"n_estimators": 200, "learning_rate": 0.01, "max_depth": 2, "random_state": 42},
        ],
    },
    {
        "name": "LogisticRegression",
        "model_cls": LogisticRegression,
        "needs_scaling": True,
        "param_grid": [
            {"C": 0.01, "max_iter": 1000, "random_state": 42},
            {"C": 0.10, "max_iter": 1000, "random_state": 42},
            {"C": 1.00, "max_iter": 1000, "random_state": 42},
            {"C": 10.0, "max_iter": 1000, "random_state": 42},
        ],
    },
    {
        "name": "SVM",
        "model_cls": SVC,
        "needs_scaling": True,
        "param_grid": [
            {"C": 0.1,  "kernel": "rbf",    "probability": True, "random_state": 42},
            {"C": 1.0,  "kernel": "rbf",    "probability": True, "random_state": 42},
            {"C": 10.0, "kernel": "rbf",    "probability": True, "random_state": 42},
            {"C": 1.0,  "kernel": "linear", "probability": True, "random_state": 42},
        ],
    },
]


# ── CSV Ingestion ──────────────────────────────────────────────────────────────
def to_float(s):
    try:
        return float(str(s).strip().replace(",", "").replace('"', ""))
    except Exception:
        return None


def ingest_sales_csv(filepath):
    records = {}
    cur_branch = None
    with open(filepath, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            c0 = row[0].strip()
            if c0.startswith("Branch Name:"):
                cur_branch = c0.replace("Branch Name:", "").strip()
                if cur_branch not in records:
                    records[cur_branch] = []
                continue
            if cur_branch and c0 in MONTHS:
                year  = int(row[2].strip()) if len(row) > 2 and row[2].strip().isdigit() else None
                total = to_float(row[3]) if len(row) > 3 else None
                existing = [r["month"] for r in records[cur_branch]]
                if c0 not in existing and total is not None:
                    records[cur_branch].append({"month": c0, "year": year, "total_scaled": total})

    output = []
    for branch, monthly in records.items():
        monthly_sorted = sorted(monthly, key=lambda m: MONTH_ORDER.get(m["month"], 99))
        if not monthly_sorted:
            continue
        ath_row  = max(monthly_sorted, key=lambda m: m["total_scaled"])
        last_row = monthly_sorted[-1]
        output.append({
            "branch":        branch,
            "all_time_high": round(ath_row["total_scaled"], 2),
            "ath_month":     ath_row["month"],
            "current_month": {
                "month": last_row["month"],
                "year":  last_row["year"],
                "total_scaled": round(last_row["total_scaled"], 2),
            },
            "monthly_sales": [
                {
                    "month":        r["month"],
                    "year":         int(r["year"]) if r["year"] else None,
                    "total_scaled": round(r["total_scaled"], 2),
                }
                for r in monthly_sorted
            ],
        })
    return output


def ingest_customers_csv(filepath):
    def parse_row(row, branch):
        if not row or not row[0].startswith("Person_"):
            return None
        name  = row[0].strip()
        addr  = row[1].strip() if len(row) > 1 else ""
        phone = row[2].strip() if len(row) > 2 else ""
        c3    = row[3].strip() if len(row) > 3 else ""
        if c3.startswith("20"):
            fo    = f"{row[3].strip()} {row[4].strip()}".rstrip(":").strip() if len(row) > 4 else ""
            lo    = f"{row[5].strip()} {row[6].strip()}".rstrip(":").strip() if len(row) > 6 else ""
            total = to_float(row[7]) if len(row) > 7 else None
            n_ord = to_float(row[8]) if len(row) > 8 else None
        else:
            fo    = f"{row[4].strip()} {row[5].strip()}".rstrip(":").strip() if len(row) > 5 else ""
            lo    = f"{row[6].strip()} {row[7].strip()}".rstrip(":").strip() if len(row) > 7 else ""
            total = to_float(row[8]) if len(row) > 8 else None
            n_ord = to_float(row[9]) if len(row) > 9 else None
        orders = int(n_ord) if n_ord is not None and n_ord < 10_000 else None
        return {
            "branch": branch, "customer_name": name,
            "address": addr or None, "phone": phone or None,
            "first_order": fo or None, "last_order": lo or None,
            "total_sales_scaled": total, "num_orders": orders,
        }

    with open(filepath, encoding="utf-8-sig") as f:
        all_lines = f.readlines()
    customers = []
    for start, end, branch in CUSTOMER_SECTION_RANGES:
        for line in all_lines[start:end]:
            row    = next(csv.reader([line]))
            parsed = parse_row(row, branch)
            if parsed:
                customers.append(parsed)
    return customers


def write_json(data, path, dry_run=False):
    if dry_run:
        print(f"  [dry-run] Would write to {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Written: {path}")


# ── Feature Engineering ────────────────────────────────────────────────────────
def _longest_streak(bools):
    best = cur = 0
    for v in bools:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def extract_features(branch_data, customers):
    """Return an ordered dict of numeric features for one branch."""
    branch  = branch_data["branch"]
    ath     = branch_data["all_time_high"]
    monthly = branch_data["monthly_sales"]
    sales   = np.array([m["total_scaled"] for m in monthly], dtype=float)
    n       = len(sales)

    # ── Pillar 1 (volume saturation) ──────────────────────────────────────────
    above_80 = [bool(s / ath >= P1_SATURATION_THRESHOLD) for s in sales] if ath else [False] * n
    above_60 = [bool(s / ath >= P1_AMBER_THRESHOLD)      for s in sales] if ath else [False] * n
    streak_80     = _longest_streak(above_80)
    streak_60     = _longest_streak(above_60)
    last_util_pct = float(sales[-1] / ath * 100) if ath else 0.0
    avg_util_pct  = float(np.mean(sales) / ath * 100) if ath else 0.0

    # ── Pillar 2 (growth trajectory) ──────────────────────────────────────────
    if n >= 2:
        x               = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, sales, 1)
        y_hat           = slope * x + intercept
        ss_res          = float(np.sum((sales - y_hat) ** 2))
        ss_tot          = float(np.sum((sales - np.mean(sales)) ** 2))
        r_squared       = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        mom_pct         = float((sales[-1] - sales[-2]) / sales[-2] * 100) if sales[-2] > 0 else 0.0
        slope_norm      = float(slope / ath) if ath else 0.0
    else:
        slope_norm = r_squared = mom_pct = 0.0

    # ── Pillar 3 (delivery density) ────────────────────────────────────────────
    bc       = [c for c in customers if c["branch"] == branch]
    n_cust   = len(bc)
    repeat   = sum(1 for c in bc if (c["num_orders"] or 0) > 1)
    rep_rate = repeat / n_cust if n_cust else 0.0
    revenues = [c["total_sales_scaled"] for c in bc if c["total_sales_scaled"]]
    avg_rev  = float(sum(revenues) / len(revenues)) if revenues else 0.0
    orders_  = [c["num_orders"] for c in bc if c["num_orders"]]
    avg_ord  = float(sum(orders_) / len(orders_)) if orders_ else 0.0

    return {
        # Pillar 1
        "last_util_pct":           round(last_util_pct, 4),
        "avg_util_pct":            round(avg_util_pct, 4),
        "streak_above_80":         float(streak_80),
        "streak_above_60":         float(streak_60),
        "pct_months_above_80":     round(sum(above_80) / n, 4),
        "pct_months_above_60":     round(sum(above_60) / n, 4),
        # Pillar 2
        "slope_normalized":        round(slope_norm, 6),
        "r_squared":               round(r_squared, 4),
        "mom_pct":                 round(mom_pct, 4),
        "n_months":                float(n),
        # Pillar 3
        "customer_count":          float(n_cust),
        "repeat_rate_pct":         round(rep_rate * 100, 4),
        "avg_order_value":         round(avg_rev, 4),
        "avg_orders_per_customer": round(avg_ord, 4),
    }


def _compute_label(branch_data, customers):
    """
    Deterministic rule-based label (mirrors expansion_model.py logic).
    Returns: 0=DO_NOT_EXPAND, 1=MONITOR, 2=EXPAND
    """
    ath     = branch_data["all_time_high"]
    monthly = branch_data["monthly_sales"]
    sales   = [m["total_scaled"] for m in monthly]
    branch  = branch_data["branch"]
    n       = len(sales)

    # Closure check
    if n >= 2:
        last_util = sales[-1] / ath * 100 if ath else 0
        mom       = (sales[-1] - sales[-2]) / sales[-2] * 100 if sales[-2] > 0 else 0
        x         = np.arange(n, dtype=float)
        slope, _  = np.polyfit(x, np.array(sales, dtype=float), 1)
        if (last_util < CLOSURE_MAX_LAST_UTIL * 100
                and mom < CLOSURE_MAX_MOM_CHANGE * 100
                and slope < 0):
            return 0  # CONSIDER CLOSURE

    # Pillar 1
    above_80  = [s / ath >= P1_SATURATION_THRESHOLD for s in sales] if ath else [False] * n
    above_60  = [s / ath >= P1_AMBER_THRESHOLD      for s in sales] if ath else [False] * n
    streak_80 = _longest_streak(above_80)
    streak_60 = _longest_streak(above_60)
    if streak_80 >= P1_MIN_CONSECUTIVE:
        p1 = "green"
    elif streak_60 >= P1_MIN_CONSECUTIVE or streak_80 == 1:
        p1 = "amber"
    else:
        p1 = "red"

    # Pillar 2
    if n < P2_MIN_MONTHS:
        p2 = "red"
    else:
        x         = np.arange(n, dtype=float)
        sl        = np.array(sales, dtype=float)
        slope, ic = np.polyfit(x, sl, 1)
        yh        = slope * x + ic
        ss_res    = float(np.sum((sl - yh) ** 2))
        ss_tot    = float(np.sum((sl - sl.mean()) ** 2))
        r2        = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        if slope <= 0:
            p2 = "red"
        elif r2 >= P2_R2_MIN_GREEN:
            p2 = "green"
        else:
            p2 = "amber"

    # Pillar 3
    bc  = [c for c in customers if c["branch"] == branch]
    nc  = len(bc)
    rep = sum(1 for c in bc if (c["num_orders"] or 0) > 1)
    rr  = rep / nc if nc else 0.0
    if nc >= P3_MIN_CUSTOMERS_GREEN and rr >= P3_REPEAT_RATE_GREEN:
        p3 = "green"
    elif nc >= P3_MIN_CUSTOMERS_AMBER and rr >= P3_REPEAT_RATE_AMBER:
        p3 = "amber"
    elif nc >= P3_MIN_CUSTOMERS_GREEN:
        p3 = "amber"
    else:
        p3 = "red"

    scores = [p1, p2, p3]
    greens = scores.count("green")
    reds   = scores.count("red")
    if greens == 3:
        return 2
    if greens == 2 and reds == 0:
        return 1
    if reds == 0:
        return 1
    return 0


# ── Dataset Builder ────────────────────────────────────────────────────────────
def build_dataset(sales_json, customers_json):
    with open(sales_json, encoding="utf-8") as f:
        sales_data = json.load(f)
    with open(customers_json, encoding="utf-8") as f:
        customers = json.load(f)

    X, y, branch_names = [], [], []
    for entry in sales_data:
        feats = extract_features(entry, customers)
        label = _compute_label(entry, customers)
        X.append(list(feats.values()))
        y.append(label)
        branch_names.append(entry["branch"])

    feature_names = list(extract_features(sales_data[0], customers).keys())
    return (
        np.array(X, dtype=float),
        np.array(y, dtype=int),
        branch_names,
        feature_names,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_pipeline(pipeline, X, y):
    """
    Leave-One-Out cross-validation for small datasets (n < 20).
    Falls back to in-sample metrics when n < 3.
    Returns (accuracy, weighted_f1, weighted_precision, weighted_recall, y_pred).
    """
    n = len(X)
    if n >= 3:
        loo   = LeaveOneOut()
        preds = []
        for train_idx, test_idx in loo.split(X):
            cloned = sklearn_clone(pipeline)
            cloned.fit(X[train_idx], y[train_idx])
            preds.append(int(cloned.predict(X[test_idx])[0]))
        y_pred = np.array(preds)
    else:
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

    acc  = float(accuracy_score(y, y_pred))
    f1   = float(f1_score(y, y_pred, average="weighted", zero_division=0))
    prec = float(precision_score(y, y_pred, average="weighted", zero_division=0))
    rec  = float(recall_score(y, y_pred, average="weighted", zero_division=0))
    return acc, f1, prec, rec, y_pred


# ── MLflow Training ────────────────────────────────────────────────────────────
def train_with_mlflow(X, y, branch_names, feature_names, dry_run=False):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    present_labels = sorted(set(y.tolist()))
    label_dist     = "  ".join(f"{LABEL_NAMES[l]}={int(np.sum(y == l))}" for l in present_labels)

    print(f"\n  MLflow experiment : '{MLFLOW_EXPERIMENT}'")
    print(f"  Branches          : {len(X)}  ({', '.join(branch_names)})")
    print(f"  Features          : {X.shape[1]}")
    print(f"  Label distribution: {label_dist}")
    print(f"  Evaluation        : Leave-One-Out cross-validation")
    print(f"  Runs planned      : {sum(len(m['param_grid']) for m in MODEL_GRID)}")
    print()

    best = {
        "run_id":     None,
        "f1":         -1.0,
        "model_name": None,
        "params":     None,
        "pipeline":   None,
        "acc":        0.0,
    }

    for model_def in MODEL_GRID:
        model_name    = model_def["name"]
        model_cls     = model_def["model_cls"]
        needs_scaling = model_def["needs_scaling"]

        for params in model_def["param_grid"]:
            param_str = "_".join(
                f"{k}={v}" for k, v in params.items() if k not in ("random_state", "probability")
            )
            run_name = f"{model_name}_{param_str}"

            if dry_run:
                print(f"  [dry-run] Would train: {run_name}")
                continue

            # Build pipeline
            steps = []
            if needs_scaling:
                steps.append(("scaler", StandardScaler()))
            steps.append(("clf", model_cls(**params)))
            pipe = Pipeline(steps)

            # Evaluate with LOO
            acc, f1, prec, rec, y_pred = evaluate_pipeline(pipe, X, y)

            # Fit final model on ALL data for artifact logging
            pipe.fit(X, y)

            with mlflow.start_run(run_name=run_name):
                # ── Parameters ──────────────────────────────────────────────
                mlflow.log_param("model_type",    model_name)
                mlflow.log_param("needs_scaling", needs_scaling)
                mlflow.log_param("n_branches",    int(len(X)))
                mlflow.log_param("n_features",    int(X.shape[1]))
                mlflow.log_param("eval_strategy", "LOO" if len(X) >= 3 else "in_sample")
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # ── Metrics ─────────────────────────────────────────────────
                mlflow.log_metric("accuracy",           round(acc, 4))
                mlflow.log_metric("weighted_f1",        round(f1,  4))
                mlflow.log_metric("weighted_precision", round(prec, 4))
                mlflow.log_metric("weighted_recall",    round(rec, 4))

                # Per-label correct-count
                for lid, lname in LABEL_NAMES.items():
                    if lid in y:
                        tp = int(np.sum((y == lid) & (y_pred == lid)))
                        mlflow.log_metric(f"correct_{lname}", tp)

                # Per-branch prediction
                for bname, true_l, pred_l in zip(branch_names, y, y_pred):
                    mlflow.log_param(
                        f"branch_{bname.replace(' ', '_')}",
                        f"true={LABEL_NAMES[int(true_l)]}_pred={LABEL_NAMES[int(pred_l)]}",
                    )

                # Feature importances (tree-based models)
                clf = pipe.named_steps["clf"]
                if hasattr(clf, "feature_importances_"):
                    for fname, imp in zip(feature_names, clf.feature_importances_):
                        mlflow.log_metric(f"feat_imp_{fname}", round(float(imp), 6))

                # Coefficient magnitudes (linear models)
                if hasattr(clf, "coef_"):
                    coefs = np.abs(clf.coef_).mean(axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
                    for fname, cval in zip(feature_names, coefs):
                        mlflow.log_metric(f"coef_abs_{fname}", round(float(cval), 6))

                # Model artifact
                mlflow.sklearn.log_model(
                    pipe,
                    artifact_path="model",
                    registered_model_name=f"expansion_{model_name}",
                )

                run_id = mlflow.active_run().info.run_id
                flag   = "★ BEST" if f1 > best["f1"] else "      "
                print(
                    f"  {flag}  [{model_name:20s}]  "
                    f"acc={acc:.3f}  f1={f1:.3f}  "
                    f"prec={prec:.3f}  rec={rec:.3f}  "
                    f"run={run_id[:8]}"
                )

                if f1 > best["f1"]:
                    best.update({
                        "run_id":     run_id,
                        "f1":         f1,
                        "acc":        acc,
                        "model_name": model_name,
                        "params":     params,
                        "pipeline":   pipe,
                    })

    if dry_run or best["run_id"] is None:
        return best

    # ── Tag best run ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("best_model",   "true")
        mlflow.set_tag("selected_at",  datetime.now().isoformat())
        mlflow.set_tag("selection_metric", "weighted_f1")

    print(
        f"\n  Best model: [{best['model_name']}]  "
        f"f1={best['f1']:.3f}  acc={best['acc']:.3f}  "
        f"run_id={best['run_id'][:8]}"
    )

    # ── Persist best-model metadata locally ───────────────────────────────────
    model_meta = {
        "run_id":        best["run_id"],
        "model_name":    best["model_name"],
        "params":        {k: str(v) for k, v in best["params"].items()},
        "weighted_f1":   round(best["f1"], 4),
        "accuracy":      round(best["acc"], 4),
        "trained_on":    datetime.now().isoformat(),
        "n_branches":    int(len(X)),
        "feature_names": feature_names,
        "label_names":   LABEL_NAMES,
    }
    with open(BEST_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)
    print(f"  Metadata saved   : {BEST_MODEL_PATH}")

    return best


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Expansion data refresh + MLflow model training"
    )
    parser.add_argument("--sales-csv",      default=DEFAULT_SALES_CSV)
    parser.add_argument("--customers-csv",  default=DEFAULT_CUSTOMERS_CSV)
    parser.add_argument("--sales-json",     default=DEFAULT_SALES_JSON)
    parser.add_argument("--customers-json", default=DEFAULT_CUSTOMERS_JSON)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without writing files or logging to MLflow",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip CSV ingestion; retrain directly from existing JSON",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  EXPANSION PIPELINE  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Phase 1: Data Refresh ──────────────────────────────────────────────────
    if not args.skip_refresh:
        print(f"\n[1/2] Ingesting sales from:\n      {args.sales_csv}")
        new_sales = ingest_sales_csv(args.sales_csv)
        for b in new_sales:
            print(f"      {b['branch']}: {len(b['monthly_sales'])} months, ATH={b['all_time_high']:,.0f}")

        print(f"\n[2/2] Ingesting customers from:\n      {args.customers_csv}")
        new_customers = ingest_customers_csv(args.customers_csv)
        by_branch = {}
        for c in new_customers:
            by_branch[c["branch"]] = by_branch.get(c["branch"], 0) + 1
        for b, n in by_branch.items():
            print(f"      {b}: {n} customers")

        write_json(new_sales,     args.sales_json,     args.dry_run)
        write_json(new_customers, args.customers_json, args.dry_run)
    else:
        print("\n[Skipping CSV refresh — using existing JSON data]")

    # ── Phase 2: MLflow Model Training ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL TRAINING  (MLflow experiment: expansion)")
    print("=" * 60)

    X, y, branch_names, feature_names = build_dataset(
        args.sales_json, args.customers_json
    )
    train_with_mlflow(X, y, branch_names, feature_names, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

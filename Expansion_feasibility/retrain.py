"""
retrain.py
----------
Data refresh pipeline for the Conut Expansion Decision System.

IMPORTANT — WHAT THIS SYSTEM IS:
    The expansion scoring system (expansion_score.py) and location optimizer
    (expansion_feasibility.py) are NOT neural networks or learned models.
    They contain:
        - Hand-written business rules with configurable thresholds
        - np.polyfit  → deterministic least-squares regression (no training)
        - KMeans k=1  → mathematically equivalent to computing a mean (no training)

    "Retraining" for this system means:
        1. Ingest new monthly sales data
        2. Ingest new customer records (with addresses when available)
        3. Rebuild the cleaned JSON files
        4. Re-run the scoring — thresholds recalibrate automatically

    This script automates that entire cycle end-to-end.

    The ONE component that IS learnable (and could be upgraded to a real ML model
    in a future version) is the threshold calibration. This script includes a
    threshold optimizer that tunes P1/P2/P3 thresholds using historical branch
    outcome labels if you have them.

Usage:
    # Full refresh with new CSV files:
    python retrain.py --sales-csv rep_s_00334_1_SMRY.csv --customers-csv rep_s_00150.csv

    # Refresh + re-optimize thresholds (requires outcome labels):
    python retrain.py --sales-csv rep_s_00334_1_SMRY.csv --customers-csv rep_s_00150.csv
                      --outcomes outcomes.json --optimize-thresholds

    # Dry run — just show what would change without writing files:
    python retrain.py --sales-csv rep_s_00334_1_SMRY.csv --dry-run
"""

import json
import csv
import io
import argparse
import copy
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SALES_CSV      = "rep_s_00334_1_SMRY.csv"
DEFAULT_CUSTOMERS_CSV  = "rep_s_00150.csv"
DEFAULT_SALES_JSON     = "monthly_sales_by_branch.json"
DEFAULT_CUSTOMERS_JSON = "customers.json"
DEFAULT_CONFIG_JSON    = "expansion_config.json"

# Default thresholds — written to expansion_config.json on first run
DEFAULT_CONFIG = {
    "last_updated":          None,
    "data_months_available": 0,
    "thresholds": {
        # Pillar 1 — Volume
        "P1_SATURATION_THRESHOLD": 0.80,
        "P1_AMBER_THRESHOLD":      0.60,
        "P1_MIN_CONSECUTIVE":      2,
        # Pillar 2 — Trajectory
        "P2_R2_MIN_GREEN":         0.70,
        "P2_R2_MIN_AMBER":         0.40,
        "P2_MIN_MONTHS":           3,
        # Pillar 3 — Delivery density
        "P3_MIN_CUSTOMERS_GREEN":  150,
        "P3_MIN_CUSTOMERS_AMBER":  50,
        "P3_REPEAT_RATE_GREEN":    0.15,
        "P3_REPEAT_RATE_AMBER":    0.05,
        # Closure check
        "CLOSURE_MAX_LAST_UTIL":   0.20,
        "CLOSURE_MAX_MOM_CHANGE":  -0.50,
    }
}

MONTHS = [
    "January", "February", "March",     "April",    "May",      "June",
    "July",    "August",   "September", "October",  "November", "December",
]

MONTH_ORDER = {m: i for i, m in enumerate(MONTHS)}

BRANCHES_CSV = ["Conut - Tyre", "Conut Jnah", "Conut", "Main Street Coffee"]

# Section boundaries in rep_s_00150.csv (line-based, from inspection)
CUSTOMER_SECTION_RANGES = [
    (0,   99,  "Conut - Tyre"),
    (99,  302, "Conut"),
    (302, 563, "Conut Jnah"),
    (563, None, "Main Street Coffee"),
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INGEST & CLEAN SALES CSV  →  monthly_sales_by_branch.json
# ─────────────────────────────────────────────────────────────────────────────

def ingest_sales_csv(filepath: str) -> list:
    """
    Parses rep_s_00334_1_SMRY.csv into a clean list of branch dicts.

    Handles:
      - Repeated page headers / footers
      - Quoted comma-formatted numbers ("1,137,352,241.41")
      - Branch name markers ("Branch Name: Conut Jnah")

    Returns list of dicts matching monthly_sales_by_branch.json schema.
    """
    def to_float(s):
        try:
            return float(str(s).strip().replace(",", "").replace('"', ""))
        except (ValueError, TypeError):
            return None

    records = {}   # branch → list of monthly rows
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
                # Avoid duplicate months (repeated page headers)
                existing_months = [r["month"] for r in records[cur_branch]]
                if c0 not in existing_months and total is not None:
                    records[cur_branch].append({
                        "month":        c0,
                        "year":         year,
                        "total_scaled": total,
                    })

    # Sort each branch's months chronologically
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
                "month":        last_row["month"],
                "year":         last_row["year"],
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — INGEST & CLEAN CUSTOMERS CSV  →  customers.json
# ─────────────────────────────────────────────────────────────────────────────

def ingest_customers_csv(filepath: str) -> list:
    """
    Parses rep_s_00150.csv into a clean list of customer dicts.
    Handles the two column layouts (10-col and 11-col) across branches.
    """
    def to_float(s):
        try:
            return float(str(s).strip().replace(",", ""))
        except (ValueError, TypeError):
            return None

    def parse_row(row, branch):
        if not row or not row[0].startswith("Person_"):
            return None
        name  = row[0].strip()
        addr  = row[1].strip() if len(row) > 1 else ""
        phone = row[2].strip() if len(row) > 2 else ""
        c3    = row[3].strip() if len(row) > 3 else ""

        if c3.startswith("20"):    # 10-col layout
            fo    = f"{row[3].strip()} {row[4].strip()}".rstrip(":").strip() if len(row) > 4 else ""
            lo    = f"{row[5].strip()} {row[6].strip()}".rstrip(":").strip() if len(row) > 6 else ""
            total = to_float(row[7]) if len(row) > 7 else None
            n_ord = to_float(row[8]) if len(row) > 8 else None
        else:                      # 11-col layout
            fo    = f"{row[4].strip()} {row[5].strip()}".rstrip(":").strip() if len(row) > 5 else ""
            lo    = f"{row[6].strip()} {row[7].strip()}".rstrip(":").strip() if len(row) > 7 else ""
            total = to_float(row[8]) if len(row) > 8 else None
            n_ord = to_float(row[9]) if len(row) > 9 else None

        orders = int(n_ord) if n_ord is not None and n_ord < 10_000 else None

        return {
            "branch":             branch,
            "customer_name":      name,
            "address":            addr if addr else None,
            "phone":              phone if phone else None,
            "first_order":        fo or None,
            "last_order":         lo or None,
            "total_sales_scaled": total,
            "num_orders":         orders,
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DIFF: what changed since the last run?
# ─────────────────────────────────────────────────────────────────────────────

def compute_diff(old_sales: list, new_sales: list) -> dict:
    """
    Compares old and new sales data and reports what changed.
    Useful for logging what each refresh cycle brought in.

    Returns a dict summarising:
      - new branches added
      - branches with new months appended
      - branches where existing month values changed (data correction)
      - branches removed
    """
    old_map = {b["branch"]: b for b in old_sales}
    new_map = {b["branch"]: b for b in new_sales}

    added    = [b for b in new_map if b not in old_map]
    removed  = [b for b in old_map if b not in new_map]
    modified = []

    for branch in new_map:
        if branch not in old_map:
            continue
        old_months = {m["month"]: m["total_scaled"] for m in old_map[branch]["monthly_sales"]}
        new_months = {m["month"]: m["total_scaled"] for m in new_map[branch]["monthly_sales"]}

        new_month_names  = [m for m in new_months if m not in old_months]
        changed_values   = [
            m for m in old_months
            if m in new_months and abs(new_months[m] - old_months[m]) > 1.0
        ]

        if new_month_names or changed_values:
            modified.append({
                "branch":         branch,
                "new_months":     new_month_names,
                "corrected_months": changed_values,
            })

    return {
        "branches_added":   added,
        "branches_removed": removed,
        "branches_modified": modified,
        "total_changes":    len(added) + len(removed) + len(modified),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — THRESHOLD OPTIMIZER (optional, requires outcome labels)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_thresholds(sales_data: list, customers: list, outcomes_path: str) -> dict:
    """
    Tunes the scoring thresholds by grid-searching over candidate values and
    finding the combination that best matches known historical outcomes.

    This is the ONE component that constitutes genuine "learning from data."

    Requires: outcomes.json — a list of labelled branch decisions:
        [
            {
                "branch":   "Conut Jnah",
                "month":    "December",
                "year":     2025,
                "label":    "expand"       // "expand" | "monitor" | "close" | "ok"
            },
            ...
        ]

    Without outcome labels, thresholds cannot be tuned from data — they
    remain as expert-defined constants (the current default).

    Returns the best-found threshold config dict.
    """
    with open(outcomes_path, encoding="utf-8") as f:
        outcomes = json.load(f)

    if not outcomes:
        print("  [optimizer] No outcome labels found — skipping threshold optimization.")
        return DEFAULT_CONFIG["thresholds"]

    print(f"  [optimizer] {len(outcomes)} labelled outcomes loaded.")

    # Import scoring functions dynamically to avoid circular dependency
    try:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "expansion_score",
            Path(__file__).parent / "expansion_score.py"
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  [optimizer] Could not load expansion_score.py: {e}")
        return DEFAULT_CONFIG["thresholds"]

    # Grid of candidate threshold values to search
    grid = {
        "P1_SATURATION_THRESHOLD": [0.70, 0.75, 0.80, 0.85],
        "P1_AMBER_THRESHOLD":      [0.50, 0.60, 0.65],
        "P2_R2_MIN_GREEN":         [0.60, 0.65, 0.70, 0.75],
        "P2_R2_MIN_AMBER":         [0.30, 0.40, 0.50],
        "CLOSURE_MAX_LAST_UTIL":   [0.15, 0.20, 0.25],
        "CLOSURE_MAX_MOM_CHANGE":  [-0.40, -0.50, -0.60],
    }

    best_accuracy = -1.0
    best_config   = copy.deepcopy(DEFAULT_CONFIG["thresholds"])

    # Build outcome lookup: branch → expected verdict
    LABEL_TO_VERDICT = {
        "expand":  "EXPAND",
        "monitor": "MONITOR",
        "close":   "CONSIDER CLOSURE",
        "ok":      "DO NOT EXPAND",
    }
    expected = {
        o["branch"]: LABEL_TO_VERDICT.get(o["label"], "DO NOT EXPAND")
        for o in outcomes
    }

    sales_map = {b["branch"]: b for b in sales_data}

    # Grid search
    keys   = list(grid.keys())
    values = list(grid.values())
    total_combos = 1
    for v in values:
        total_combos *= len(v)
    print(f"  [optimizer] Searching {total_combos:,} threshold combinations...")

    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))

        # Apply thresholds to scoring module
        mod.P1_SATURATION_THRESHOLD = cfg["P1_SATURATION_THRESHOLD"]
        mod.P1_AMBER_THRESHOLD      = cfg["P1_AMBER_THRESHOLD"]
        mod.P2_R2_MIN_GREEN         = cfg["P2_R2_MIN_GREEN"]
        mod.P2_R2_MIN_AMBER         = cfg["P2_R2_MIN_AMBER"]
        mod.CLOSURE_MAX_LAST_UTIL   = cfg["CLOSURE_MAX_LAST_UTIL"]
        mod.CLOSURE_MAX_MOM_CHANGE  = cfg["CLOSURE_MAX_MOM_CHANGE"]

        correct = 0
        for branch, exp_verdict in expected.items():
            if branch not in sales_map:
                continue
            try:
                closure = mod.check_closure(sales_map[branch])
                if closure["should_close"]:
                    got_verdict = "CONSIDER CLOSURE"
                else:
                    p1 = mod.score_pillar_1(sales_map[branch])
                    p2 = mod.score_pillar_2(sales_map[branch])
                    p3 = mod.score_pillar_3(branch, customers)
                    got_verdict, _ = mod.compute_verdict(p1, p2, p3)
                if got_verdict == exp_verdict:
                    correct += 1
            except Exception:
                pass

        accuracy = correct / len(expected) if expected else 0.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config   = copy.deepcopy(DEFAULT_CONFIG["thresholds"])
            best_config.update(cfg)

    print(f"  [optimizer] Best accuracy: {best_accuracy*100:.1f}% "
          f"({int(best_accuracy*len(expected))}/{len(expected)} correct)")

    return best_config


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — WRITE OUTPUTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def write_json(data, path: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"  [dry-run] Would write {len(data)} records to {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅  Written: {path}")


def write_config(config: dict, path: str, dry_run: bool = False) -> None:
    config["last_updated"] = datetime.now().isoformat()
    if dry_run:
        print(f"  [dry-run] Would write config to {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  ✅  Config written: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — RUN SCORING AFTER REFRESH
# ─────────────────────────────────────────────────────────────────────────────

def run_scoring(sales_json: str, customers_json: str) -> None:
    """Runs the full branch scoring after refresh and prints results."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "expansion_score",
            Path(__file__).parent / "expansion_score.py"
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.score_all_branches(sales_json, customers_json)
    except FileNotFoundError:
        print("  [scoring] expansion_score.py not found — skipping auto-score.")
    except Exception as e:
        print(f"  [scoring] Error during scoring: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Conut Expansion System — Data Refresh Pipeline"
    )
    parser.add_argument("--sales-csv",      default=DEFAULT_SALES_CSV,
                        help="Path to rep_s_00334_1_SMRY.csv")
    parser.add_argument("--customers-csv",  default=DEFAULT_CUSTOMERS_CSV,
                        help="Path to rep_s_00150.csv")
    parser.add_argument("--sales-json",     default=DEFAULT_SALES_JSON,
                        help="Output path for monthly_sales_by_branch.json")
    parser.add_argument("--customers-json", default=DEFAULT_CUSTOMERS_JSON,
                        help="Output path for customers.json")
    parser.add_argument("--config",         default=DEFAULT_CONFIG_JSON,
                        help="Path to expansion_config.json")
    parser.add_argument("--outcomes",       default=None,
                        help="Path to outcomes.json for threshold optimization")
    parser.add_argument("--optimize-thresholds", action="store_true",
                        help="Run threshold optimizer (requires --outcomes)")
    parser.add_argument("--no-score",       action="store_true",
                        help="Skip running scoring after refresh")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Show what would change without writing files")
    args = parser.parse_args()

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  CONUT EXPANSION SYSTEM — DATA REFRESH")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{SEP}\n")

    # ── Load existing data (for diff) ─────────────────────────────────────────
    old_sales = []
    if Path(args.sales_json).exists():
        with open(args.sales_json, encoding="utf-8") as f:
            old_sales = json.load(f)
        print(f"  Existing sales data loaded: {len(old_sales)} branches")
    else:
        print("  No existing sales data found — this is a first run.")

    # ── Step 1: Ingest sales CSV ──────────────────────────────────────────────
    print(f"\n  [1/5] Ingesting sales data from {args.sales_csv} ...")
    new_sales = ingest_sales_csv(args.sales_csv)
    print(f"        {len(new_sales)} branches parsed")
    for b in new_sales:
        n = len(b["monthly_sales"])
        print(f"        • {b['branch']}: {n} months  "
              f"(ATH={b['all_time_high']:,.0f} in {b['ath_month']})")

    # ── Step 2: Ingest customers CSV ──────────────────────────────────────────
    print(f"\n  [2/5] Ingesting customer data from {args.customers_csv} ...")
    new_customers = ingest_customers_csv(args.customers_csv)
    by_branch = {}
    for c in new_customers:
        by_branch[c["branch"]] = by_branch.get(c["branch"], 0) + 1
    print(f"        {len(new_customers)} customers parsed")
    for b, n in by_branch.items():
        addr_count = sum(1 for c in new_customers if c["branch"] == b and c["address"])
        print(f"        • {b}: {n} customers  ({addr_count} with address)")

    # ── Step 3: Diff ──────────────────────────────────────────────────────────
    print(f"\n  [3/5] Computing diff against previous data ...")
    diff = compute_diff(old_sales, new_sales)
    if diff["total_changes"] == 0:
        print("        No changes detected — data is identical to previous run.")
    else:
        if diff["branches_added"]:
            print(f"        ➕ New branches    : {', '.join(diff['branches_added'])}")
        if diff["branches_removed"]:
            print(f"        ➖ Removed branches: {', '.join(diff['branches_removed'])}")
        for m in diff["branches_modified"]:
            if m["new_months"]:
                print(f"        📅 {m['branch']}: {len(m['new_months'])} new month(s) "
                      f"— {', '.join(m['new_months'])}")
            if m["corrected_months"]:
                print(f"        ✏️  {m['branch']}: {len(m['corrected_months'])} value correction(s) "
                      f"— {', '.join(m['corrected_months'])}")

    # ── Step 4: Threshold optimization (optional) ─────────────────────────────
    print(f"\n  [4/5] Threshold configuration ...")
    config = copy.deepcopy(DEFAULT_CONFIG)

    if args.optimize_thresholds and args.outcomes:
        print(f"        Running threshold optimizer with labels from {args.outcomes} ...")
        optimized = optimize_thresholds(new_sales, new_customers, args.outcomes)
        config["thresholds"] = optimized
        print(f"        Optimized thresholds:")
        for k, v in optimized.items():
            default_v = DEFAULT_CONFIG["thresholds"].get(k)
            changed   = " ← changed" if v != default_v else ""
            print(f"          {k:<35}: {v}{changed}")
    else:
        print("        Using default expert-defined thresholds.")
        print("        (Pass --optimize-thresholds --outcomes outcomes.json to tune from data)")

    config["data_months_available"] = max(
        (len(b["monthly_sales"]) for b in new_sales), default=0
    )

    # ── Step 5: Write outputs ─────────────────────────────────────────────────
    print(f"\n  [5/5] Writing outputs ...")
    write_json(new_sales,     args.sales_json,     args.dry_run)
    write_json(new_customers, args.customers_json, args.dry_run)
    write_config(config,      args.config,         args.dry_run)

    # ── Run scoring ───────────────────────────────────────────────────────────
    if not args.no_score and not args.dry_run:
        print(f"\n  Running expansion scoring on refreshed data ...\n")
        run_scoring(args.sales_json, args.customers_json)

    print(f"\n{SEP}")
    print("  REFRESH COMPLETE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
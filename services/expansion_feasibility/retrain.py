"""
update_pipeline.py — Expansion Feasibility Data Refresh
Run this when new monthly CSV data arrives.

Usage:
    python update_pipeline.py
    python update_pipeline.py --dry-run
"""
import json, csv, argparse, copy, os
from pathlib import Path
from datetime import datetime
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data"))

DEFAULT_SALES_CSV      = os.path.join(DATA_DIR, "rep_s_00334_1_SMRY.csv")
DEFAULT_CUSTOMERS_CSV  = os.path.join(DATA_DIR, "rep_s_00150.csv")
DEFAULT_SALES_JSON     = os.path.join(BASE_DIR, "monthly_sales_by_branch.json")
DEFAULT_CUSTOMERS_JSON = os.path.join(BASE_DIR, "customers.json")

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
MONTH_ORDER = {m: i for i, m in enumerate(MONTHS)}
CUSTOMER_SECTION_RANGES = [
    (0,   99,  "Conut - Tyre"),
    (99,  302, "Conut"),
    (302, 563, "Conut Jnah"),
    (563, None, "Main Street Coffee"),
]

def to_float(s):
    try: return float(str(s).strip().replace(",","").replace('"',""))
    except: return None

def ingest_sales_csv(filepath):
    records = {}
    cur_branch = None
    with open(filepath, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row: continue
            c0 = row[0].strip()
            if c0.startswith("Branch Name:"):
                cur_branch = c0.replace("Branch Name:","").strip()
                if cur_branch not in records: records[cur_branch] = []
                continue
            if cur_branch and c0 in MONTHS:
                year  = int(row[2].strip()) if len(row)>2 and row[2].strip().isdigit() else None
                total = to_float(row[3]) if len(row)>3 else None
                existing = [r["month"] for r in records[cur_branch]]
                if c0 not in existing and total is not None:
                    records[cur_branch].append({"month": c0, "year": year, "total_scaled": total})
    output = []
    for branch, monthly in records.items():
        monthly_sorted = sorted(monthly, key=lambda m: MONTH_ORDER.get(m["month"],99))
        if not monthly_sorted: continue
        ath_row  = max(monthly_sorted, key=lambda m: m["total_scaled"])
        last_row = monthly_sorted[-1]
        output.append({
            "branch":        branch,
            "all_time_high": round(ath_row["total_scaled"],2),
            "ath_month":     ath_row["month"],
            "current_month": {"month": last_row["month"], "year": last_row["year"],
                              "total_scaled": round(last_row["total_scaled"],2)},
            "monthly_sales": [{"month": r["month"], "year": int(r["year"]) if r["year"] else None,
                               "total_scaled": round(r["total_scaled"],2)} for r in monthly_sorted],
        })
    return output

def ingest_customers_csv(filepath):
    def parse_row(row, branch):
        if not row or not row[0].startswith("Person_"): return None
        name=row[0].strip(); addr=row[1].strip() if len(row)>1 else ""; phone=row[2].strip() if len(row)>2 else ""
        c3=row[3].strip() if len(row)>3 else ""
        if c3.startswith("20"):
            fo=f"{row[3].strip()} {row[4].strip()}".rstrip(":").strip() if len(row)>4 else ""
            lo=f"{row[5].strip()} {row[6].strip()}".rstrip(":").strip() if len(row)>6 else ""
            total=to_float(row[7]) if len(row)>7 else None; n_ord=to_float(row[8]) if len(row)>8 else None
        else:
            fo=f"{row[4].strip()} {row[5].strip()}".rstrip(":").strip() if len(row)>5 else ""
            lo=f"{row[6].strip()} {row[7].strip()}".rstrip(":").strip() if len(row)>7 else ""
            total=to_float(row[8]) if len(row)>8 else None; n_ord=to_float(row[9]) if len(row)>9 else None
        orders=int(n_ord) if n_ord is not None and n_ord<10_000 else None
        return {"branch":branch,"customer_name":name,"address":addr or None,"phone":phone or None,
                "first_order":fo or None,"last_order":lo or None,"total_sales_scaled":total,"num_orders":orders}
    with open(filepath, encoding="utf-8-sig") as f:
        all_lines = f.readlines()
    customers = []
    for start, end, branch in CUSTOMER_SECTION_RANGES:
        for line in all_lines[start:end]:
            row    = next(csv.reader([line]))
            parsed = parse_row(row, branch)
            if parsed: customers.append(parsed)
    return customers

def write_json(data, path, dry_run=False):
    if dry_run: print(f"  [dry-run] Would write to {path}"); return
    with open(path,"w",encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅  Written: {path}")

def main():
    parser = argparse.ArgumentParser(description="Expansion data refresh pipeline")
    parser.add_argument("--sales-csv",      default=DEFAULT_SALES_CSV)
    parser.add_argument("--customers-csv",  default=DEFAULT_CUSTOMERS_CSV)
    parser.add_argument("--sales-json",     default=DEFAULT_SALES_JSON)
    parser.add_argument("--customers-json", default=DEFAULT_CUSTOMERS_JSON)
    parser.add_argument("--dry-run",        action="store_true")
    args = parser.parse_args()

    print("="*60)
    print(f"  EXPANSION DATA REFRESH  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print(f"\n[1/2] Ingesting sales from {args.sales_csv}...")
    new_sales = ingest_sales_csv(args.sales_csv)
    for b in new_sales:
        print(f"      {b['branch']}: {len(b['monthly_sales'])} months, ATH={b['all_time_high']:,.0f}")

    print(f"\n[2/2] Ingesting customers from {args.customers_csv}...")
    new_customers = ingest_customers_csv(args.customers_csv)
    by_branch = {}
    for c in new_customers: by_branch[c["branch"]] = by_branch.get(c["branch"],0)+1
    for b,n in by_branch.items(): print(f"      {b}: {n} customers")

    write_json(new_sales,     args.sales_json,     args.dry_run)
    write_json(new_customers, args.customers_json, args.dry_run)

    print("\n" + "="*60)
    print("  REFRESH COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
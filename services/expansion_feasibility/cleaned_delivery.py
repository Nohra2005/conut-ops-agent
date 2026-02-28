import csv, json, argparse
from pathlib import Path
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data"))

SECTION_RANGES = [
    (0,   99,  "Conut - Tyre"),
    (99,  302, "Conut"),
    (302, 563, "Conut Jnah"),
    (563, None, "Main Street Coffee"),
]

def to_float(s):
    try: return float(str(s).strip().replace(",", ""))
    except: return None

def parse_row(row, branch):
    if not row or not row[0].startswith("Person_"): return None
    name = row[0].strip()
    addr = row[1].strip() if len(row) > 1 else ""
    phone = row[2].strip() if len(row) > 2 else ""
    c3 = row[3].strip() if len(row) > 3 else ""
    if c3.startswith("20"):
        fo    = f"{row[3].strip()} {row[4].strip()}".rstrip(":").strip() if len(row) > 4 else ""
        lo    = f"{row[5].strip()} {row[6].strip()}".rstrip(":").strip() if len(row) > 6 else ""
        total = to_float(row[7]) if len(row) > 7 else None
        n_ord = row[8].strip() if len(row) > 8 else ""
    else:
        fo    = f"{row[4].strip()} {row[5].strip()}".rstrip(":").strip() if len(row) > 5 else ""
        lo    = f"{row[6].strip()} {row[7].strip()}".rstrip(":").strip() if len(row) > 7 else ""
        total = to_float(row[8]) if len(row) > 8 else None
        n_ord = row[9].strip() if len(row) > 9 else ""
    orders_f = to_float(n_ord)
    orders = int(orders_f) if (orders_f is not None and orders_f < 10_000) else None
    return {"branch": branch, "customer_name": name, "address": addr or None,
            "phone": phone or None, "first_order": fo or None, "last_order": lo or None,
            "total_sales_scaled": total, "num_orders": orders}

def clean(input_path):
    with open(input_path, encoding="utf-8-sig") as f:
        all_lines = f.readlines()
    customers = []
    for start, end, branch in SECTION_RANGES:
        for line in all_lines[start:end]:
            row = next(csv.reader([line]))
            record = parse_row(row, branch)
            if record: customers.append(record)
    return customers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=os.path.join(DATA_DIR, "rep_s_00150.csv"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "customers.json"))
    args = parser.parse_args()
    customers = clean(Path(args.input))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(customers, f, indent=2, ensure_ascii=False)
    print(f"✅  {len(customers)} customers written to {args.output}")
    by_branch = {}
    for c in customers:
        by_branch[c["branch"]] = by_branch.get(c["branch"], 0) + 1
    for branch, count in by_branch.items():
        print(f"    {branch}: {count}")

if __name__ == "__main__":
    main()
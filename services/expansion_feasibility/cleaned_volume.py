import csv, json, argparse
from pathlib import Path
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data"))

MONTHS = {"January","February","March","April","May","June",
          "July","August","September","October","November","December"}
SKIP_PREFIXES = ("Branch Name:","Total","Grand Total","Month","REP_S",
                 "30-Jan","Conut - Tyre","Monthly Sales")

def _to_float(s):
    try: return float(s.strip().replace(",","").replace('"',""))
    except: return None

def parse_monthly_sales(filepath):
    records = []
    current_branch = None
    with open(filepath, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row: continue
            c0 = row[0].strip()
            if c0.startswith("Branch Name:"):
                current_branch = c0.replace("Branch Name:","").strip(); continue
            if any(c0.startswith(p) for p in SKIP_PREFIXES): continue
            if not c0 or c0 not in MONTHS: continue
            if current_branch is None: continue
            year  = int(row[2].strip()) if len(row) > 2 and row[2].strip().isdigit() else None
            total = _to_float(row[3]) if len(row) > 3 else None
            records.append({"branch": current_branch, "month": c0, "year": year, "total_scaled": total})
    return pd.DataFrame(records, columns=["branch","month","year","total_scaled"])

def enrich_with_saturation(df):
    branch_stats = df.groupby("branch")["total_scaled"].agg(all_time_high="max").reset_index()
    last_idx = df.groupby("branch").apply(lambda g: g.index[-1])
    current = (df.loc[last_idx.values, ["branch","total_scaled"]]
               .rename(columns={"total_scaled":"current_month"}).reset_index(drop=True))
    stats = branch_stats.merge(current, on="branch")
    stats["utilization_pct"] = ((stats["current_month"] / stats["all_time_high"]) * 100).round(2)
    stats["is_saturated"] = stats["utilization_pct"] >= 80.0
    return df.merge(stats, on="branch")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=os.path.join(DATA_DIR, "rep_s_00334_1_SMRY.csv"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "monthly_sales_by_branch.json"))
    args = parser.parse_args()
    df = parse_monthly_sales(Path(args.input))
    df = enrich_with_saturation(df)
    output = []
    for branch in df["branch"].unique():
        bdf = df[df["branch"] == branch]
        ath_row  = bdf.loc[bdf["total_scaled"].idxmax()]
        last_row = bdf.iloc[-1]
        output.append({
            "branch": branch,
            "all_time_high": round(float(ath_row["total_scaled"]),2),
            "ath_month": ath_row["month"],
            "current_month": {"month": last_row["month"], "year": int(last_row["year"]),
                              "total_scaled": round(float(last_row["total_scaled"]),2)},
            "monthly_sales": [{"month": r["month"], "year": int(r["year"]),
                               "total_scaled": round(float(r["total_scaled"]),2)} for _,r in bdf.iterrows()],
        })
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅  Saved {len(output)} branches to {args.output}")

if __name__ == "__main__":
    main()
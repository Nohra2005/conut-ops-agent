"""
clean_rep_s_00334.py
--------------------
Cleans rep_s_00334_1_SMRY.csv (Monthly Sales by Branch) from the Conut
Bakery scaled dataset.

The raw file is a report-style CSV with:
  - Page headers / footers mixed in with data
  - Branch names embedded as "Branch Name: X" marker rows
  - Quoted, comma-formatted numbers  (e.g. "1,137,352,241.41")
  - Only 5 months of data (Aug–Dec 2025) — the business opened mid-year

Output
------
  - Prints a clean summary table to the terminal
  - Saves  monthly_sales_by_branch.json  (one record per branch/month)

Usage
-----
    python clean_rep_s_00334.py
    python clean_rep_s_00334.py --input rep_s_00334_1_SMRY.csv --output monthly_sales.json
"""

import csv
import json
import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTHS = {
    "January", "February", "March",    "April",   "May",      "June",
    "July",    "August",   "September","October",  "November", "December",
}

# Rows whose first cell starts with these strings are metadata — skip them
SKIP_PREFIXES = (
    "Branch Name:",   # section header  → used to set current branch
    "Total",          # subtotal rows
    "Grand Total",    # grand total row
    "Month",          # column header row
    "REP_S",          # copyright footer
    "30-Jan",         # date stamp
    "Conut - Tyre",   # very first report header line
    "Monthly Sales",  # report title
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_monthly_sales(filepath: Path) -> pd.DataFrame:
    """
    Reads rep_s_00334_1_SMRY.csv and returns a tidy DataFrame with columns:

        branch       (str)   — branch name
        month        (str)   — e.g. "October"
        year         (int)   — e.g. 2025
        total_scaled (float) — raw scaled sales total for that month

    Rows that are page headers, subtotals, or footers are discarded.
    """
    records = []
    current_branch = None

    with open(filepath, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            cell0 = row[0].strip()

            # ── Detect branch section header ──────────────────────────────
            if cell0.startswith("Branch Name:"):
                current_branch = cell0.replace("Branch Name:", "").strip()
                continue

            # ── Skip metadata / junk rows ─────────────────────────────────
            if any(cell0.startswith(p) for p in SKIP_PREFIXES):
                continue
            if not cell0:          # skip rows where col 0 is blank
                continue

            # ── Accept only rows whose first cell is a month name ─────────
            if cell0 not in MONTHS:
                continue

            if current_branch is None:
                continue           # safety: data before any branch header

            # ── Parse the data row ────────────────────────────────────────
            month = cell0
            year  = int(row[2].strip()) if len(row) > 2 and row[2].strip().isdigit() else None
            total = _to_float(row[3])   if len(row) > 3 else None

            records.append({
                "branch":       current_branch,
                "month":        month,
                "year":         year,
                "total_scaled": total,
            })

    df = pd.DataFrame(records, columns=["branch", "month", "year", "total_scaled"])
    return df


def _to_float(s: str):
    """Strip quotes and commas, convert to float. Returns None on failure."""
    try:
        return float(s.strip().replace(",", "").replace('"', ""))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Enrichment — saturation metrics
# ---------------------------------------------------------------------------

def enrich_with_saturation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three columns to the DataFrame:

        all_time_high     — the highest monthly total ever recorded for that branch
        current_month     — the most recent month's total for that branch
        utilization_pct   — (current_month / all_time_high) * 100
        is_saturated      — True if utilization_pct >= 80

    These are the key inputs to the expansion feasibility tool.
    """
    branch_stats = (
        df.groupby("branch")["total_scaled"]
        .agg(all_time_high="max")
        .reset_index()
    )

    # "Current month" = last row per branch in file order (Dec 2025 in this dataset)
    last_idx = df.groupby("branch").apply(lambda g: g.index[-1])
    current = (
        df.loc[last_idx.values, ["branch", "total_scaled"]]
        .rename(columns={"total_scaled": "current_month"})
        .reset_index(drop=True)
    )

    stats = branch_stats.merge(current, on="branch")
    stats["utilization_pct"] = (
        (stats["current_month"] / stats["all_time_high"]) * 100
    ).round(2)
    stats["is_saturated"] = stats["utilization_pct"] >= 80.0

    return df.merge(stats, on="branch")


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Prints a clean branch-level saturation summary to the terminal."""
    sep = "=" * 68

    print(f"\n{sep}")
    print("  CONUT — MONTHLY SALES CLEAN SUMMARY (rep_s_00334_1_SMRY)")
    print(f"{sep}\n")

    # One summary row per branch
    summary = (
        df[["branch", "all_time_high", "current_month", "utilization_pct", "is_saturated"]]
        .drop_duplicates("branch")
        .sort_values("utilization_pct", ascending=False)
        .reset_index(drop=True)
    )

    for _, row in summary.iterrows():
        status = "⚠️  SATURATED" if row["is_saturated"] else "✅ OK"
        print(f"  Branch : {row['branch']}")
        print(f"    All-Time High   : {row['all_time_high']:>20,.2f}")
        print(f"    Current Month   : {row['current_month']:>20,.2f}")
        print(f"    Utilization     : {row['utilization_pct']:>19.1f}%")
        print(f"    Status          : {status}")
        print()

    print(f"{sep}")
    print("  RAW MONTHLY DATA (all branches)")
    print(f"{sep}\n")
    print(df[["branch", "month", "year", "total_scaled"]].to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean rep_s_00334_1_SMRY.csv and compute branch saturation."
    )
    parser.add_argument(
        "--input",  default="rep_s_00334_1_SMRY.csv",
        help="Path to rep_s_00334_1_SMRY.csv (default: current directory)"
    )
    parser.add_argument(
        "--output", default="monthly_sales_by_branch.json",
        help="Output JSON path (default: monthly_sales_by_branch.json)"
    )
    args = parser.parse_args()

    filepath = Path(args.input)
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return

    # Parse and enrich
    df = parse_monthly_sales(filepath)
    df = enrich_with_saturation(df)

    # Print to terminal
    print_summary(df)

    # Save to JSON — nested by branch
    out_path = Path(args.output)

    output = []
    for branch in df["branch"].unique():
        bdf      = df[df["branch"] == branch]
        ath_row  = bdf.loc[bdf["total_scaled"].idxmax()]
        last_row = bdf.iloc[-1]

        output.append({
            "branch":        branch,
            "all_time_high": round(float(ath_row["total_scaled"]), 2),
            "ath_month":     ath_row["month"],
            "current_month": {
                "month":        last_row["month"],
                "year":         int(last_row["year"]),
                "total_scaled": round(float(last_row["total_scaled"]), 2),
            },
            "monthly_sales": [
                {
                    "month":        r["month"],
                    "year":         int(r["year"]),
                    "total_scaled": round(float(r["total_scaled"]), 2),
                }
                for _, r in bdf.iterrows()
            ],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅  Saved to {out_path}  ({len(output)} branches)\n")


if __name__ == "__main__":
    main()
import csv
import os
import pandas as pd
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data"))
OUTPUT_DIR = BASE_DIR

# ── Items to exclude from basket analysis ─────────────────────────────────────
# These are logistics, free add-ons, or modifiers — not real purchased items
SKIP_ITEMS = {
    "DELIVERY CHARGE", "WATER", "VIA SPARKLING WATER",
    "FULL FAT MILK", "SKIMMED MILK", "OAT MILK", "ALMOND MILK",
    "WHIPPED CREAM...", "NO WHIPPED CREAM", "PRESSED", "REGULAR.",
    "REGULAR", "DECAF", "HOT", "ICED", "ICE CREAM ON TOP",
    "ICE CREAM ON THE SIDE", "NO TOPPINGS.", "ADD ICE CREAM"
}

# Prefixes that indicate a free modifier/topping (zero price expected)
MODIFIER_PREFIXES = {"[", "FREE ", "NO "}


def _clean_item_name(name):
    """Standardize item name — strip spaces, remove (R) return markers, uppercase."""
    name = name.strip().lstrip()
    name = name.replace(",(R)", "").replace(", (R)", "").replace(" (R)", "")
    name = name.replace(",", "").strip()
    name = name.rstrip(".").rstrip(",").strip()
    return name.upper()


def _is_modifier(item, price):
    """Return True if item is a free modifier that should be excluded."""
    if price == 0.0 and any(item.startswith(p) for p in MODIFIER_PREFIXES):
        return True
    if item in SKIP_ITEMS:
        return True
    return False


# ── 1. Clean customer baskets (REP_S_00502) ───────────────────────────────────
def clean_baskets():
    """
    Parses REP_S_00502 into a clean basket-level CSV.

    Output: clean_baskets.csv
        customer | item | qty | price | date_range

    Rules:
    - Skip negative qty rows (returns/cancellations)
    - Skip zero-price modifiers and free add-ons
    - Skip logistics items (delivery charge, water)
    - Handle page breaks mid-basket correctly
    - Strip (R) markers and standardize item names
    - Deduplicates on (customer, item) keeping highest qty if file already exists
    """
    filepath = os.path.join(DATA_DIR, "REP_S_00502.csv")
    rows = []
    current_customer = None
    date_range = None

    with open(filepath, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue

            cell = row[0].strip()

            # Extract date range from header row e.g. "30-Jan-26"
            if len(row) >= 3 and "From Date:" in str(row[1]):
                date_range = f"{row[1].strip()} to {row[2].strip()}"
                continue

            # Skip repeated page header rows
            if cell in ["Full Name", "Branch :Conut - Tyre", "Branch :Conut",
                        "Branch :Conut Jnah", "Branch :Main Street Coffee"]:
                continue

            # Skip report header rows
            if cell in ["Sales by customer in details (delivery)", "Conut - Tyre"]:
                continue

            # New customer
            if cell.startswith("Person_"):
                current_customer = cell
                continue

            # Skip total rows
            if cell == "Total :":
                continue

            # Item rows: col0=empty, col1=qty, col2=item, col3=price
            if cell == "" and current_customer and len(row) >= 4:
                try:
                    qty   = float(row[1].strip())
                    item  = _clean_item_name(row[2])
                    price = float(row[3].strip().replace(",", ""))

                    # Skip returns (negative qty)
                    if qty <= 0:
                        continue

                    # Skip modifiers and logistics
                    if _is_modifier(item, price):
                        continue

                    rows.append({
                        "customer": current_customer,
                        "item": item,
                        "qty": qty,
                        "price": price,
                        "date_range": date_range or "unknown"
                    })

                except (ValueError, IndexError):
                    continue

    new_df = pd.DataFrame(rows)

    # ── Append + deduplicate logic ─────────────────────────────────────────────
    output_file = os.path.join(OUTPUT_DIR, "clean_baskets.csv")

    if os.path.exists(output_file):
        print("Existing clean_baskets.csv found. Merging...")
        existing_df = pd.read_csv(output_file)
        combined    = pd.concat([existing_df, new_df], ignore_index=True)
        # Keep the row with the highest qty for each (customer, item, date_range)
        combined    = combined.sort_values("qty", ascending=False)
        combined    = combined.drop_duplicates(subset=["customer", "item", "date_range"], keep="first")
        combined    = combined.sort_values(["customer", "item"])
        combined.to_csv(output_file, index=False)
        print(f"Updated. Total basket rows: {len(combined)}")
    else:
        print("Creating clean_baskets.csv...")
        new_df.to_csv(output_file, index=False)
        print(f"Created. Total basket rows: {len(new_df)}")

    return new_df


# ── 2. Clean branch sales (rep_s_00191) ───────────────────────────────────────
def clean_branch_sales():
    """
    Parses rep_s_00191_SMRY into a clean per-branch, per-item sales CSV.

    Output: clean_branch_sales.csv
        branch | item | qty | revenue | revenue_per_unit | category

    Rules:
    - Skip zero-qty and zero-revenue rows
    - Skip modifier rows (dressings, milk options etc.)
    - Assign category (COFFEE, FRAPPE, SHAKE, FOOD, OTHER)
    - Deduplicates on (branch, item) summing qty/revenue if file already exists
    """
    filepath = os.path.join(DATA_DIR, "rep_s_00191_SMRY.csv")
    rows = []
    current_branch   = None
    current_category = None

    SKIP_KEYWORDS = [
        "Total", "Group:", "Division:", "Page", "Description",
        "Barcode", "Branch:", "Copyright"
    ]

    CATEGORY_MAP = {
        "Hot-Coffee Based": "COFFEE",
        "Frappes":          "FRAPPE",
        "Shakes":           "SHAKE",
        "Hot and Cold Drinks": "COLD_DRINK",
        "ITEMS":            "FOOD",
        "Extras and Sides": "EXTRA",
    }

    with open(filepath, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue

            cell = row[0].strip()

            # Detect branch
            if "Branch:" in cell and "Total" not in cell:
                current_branch = cell.replace("Branch:", "").strip()
                continue

            # Detect category/division
            if "Division:" in cell:
                div = cell.replace("Division:", "").strip()
                current_category = CATEGORY_MAP.get(div, "OTHER")
                continue

            if "Group:" in cell:
                grp = cell.replace("Group:", "").strip()
                current_category = CATEGORY_MAP.get(grp, current_category)
                continue

            # Skip junk rows
            if any(k in cell for k in SKIP_KEYWORDS):
                continue

            # Skip modifier rows (start with [ or are free options)
            if cell.startswith("["):
                continue

            # Data rows
            if current_branch and cell and len(row) >= 4:
                try:
                    qty     = float(row[2].strip().replace(",", ""))
                    revenue = float(row[3].strip().replace(",", ""))

                    if qty <= 0 or revenue <= 0:
                        continue

                    item = _clean_item_name(cell)
                    rev_per_unit = round(revenue / qty, 2)

                    rows.append({
                        "branch":           current_branch,
                        "item":             item,
                        "qty":              qty,
                        "revenue":          revenue,
                        "revenue_per_unit": rev_per_unit,
                        "category":         current_category or "OTHER"
                    })

                except (ValueError, IndexError):
                    continue

    new_df = pd.DataFrame(rows)

    # ── Append + deduplicate logic ─────────────────────────────────────────────
    output_file = os.path.join(OUTPUT_DIR, "clean_branch_sales.csv")

    if os.path.exists(output_file):
        print("Existing clean_branch_sales.csv found. Merging...")
        existing_df = pd.read_csv(output_file)
        combined    = pd.concat([existing_df, new_df], ignore_index=True)
        # Sum qty and revenue for same (branch, item) across runs
        combined    = combined.groupby(["branch", "item", "category"], as_index=False).agg(
            qty=("qty", "sum"),
            revenue=("revenue", "sum")
        )
        combined["revenue_per_unit"] = (combined["revenue"] / combined["qty"]).round(2)
        combined.to_csv(output_file, index=False)
        print(f"Updated. Total sales rows: {len(combined)}")
    else:
        print("Creating clean_branch_sales.csv...")
        new_df.to_csv(output_file, index=False)
        print(f"Created. Total sales rows: {len(new_df)}")

    return new_df


# ── Run both ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("GROWTH SERVICE PREPROCESSING")
    print("=" * 50)

    print("\n[1/2] Cleaning customer baskets...")
    baskets_df = clean_baskets()
    print(baskets_df.head(10).to_string())

    print("\n[2/2] Cleaning branch sales...")
    sales_df = clean_branch_sales()
    print(sales_df.head(10).to_string())

    print("\nDone. Files saved to services/growth/")
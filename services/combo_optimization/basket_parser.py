"""
Step 1 — Basket Parser for REP_S_00502.csv
==========================================
Parses the messy report-style CSV and outputs one clean basket per customer containing:
  - customer_id      : anonymized ID (e.g. Person_0130)
  - branch           : which Conut location they ordered from
  - order_type       : "Delivery" or "Dine-In / Live"
  - items            : list of real products (noise filtered out)
  - item_quantities  : dict of {item: net_qty} for that basket
  - num_items        : total number of distinct real products
"""

import csv
import re
import json
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — things to filter out
# ─────────────────────────────────────────────────────────────────────────────

# These exact item name patterns are always noise (modifiers, logistics, etc.)
NOISE_KEYWORDS = [
    "DELIVERY CHARGE",
    "WHIPPED CREAM",
    "NO WHIPPED CREAM",
    "PRESSED",
    "REGULAR",
    "NO TOPPINGS",
    "NO SPREAD",
    "NO SAUCE",
    "FULL FAT MILK",
    "SKIMMED MILK",
    "OAT MILK",
    "ALMOND MILK",
    "CARAMEL SAUCE",
    "NUTELLA SAUCE",
    "NUTELLA SPREAD",
    "WHITE CHOCOLATE SPREAD",
    "LOTUS SAUCE",
    "DARK CHOCOLATE DIP",
    "LOTUS DIP",
    "PISTACHIO TOPPING",
    "CARAMEL TOPPING",
    "ICE CREAM ON THE SIDE",
    "CRUSHED LOTUS",
    "CRUSHED OREO",
    "CRISPY CREPE",
    "BROWNIES",       # appears as a free topping/modifier (price=0)
    "NUTELLA SAUCE",
]

def is_noise_item(item_name: str, price: float) -> bool:
    """
    Returns True if the item should be excluded from the basket.
    
    Rules:
      1. Price = 0.00  →  it's a free modifier/topping, not a real product
      2. Starts with [  →  it's a dressing/option selector e.g. [CHOCOLATE DRESSING]
      3. Ends with (R)  →  it's a free "right-side" sauce/topping variant
      4. Matches a known noise keyword
    """
    name = item_name.strip().upper()

    # Rule 1: Free items are modifiers
    if price == 0.0:
        return True

    # Rule 2: Items in brackets are option selectors
    if name.startswith("["):
        return True

    # Rule 3: Items ending with "(R)" are free variants
    if name.endswith("(R)") or name.endswith("(R),"):
        return True

    # Rule 4: Explicit noise keyword match
    for keyword in NOISE_KEYWORDS:
        if keyword in name:
            return True

    return False


def clean_price(price_str: str) -> float:
    """Converts price strings like '1,251,486.48' or '-893,918.92' to float."""
    try:
        return float(price_str.replace(",", "").replace('"', "").strip())
    except (ValueError, AttributeError):
        return 0.0


def clean_item_name(name: str) -> str:
    """Strips leading/trailing spaces, dots, and normalizes the item name."""
    name = name.strip()
    # Remove trailing punctuation like "." and ","
    name = name.rstrip(".,")
    # Remove leading special characters
    name = name.lstrip("[")
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    return name.upper()


def is_page_header_line(row: list) -> bool:
    """
    Detects report page header rows (date lines, column headers, report titles).
    NOTE: Item rows also have empty col0, so we ONLY skip empty-col0 rows
    when they have no meaningful qty/price data (i.e. they are pure header noise).
    """
    if not row:
        return True

    first = str(row[0]).strip()
    second = str(row[1]).strip() if len(row) > 1 else ""

    # Column header row: "Full Name, Qty, Description, Price"
    if first == "Full Name":
        return True

    # Date/page line: e.g. "30-Jan-26, From Date: ..."
    if re.match(r"\d{2}-\w{3}-\d{2}", first):
        return True

    # Named report title lines (non-empty first column that is a branch/company name)
    if first in ("Conut - Tyre", "Conut Jnah", "Main Street Coffee"):
        return True

    # Sales report title
    if "Sales by customer" in first:
        return True

    # Empty first column: could be an item row OR a stray blank line
    # Item rows have a numeric qty in column 1 → do NOT skip them
    if first == "":
        try:
            float(second)       # if col1 is a number → it's an item row, keep it
            return False
        except ValueError:
            return True         # col1 is not a number → pure blank/noise line

    return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_baskets(filepath: str) -> list[dict]:
    """
    Reads the raw CSV and returns a list of clean customer baskets.
    """

    baskets = []
    skipped_customers = []   # customers whose orders fully cancelled out

    # State machine variables
    current_branch = None
    current_customer = None
    current_items = defaultdict(float)   # item_name → net_qty (handles cancellations)
    current_has_delivery = False

    def flush_customer():
        """Save the current customer's basket and reset state."""
        nonlocal current_customer, current_items, current_has_delivery

        if current_customer is None:
            return

        # Build basket: only keep items with net positive quantity
        basket_items = {}
        for item, net_qty in current_items.items():
            if net_qty > 0:
                basket_items[item] = net_qty

        if basket_items:
            # Normal customer with real purchases → add to baskets
            baskets.append({
                "customer_id":      current_customer,
                "branch":           current_branch,
                "order_type":       "Delivery" if current_has_delivery else "Dine-In / Live",
                "items":            list(basket_items.keys()),
                "item_quantities":  basket_items,
                "num_items":        len(basket_items),
            })
        else:
            # Customer existed but net qty = 0 on everything → fully cancelled order
            skipped_customers.append({
                "customer_id": current_customer,
                "branch":      current_branch,
                "reason":      "Fully cancelled order — all items netted to zero",
            })

        # Reset for next customer
        current_customer = None
        current_items = defaultdict(float)
        current_has_delivery = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)

        for row in reader:
            # Pad row to at least 4 columns to avoid index errors
            while len(row) < 4:
                row.append("")

            col0 = str(row[0]).strip()   # Full Name / Branch / empty
            col1 = str(row[1]).strip()   # Qty
            col2 = str(row[2]).strip()   # Description
            col3 = str(row[3]).strip()   # Price

            # ── Skip page/report header lines ──────────────────────────────
            if is_page_header_line(row):
                continue

            # ── Detect BRANCH change ───────────────────────────────────────
            # Format: "Branch :Conut - Tyre"  or  "Branch :Conut Jnah"
            if col0.startswith("Branch :") or col0.startswith("Branch:"):
                flush_customer()
                current_branch = col0.split(":", 1)[1].strip()
                continue

            # ── Detect TOTAL BRANCH line (end of a branch section) ─────────
            if col0.startswith("Total Branch"):
                flush_customer()
                continue

            # ── Detect TOTAL line (end of a customer block) ────────────────
            if col0.startswith("Total :"):
                # Don't flush yet — we rely on next Person line or Branch line
                # because a customer can span multiple pages
                continue

            # ── Detect new CUSTOMER ────────────────────────────────────────
            # Customer lines: "Person_0130" or "0 Person_0017" (Jnah branch adds "0 " prefix)
            customer_match = re.search(r"(Person_\d+)", col0)
            if customer_match:
                flush_customer()                          # save previous customer
                current_customer = customer_match.group(1)
                continue

            # ── Parse ITEM LINES ───────────────────────────────────────────
            # Item lines have: col0=empty, col1=qty, col2=description, col3=price
            if col0 == "" and col1 != "" and col2 != "":
                try:
                    qty = float(col1)
                except ValueError:
                    continue

                price = clean_price(col3)
                item_name = clean_item_name(col2)

                # Skip empty item names
                if not item_name:
                    continue

                # Check if this is the delivery charge (mark order type)
                if "DELIVERY CHARGE" in item_name.upper():
                    current_has_delivery = True
                    continue  # don't add to basket items

                # Filter noise
                if is_noise_item(item_name, price):
                    continue

                # Accumulate net quantity (handles +1/-1 cancellation pairs)
                current_items[item_name] += qty

    # Flush the very last customer in the file
    flush_customer()

    return baskets, skipped_customers


# ─────────────────────────────────────────────────────────────────────────────
# RUN & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_FILE = "REP_S_00502.csv"
    OUTPUT_FILE = "clean_baskets.json"

    print("Parsing baskets...")
    baskets, skipped = parse_baskets(INPUT_FILE)

    # ── Summary stats ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Total customers with purchases : {len(baskets)}")
    print(f"  Customers with cancelled orders: {len(skipped)}  ← excluded from baskets")

    branches = defaultdict(int)
    order_types = defaultdict(int)
    all_items = defaultdict(int)
    basket_sizes = []

    for b in baskets:
        branches[b["branch"]] += 1
        order_types[b["order_type"]] += 1
        basket_sizes.append(b["num_items"])
        for item in b["items"]:
            all_items[item] += 1

    print(f"\n  Customers per branch:")
    for branch, count in sorted(branches.items()):
        print(f"    {branch:<30} {count}")

    print(f"\n  Order type breakdown:")
    for otype, count in sorted(order_types.items()):
        print(f"    {otype:<30} {count}")

    print(f"\n  Avg basket size (distinct items): {sum(basket_sizes)/len(basket_sizes):.1f}")
    print(f"  Min basket size                 : {min(basket_sizes)}")
    print(f"  Max basket size                 : {max(basket_sizes)}")

    print(f"\n  Top 15 most purchased items:")
    for item, count in sorted(all_items.items(), key=lambda x: -x[1])[:15]:
        print(f"    {item:<45} ordered by {count} customers")

    print(f"{'='*55}\n")



    # ── Print a few sample baskets so you can visually verify ─────────────
    print("\n── SAMPLE BASKETS (first 5) ─────────────────────────────────")
    for basket in baskets[:5]:
        print(f"\nCustomer  : {basket['customer_id']}")
        print(f"Branch    : {basket['branch']}")
        print(f"Order Type: {basket['order_type']}")
        print(f"Items ({basket['num_items']}):")
        for item, qty in basket["item_quantities"].items():
            print(f"  x{int(qty)}  {item}")

    # ── Report skipped (cancelled) customers ───────────────────────────────
    if skipped:
        print(f"\n── SKIPPED CUSTOMERS (fully cancelled orders) ───────────────")
        for s in skipped:
            print(f"  {s['customer_id']:<15}  Branch: {s['branch']:<25}  Reason: {s['reason']}")

    # ── Save full output including skipped log ─────────────────────────────
    full_output = {
        "baskets":          baskets,
        "cancelled_orders": skipped,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print(f"\nFull output (baskets + cancelled log) saved to: {OUTPUT_FILE}")


import csv, os, re, json
from collections import defaultdict

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "REP_S_00502.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "clean_baskets.json")

NOISE_KEYWORDS = [
    "DELIVERY CHARGE","WHIPPED CREAM","NO WHIPPED CREAM","PRESSED","REGULAR",
    "NO TOPPINGS","NO SPREAD","NO SAUCE","FULL FAT MILK","SKIMMED MILK",
    "OAT MILK","ALMOND MILK","CARAMEL SAUCE","NUTELLA SAUCE","NUTELLA SPREAD",
    "WHITE CHOCOLATE SPREAD","LOTUS SAUCE","DARK CHOCOLATE DIP","LOTUS DIP",
    "PISTACHIO TOPPING","CARAMEL TOPPING","ICE CREAM ON THE SIDE",
    "CRUSHED LOTUS","CRUSHED OREO","CRISPY CREPE","BROWNIES",
]

def is_noise_item(item_name, price):
    name = item_name.strip().upper()
    if price == 0.0: return True
    if name.startswith("["): return True
    if name.endswith("(R)") or name.endswith("(R),"): return True
    for keyword in NOISE_KEYWORDS:
        if keyword in name: return True
    return False

def clean_price(price_str):
    try: return float(price_str.replace(",","").replace('"',"").strip())
    except: return 0.0

def clean_item_name(name):
    name = name.strip().rstrip(".,").lstrip("[")
    return re.sub(r"\s+", " ", name).upper()

def is_page_header_line(row):
    if not row: return True
    first  = str(row[0]).strip()
    second = str(row[1]).strip() if len(row) > 1 else ""
    if first == "Full Name": return True
    if re.match(r"\d{2}-\w{3}-\d{2}", first): return True
    if first in ("Conut - Tyre","Conut Jnah","Main Street Coffee"): return True
    if "Sales by customer" in first: return True
    if first == "":
        try: float(second); return False
        except ValueError: return True
    return False

def parse_baskets(filepath):
    baskets, skipped = [], []
    current_branch = current_customer = None
    current_items = defaultdict(float)
    current_has_delivery = False

    def flush_customer():
        nonlocal current_customer, current_items, current_has_delivery
        if current_customer is None: return
        basket_items = {item: qty for item, qty in current_items.items() if qty > 0}
        if basket_items:
            baskets.append({
                "customer_id": current_customer, "branch": current_branch,
                "order_type": "Delivery" if current_has_delivery else "Dine-In / Live",
                "items": list(basket_items.keys()),
                "item_quantities": basket_items, "num_items": len(basket_items),
            })
        else:
            skipped.append({"customer_id": current_customer, "branch": current_branch,
                            "reason": "Fully cancelled order"})
        current_customer = None
        current_items = defaultdict(float)
        current_has_delivery = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            while len(row) < 4: row.append("")
            col0, col1, col2, col3 = (str(row[i]).strip() for i in range(4))
            if is_page_header_line(row): continue
            if col0.startswith("Branch :") or col0.startswith("Branch:"):
                flush_customer()
                current_branch = col0.split(":", 1)[1].strip(); continue
            if col0.startswith("Total Branch"): flush_customer(); continue
            if col0.startswith("Total :"): continue
            customer_match = re.search(r"(Person_\d+)", col0)
            if customer_match:
                flush_customer()
                current_customer = customer_match.group(1); continue
            if col0 == "" and col1 != "" and col2 != "":
                try: qty = float(col1)
                except ValueError: continue
                price = clean_price(col3)
                item_name = clean_item_name(col2)
                if not item_name: continue
                if "DELIVERY CHARGE" in item_name.upper():
                    current_has_delivery = True; continue
                if is_noise_item(item_name, price): continue
                current_items[item_name] += qty
    flush_customer()
    return baskets, skipped

if __name__ == "__main__":
    print("Parsing baskets...")
    baskets, skipped = parse_baskets(INPUT_FILE)
    print(f"  {len(baskets)} customers  |  {len(skipped)} cancelled")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"baskets": baskets, "cancelled_orders": skipped}, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")
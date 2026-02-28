import pandas as pd
import os
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOOD_KEYWORDS  = {"CHIMNEY", "CONUT", "MINI", "SHARING BOX", "BROWNIES", "BITES"}
COFFEE_ITEMS   = {"CAFFE LATTE", "CAPPUCCINO", "DOUBLE ESPRESSO", "SINGLE ESPRESSO",
                  "CAFE MOCHA", "CAFFE AMERICANO", "AMERICAN COFFEE", "CARAMEL MACHIATO",
                  "WHITE MOCHA", "FLAT WHITE", "ESPRESSO MACCHIATO", "HOT CHOCOLATE COMBO"}
FRAPPE_ITEMS   = {"CARAMEL FRAPPE", "MOCHA FRAPPE", "CARAMEL MOCHA FRAPPE", "HAZELNUT FRAPPE",
                  "TOFFEE NUT FRAPPE", "VANILLA FRAPPE", "MATCHA FRAPPE", "ESPRESSO FRAPPE",
                  "HAZELNUT MOCHA FRAPPE", "WHITE MOCHA FRAPPE", "SALTED CARAMEL FRAPPE"}
SHAKE_ITEMS    = {"OREO MILKSHAKE", "VANILLA MILKSHAKE", "STRAWBERRY MILKSHAKE",
                  "PISTACHIO MILKSHAKE", "DOUBLE CHOCOLATE MILKSHAKE", "TOFFEE NUT MILKSHAKE",
                  "FRUIT LOOPS MILKSHAKE", "SALTED CARAMEL MILKSHAKE", "MATCHA MILKSHAKE",
                  "GRANOLA BERRIES MILKSHAKE", "PEANUT BUTTER MILKSHAKE"}


def _categorize(item):
    if item in COFFEE_ITEMS: return "COFFEE"
    if item in FRAPPE_ITEMS: return "FRAPPE"
    if item in SHAKE_ITEMS:  return "SHAKE"
    if any(k in item for k in FOOD_KEYWORDS): return "FOOD"
    return "OTHER"


def _load_baskets():
    df = pd.read_csv(os.path.join(BASE_DIR, "clean_baskets.csv"))
    baskets = defaultdict(list)
    for _, row in df.iterrows():
        baskets[row["customer"]].append(row["item"])
    return baskets


def _load_sales():
    return pd.read_csv(os.path.join(BASE_DIR, "clean_branch_sales.csv"))


def get_crosssell_opportunity():
    baskets  = _load_baskets()
    sales_df = _load_sales()

    food_orders = food_with_drink = food_no_drink = 0
    drink_with_food = Counter()

    for customer, items in baskets.items():
        unique_items = list(set(items))
        cats = [_categorize(i) for i in unique_items]
        has_food  = "FOOD" in cats
        has_drink = any(c in cats for c in ["COFFEE", "FRAPPE", "SHAKE"])

        if has_food:
            food_orders += 1
            if has_drink:
                food_with_drink += 1
                for item in unique_items:
                    if _categorize(item) in ["COFFEE", "FRAPPE", "SHAKE"]:
                        drink_with_food[item] += 1
            else:
                food_no_drink += 1

    gap_pct = round(100 * food_no_drink / food_orders, 1) if food_orders else 0
    best_bundles = [
        {"drink": item, "times_ordered_with_food": count}
        for item, count in drink_with_food.most_common(5)
    ]

    drink_rows    = sales_df[sales_df["category"].isin(["COFFEE", "FRAPPE", "SHAKE"])]
    avg_drink_price = drink_rows["revenue_per_unit"].mean()

    opportunities = {}
    for rate in [0.10, 0.20, 0.30]:
        converted = round(food_no_drink * rate)
        extra_rev  = round(converted * avg_drink_price)
        opportunities[f"{int(rate*100)}_pct_conversion"] = {
            "orders_converted": converted,
            "projected_extra_revenue_LBP": extra_rev
        }

    return {
        "insight": f"{gap_pct}% of food orders have NO drink - biggest growth lever",
        "total_food_orders": food_orders,
        "orders_with_drink": food_with_drink,
        "orders_without_drink": food_no_drink,
        "gap_pct": gap_pct,
        "best_bundles": best_bundles,
        "revenue_opportunity": opportunities
    }


def get_branch_benchmarks():
    sales_df = _load_sales()
    results  = {}

    for category in ["COFFEE", "FRAPPE", "SHAKE"]:
        cat_df        = sales_df[sales_df["category"] == category]
        branch_totals = cat_df.groupby("branch")["qty"].sum().sort_values(ascending=False)
        top_branch    = branch_totals.index[0]
        top_qty       = branch_totals.iloc[0]

        gaps = {}
        for branch in branch_totals.index[1:]:
            branch_qty = branch_totals[branch]
            gap_pct    = round(100 * (top_qty - branch_qty) / top_qty, 1)
            top_items  = set(cat_df[cat_df["branch"] == top_branch]["item"])
            br_items   = set(cat_df[cat_df["branch"] == branch]["item"])
            missing    = list(top_items - br_items)

            top_df = cat_df[cat_df["branch"] == top_branch].set_index("item")["qty"]
            br_df  = cat_df[cat_df["branch"] == branch].set_index("item")["qty"]
            underperforming = []
            for item in top_items & br_items:
                ratio = br_df.get(item, 0) / top_df.get(item, 1)
                if ratio < 0.5:
                    underperforming.append({
                        "item": item,
                        "this_branch_qty": int(br_df.get(item, 0)),
                        "top_branch_qty":  int(top_df.get(item, 0)),
                        "gap_pct": round((1 - ratio) * 100, 1)
                    })

            gaps[branch] = {
                "qty": int(branch_qty),
                "gap_pct_behind_top": gap_pct,
                "missing_items": missing[:5],
                "underperforming": sorted(underperforming, key=lambda x: -x["gap_pct"])[:5]
            }

        results[category] = {"top_branch": top_branch, "top_qty": int(top_qty), "gaps": gaps}

    return results


def get_high_value_items(branch: str = None):
    sales_df = _load_sales()
    if branch:
        sales_df = sales_df[sales_df["branch"] == branch]
    results = {}
    for category in ["COFFEE", "FRAPPE", "SHAKE"]:
        cat_df = sales_df[sales_df["category"] == category]
        top    = cat_df.sort_values("revenue_per_unit", ascending=False).head(5)
        results[category] = top[["branch", "item", "qty", "revenue_per_unit"]].to_dict(orient="records")
    return results


def get_full_strategy():
    return {
        "crosssell_opportunity": get_crosssell_opportunity(),
        "branch_benchmarks":     get_branch_benchmarks(),
        "high_value_items":      get_high_value_items()
    }


if __name__ == "__main__":
    import json
    print("=== Cross-sell Opportunity ===")
    print(json.dumps(get_crosssell_opportunity(), indent=2))
    print("\n=== High Value Items (Conut Jnah) ===")
    print(json.dumps(get_high_value_items("Conut Jnah"), indent=2))
import json, os, itertools
from collections import defaultdict, Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "clean_baskets.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "combo_results.json")
CHARTS_DIR  = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

BRAND_DARK = "#2C2C2C"; BRAND_GOLD = "#C9A84C"
BRAND_CREAM = "#F5F0E8"; BRAND_ACCENT = "#8B4513"

print("Loading baskets...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

raw_baskets = data["baskets"]
cancelled   = data["cancelled_orders"]
baskets = [{"customer_id": b["customer_id"], "branch": b["branch"],
            "order_type": b["order_type"], "items": frozenset(b["items"])}
           for b in raw_baskets if len(b["items"]) >= 1]
N = len(baskets)
item_sets_only = [b["items"] for b in baskets]
print(f"  {N} baskets  |  {len(cancelled)} cancelled")

item_counts  = Counter(item for b in baskets for item in b["items"])
basket_sizes = [len(b["items"]) for b in baskets]
size_counts  = Counter(basket_sizes)
branch_counts = Counter(b["branch"] for b in baskets)
top_items = item_counts.most_common(20)

print(f"\n  Unique products : {len(item_counts)}")
print(f"  Avg basket size : {sum(basket_sizes)/N:.1f}")
print(f"  Multi-item baskets: {N - size_counts[1]}")

# Charts
fig, ax = plt.subplots(figsize=(12,7)); fig.patch.set_facecolor(BRAND_CREAM); ax.set_facecolor(BRAND_CREAM)
items_p = [x[0] for x in top_items[:15]]; counts_p = [x[1] for x in top_items[:15]]
colors = [BRAND_GOLD if i==0 else BRAND_DARK for i in range(len(items_p))]
bars = ax.barh(items_p[::-1], counts_p[::-1], color=colors[::-1], edgecolor="white")
for bar, cnt in zip(bars, counts_p[::-1]):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f"{cnt}", va="center", fontsize=9, color=BRAND_DARK)
ax.set_title("Top 15 Most Purchased Items", fontsize=13, fontweight="bold", color=BRAND_DARK)
ax.spines[["top","right","left"]].set_visible(False); plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR,"1_top_items.png"), dpi=150, bbox_inches="tight"); plt.close()

MIN_SUPPORT=0.04; MIN_CONFIDENCE=0.30; MIN_LIFT=1.20; MAX_COMBO_SIZE=3
print(f"\n── APRIORI  (support≥{MIN_SUPPORT*100:.0f}%, confidence≥{MIN_CONFIDENCE*100:.0f}%, lift≥{MIN_LIFT})")

def get_support(itemset, baskets):
    return sum(1 for b in baskets if itemset.issubset(b)) / len(baskets)

def apriori(baskets, min_support, max_size):
    all_freq = {}
    item_count = Counter(item for b in baskets for item in b)
    n = len(baskets)
    freq_k = {frozenset([item]): cnt/n for item, cnt in item_count.items() if cnt/n >= min_support}
    all_freq.update(freq_k)
    print(f"  Size-1: {len(freq_k)}")
    for size in range(2, max_size+1):
        prev = list(freq_k.keys())
        candidates = {prev[i]|prev[j] for i in range(len(prev)) for j in range(i+1,len(prev)) if len(prev[i]|prev[j])==size}
        freq_k = {c: get_support(c, baskets) for c in candidates if get_support(c, baskets) >= min_support}
        print(f"  Size-{size}: {len(freq_k)}")
        if not freq_k: break
        all_freq.update(freq_k)
    return all_freq

frequent_itemsets = apriori(item_sets_only, MIN_SUPPORT, MAX_COMBO_SIZE)
print(f"  Total frequent itemsets: {len(frequent_itemsets)}")

rules = []
for itemset, support in frequent_itemsets.items():
    if len(itemset) < 2: continue
    items_list = list(itemset)
    for r in range(1, len(items_list)):
        for ant_tuple in itertools.combinations(items_list, r):
            antecedent = frozenset(ant_tuple)
            consequent = itemset - antecedent
            sup_ant = get_support(antecedent, item_sets_only)
            sup_con = get_support(consequent, item_sets_only)
            if sup_ant == 0 or sup_con == 0: continue
            confidence = support / sup_ant
            lift = confidence / sup_con
            if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                rules.append({"antecedent": sorted(antecedent), "consequent": sorted(consequent),
                               "combo": sorted(itemset), "support": round(support,4),
                               "confidence": round(confidence,4), "lift": round(lift,4),
                               "n_customers": int(support*N)})

rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
seen, unique_combos = set(), []
for rule in rules:
    key = tuple(rule["combo"])
    if key not in seen: seen.add(key); unique_combos.append(rule)

print(f"\n── TOP COMBOS")
for i, c in enumerate(unique_combos, 1):
    print(f"  {i}. {' + '.join(c['combo'])}  lift={c['lift']}  conf={c['confidence_pct'] if 'confidence_pct' in c else round(c['confidence']*100,1)}%")

output = {
    "summary": {"total_baskets": N, "cancelled_orders": len(cancelled),
                "unique_products": len(item_counts), "avg_basket_size": round(sum(basket_sizes)/N,2),
                "unique_combos_found": len(unique_combos)},
    "top_items": [{"item": item, "n_customers": cnt, "support_pct": round(cnt/N*100,1)}
                  for item, cnt in item_counts.most_common(30)],
    "combo_recommendations": [
        {"rank": i+1, "combo": c["combo"], "if_buys": c["antecedent"], "also_buys": c["consequent"],
         "support_pct": round(c["support"]*100,1), "confidence_pct": round(c["confidence"]*100,1),
         "lift": c["lift"], "n_customers": c["n_customers"],
         "interpretation": f"{c['n_customers']} customers ordered {' + '.join(c['combo'])} together. "
                           f"When someone buys {' + '.join(c['antecedent'])}, there is a "
                           f"{c['confidence']*100:.0f}% chance they also buy {' + '.join(c['consequent'])}. "
                           f"This is {c['lift']:.1f}x more likely than random."}
        for i, c in enumerate(unique_combos)
    ]
}
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {len(unique_combos)} combos to {OUTPUT_FILE}")
"""
Step 2 — EDA + Combo Optimization
===================================
Inputs : clean_baskets.json  (output of basket_parser.py)
Outputs: combo_results.json  — ranked combo recommendations
         charts/             — EDA visualization PNGs

Algorithm: Apriori — built from scratch, no external ML library needed.

Sections:
  A. Load data
  B. EDA  — understand the data before modeling
  C. Apriori — find frequent itemsets
  D. Association Rules — score each combo (Support / Confidence / Lift)
  E. Save results + print ranked recommendations
"""

import json
import os
import itertools
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "clean_baskets.json"
OUTPUT_FILE  = "combo_results.json"
CHARTS_DIR   = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

BRAND_DARK   = "#2C2C2C"
BRAND_GOLD   = "#C9A84C"
BRAND_CREAM  = "#F5F0E8"
BRAND_ACCENT = "#8B4513"

# =============================================================================
# A. LOAD DATA
# =============================================================================

print("Loading baskets...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

raw_baskets = data["baskets"]
cancelled   = data["cancelled_orders"]

baskets = []
for b in raw_baskets:
    item_set = frozenset(b["items"])
    if len(item_set) >= 1:
        baskets.append({
            "customer_id": b["customer_id"],
            "branch":      b["branch"],
            "order_type":  b["order_type"],
            "items":       item_set,
        })

N = len(baskets)
item_sets_only = [b["items"] for b in baskets]
print(f"  {N} baskets loaded  |  {len(cancelled)} cancelled orders excluded")


# =============================================================================
# B. EDA
# =============================================================================
print("\n── EDA ─────────────────────────────────────────────────────────────")

item_counts  = Counter(item for b in baskets for item in b["items"])
basket_sizes = [len(b["items"]) for b in baskets]
size_counts  = Counter(basket_sizes)
branch_counts = Counter(b["branch"] for b in baskets)
type_counts   = Counter(b["order_type"] for b in baskets)
top_items     = item_counts.most_common(20)

print(f"\n  Unique products            : {len(item_counts)}")
print(f"  Avg basket size            : {sum(basket_sizes)/N:.1f} items")
print(f"  Single-item baskets        : {size_counts[1]}  (can't form combos)")
print(f"  Multi-item baskets         : {N - size_counts[1]}  (usable for combos)")

print(f"\n  Branch breakdown:")
for branch, cnt in branch_counts.most_common():
    print(f"    {branch:<30} {cnt:>4} customers  ({cnt/N*100:.0f}%)")

print(f"\n  Order type:")
for otype, cnt in type_counts.most_common():
    print(f"    {otype:<30} {cnt:>4} customers  ({cnt/N*100:.0f}%)")

print(f"\n  Top 20 most purchased items:")
for rank, (item, cnt) in enumerate(top_items, 1):
    bar = "█" * int(cnt / top_items[0][1] * 30)
    print(f"    {rank:>2}. {item:<45} {cnt:>3}  {bar}")


# Chart 1: Top 15 Items
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BRAND_CREAM)
ax.set_facecolor(BRAND_CREAM)
items_plot  = [x[0] for x in top_items[:15]]
counts_plot = [x[1] for x in top_items[:15]]
colors = [BRAND_GOLD if i == 0 else BRAND_DARK for i in range(len(items_plot))]
bars = ax.barh(items_plot[::-1], counts_plot[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
for bar, cnt in zip(bars, counts_plot[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{cnt}", va="center", fontsize=9, color=BRAND_DARK, fontweight="bold")
ax.set_xlabel("Number of Customers", fontsize=11, color=BRAND_DARK)
ax.set_title("Top 15 Most Purchased Items", fontsize=13, fontweight="bold", color=BRAND_DARK, pad=15)
ax.tick_params(colors=BRAND_DARK)
ax.spines[["top","right","left"]].set_visible(False)
ax.spines["bottom"].set_color(BRAND_DARK)
ax.set_xlim(0, max(counts_plot) * 1.15)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/1_top_items.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Chart saved: {CHARTS_DIR}/1_top_items.png")


# Chart 2: Basket Size Distribution
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BRAND_CREAM)
ax.set_facecolor(BRAND_CREAM)
max_size = max(basket_sizes)
sizes    = list(range(1, max_size + 1))
freqs    = [size_counts.get(s, 0) for s in sizes]
colors_s = [BRAND_ACCENT if s == 1 else BRAND_GOLD for s in sizes]
bars = ax.bar(sizes, freqs, color=colors_s, edgecolor="white", linewidth=0.8)
for bar, freq in zip(bars, freqs):
    if freq > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(freq), ha="center", fontsize=10, color=BRAND_DARK, fontweight="bold")
ax.set_xlabel("Number of Distinct Items in Basket", fontsize=11, color=BRAND_DARK)
ax.set_ylabel("Number of Customers", fontsize=11, color=BRAND_DARK)
ax.set_title("Basket Size Distribution", fontsize=13, fontweight="bold", color=BRAND_DARK, pad=15)
ax.set_xticks(sizes)
legend_patches = [
    mpatches.Patch(color=BRAND_ACCENT, label="Single-item (can't form combos)"),
    mpatches.Patch(color=BRAND_GOLD,   label="Multi-item (usable for combos)"),
]
ax.legend(handles=legend_patches, fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(colors=BRAND_DARK)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/2_basket_sizes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {CHARTS_DIR}/2_basket_sizes.png")


# Chart 3: Branch + Order Type
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BRAND_CREAM)
branch_labels = list(branch_counts.keys())
branch_vals   = [branch_counts[b] for b in branch_labels]
type_labels   = list(type_counts.keys())
type_vals     = [type_counts[t] for t in type_labels]
for ax, labels, vals, palette, title in [
    (axes[0], branch_labels, branch_vals, [BRAND_GOLD, BRAND_DARK, BRAND_ACCENT], "Customers by Branch"),
    (axes[1], type_labels,   type_vals,   [BRAND_GOLD, BRAND_DARK],               "Delivery vs Dine-In"),
]:
    ax.set_facecolor(BRAND_CREAM)
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=palette[:len(labels)],
        autopct="%1.0f%%", startangle=140,
        textprops={"fontsize": 9, "color": BRAND_DARK},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title(title, fontsize=12, fontweight="bold", color=BRAND_DARK, pad=12)
plt.suptitle("Customer Distribution", fontsize=14, fontweight="bold", color=BRAND_DARK)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/3_customer_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved: {CHARTS_DIR}/3_customer_distribution.png")


# =============================================================================
# C. APRIORI ALGORITHM
# =============================================================================
"""
How it works:
  Round 1: Count every single item. Keep items above min_support threshold.
  Round 2: Combine frequent singles into pairs. Count and keep frequent pairs.
  Round 3: Combine frequent pairs into triplets. Keep frequent triplets.
  Stop when no new frequent sets are found.

  Key rule (Apriori property): if {A,B} is infrequent, then {A,B,C} must
  also be infrequent — so we prune and skip those candidates entirely.
"""

MIN_SUPPORT    = 0.04   # combo must appear in >= 4% of baskets
MIN_CONFIDENCE = 0.30
MIN_LIFT       = 1.20
MAX_COMBO_SIZE = 3

print("\n── APRIORI ─────────────────────────────────────────────────────────")
print(f"  Min support    : {MIN_SUPPORT*100:.0f}%  (~{int(MIN_SUPPORT*N)} of {N} baskets)")
print(f"  Min confidence : {MIN_CONFIDENCE*100:.0f}%")
print(f"  Min lift       : {MIN_LIFT}")


def get_support(itemset, baskets):
    count = sum(1 for b in baskets if itemset.issubset(b))
    return count / len(baskets)


def apriori(baskets, min_support, max_size):
    all_frequent = {}

    # Round 1: frequent singletons
    item_count = Counter(item for basket in baskets for item in basket)
    n = len(baskets)
    freq_k = {
        frozenset([item]): cnt / n
        for item, cnt in item_count.items()
        if cnt / n >= min_support
    }
    all_frequent.update(freq_k)
    print(f"  Size-1 frequent items : {len(freq_k)}")

    # Rounds 2, 3, ...: grow by one item each round
    for size in range(2, max_size + 1):
        prev_sets  = list(freq_k.keys())
        candidates = set()

        for i in range(len(prev_sets)):
            for j in range(i + 1, len(prev_sets)):
                union = prev_sets[i] | prev_sets[j]
                if len(union) == size:
                    candidates.add(union)

        freq_k = {}
        for candidate in candidates:
            sup = get_support(candidate, baskets)
            if sup >= min_support:
                freq_k[candidate] = sup

        print(f"  Size-{size} frequent sets  : {len(freq_k)}")
        if not freq_k:
            break
        all_frequent.update(freq_k)

    return all_frequent


frequent_itemsets = apriori(item_sets_only, MIN_SUPPORT, MAX_COMBO_SIZE)
print(f"\n  Total frequent itemsets: {len(frequent_itemsets)}")


# =============================================================================
# D. ASSOCIATION RULES
# =============================================================================
"""
For every frequent itemset, split it into antecedent and consequent and score:

  Support    = how common is the full combo across all baskets
  Confidence = P(combo) / P(antecedent)  →  how reliably does A lead to B?
  Lift       = confidence / P(consequent) →  how much better than random?
               Lift = 1.0 means A and B are independent (no real link)
               Lift > 1.0 means buying A makes buying B MORE likely  ✓
               Lift < 1.0 means they AVOID each other
"""

print("\n── ASSOCIATION RULES ───────────────────────────────────────────────")

rules = []

for itemset, support in frequent_itemsets.items():
    if len(itemset) < 2:
        continue

    items_list = list(itemset)
    for r in range(1, len(items_list)):
        for ant_tuple in itertools.combinations(items_list, r):
            antecedent = frozenset(ant_tuple)
            consequent = itemset - antecedent

            sup_ant = get_support(antecedent, item_sets_only)
            sup_con = get_support(consequent, item_sets_only)
            if sup_ant == 0 or sup_con == 0:
                continue

            confidence = support / sup_ant
            lift       = confidence / sup_con

            if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                rules.append({
                    "antecedent":   sorted(antecedent),
                    "consequent":   sorted(consequent),
                    "combo":        sorted(itemset),
                    "support":      round(support, 4),
                    "confidence":   round(confidence, 4),
                    "lift":         round(lift, 4),
                    "n_customers":  int(support * N),
                })

rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
print(f"  Rules passing thresholds: {len(rules)}")

# Deduplicate: A→B and B→A are the same combo for our purposes
seen_combos  = set()
unique_combos = []
for rule in rules:
    key = tuple(rule["combo"])
    if key not in seen_combos:
        seen_combos.add(key)
        unique_combos.append(rule)


# =============================================================================
# E. RESULTS
# =============================================================================

print("\n── TOP COMBO RECOMMENDATIONS ───────────────────────────────────────")
print(f"  {'COMBO':<55} {'Support':>8} {'Confidence':>11} {'Lift':>6} {'Customers':>10}")
print("  " + "─" * 95)

for i, c in enumerate(unique_combos, 1):
    combo_str = "  +  ".join(c["combo"])
    print(f"  {i:>2}. {combo_str:<52} {c['support']*100:>7.1f}%  "
          f"{c['confidence']*100:>10.1f}%  {c['lift']:>6.2f}  {c['n_customers']:>7} customers")

print(f"\n  Total unique combos: {len(unique_combos)}")


# Chart 4: Combo Recommendations
if unique_combos:
    top_n   = unique_combos[:12]
    labels  = [" + ".join(c["combo"]) for c in top_n]
    lifts   = [c["lift"] for c in top_n]
    confs   = [c["confidence"] * 100 for c in top_n]
    sups    = [c["support"] * 100 for c in top_n]
    ncusts  = [c["n_customers"] for c in top_n]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BRAND_CREAM)
    fig.suptitle("Top Combo Recommendations — Conut", fontsize=15,
                 fontweight="bold", color=BRAND_DARK)

    # Left: Lift bar chart
    ax = axes[0]
    ax.set_facecolor(BRAND_CREAM)
    bar_colors = [BRAND_GOLD if l == max(lifts) else BRAND_DARK for l in lifts]
    bars = ax.barh(range(len(top_n)), lifts, color=bar_colors, edgecolor="white")
    ax.axvline(x=1.0, color=BRAND_ACCENT, linestyle="--", linewidth=1.2,
               label="Lift = 1 (random chance)")
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(labels, fontsize=8, color=BRAND_DARK)
    ax.invert_yaxis()
    for bar, val in zip(bars, lifts):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}x", va="center", fontsize=8, fontweight="bold", color=BRAND_DARK)
    ax.set_xlabel("Lift  (higher = stronger pattern)", fontsize=10, color=BRAND_DARK)
    ax.set_title("Ranked by Lift", fontsize=11, fontweight="bold", color=BRAND_DARK)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(colors=BRAND_DARK)

    # Right: Support vs Confidence scatter
    ax2 = axes[1]
    ax2.set_facecolor(BRAND_CREAM)
    sc = ax2.scatter(sups, confs, c=lifts, cmap="YlOrBr",
                     s=[n * 20 for n in ncusts],
                     edgecolors=BRAND_DARK, linewidth=0.8, zorder=3)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Lift", fontsize=9, color=BRAND_DARK)
    cbar.ax.tick_params(labelcolor=BRAND_DARK)
    for i, c in enumerate(top_n):
        ax2.annotate(" + ".join(c["combo"]), (sups[i], confs[i]),
                     textcoords="offset points", xytext=(5, 4),
                     fontsize=6.5, color=BRAND_DARK)
    ax2.set_xlabel("Support — % of all orders with this combo", fontsize=10, color=BRAND_DARK)
    ax2.set_ylabel("Confidence — % chance of buying B given A", fontsize=10, color=BRAND_DARK)
    ax2.set_title("Support vs Confidence\n(bubble size = number of customers)",
                  fontsize=11, fontweight="bold", color=BRAND_DARK)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.tick_params(colors=BRAND_DARK)

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/4_combo_recommendations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {CHARTS_DIR}/4_combo_recommendations.png")


# Save full results JSON
output = {
    "summary": {
        "total_baskets":        N,
        "cancelled_orders":     len(cancelled),
        "unique_products":      len(item_counts),
        "avg_basket_size":      round(sum(basket_sizes) / N, 2),
        "min_support_used":     MIN_SUPPORT,
        "min_confidence_used":  MIN_CONFIDENCE,
        "min_lift_used":        MIN_LIFT,
        "frequent_itemsets":    len(frequent_itemsets),
        "unique_combos_found":  len(unique_combos),
    },
    "top_items": [
        {"item": item, "n_customers": cnt, "support_pct": round(cnt/N*100, 1)}
        for item, cnt in item_counts.most_common(30)
    ],
    "combo_recommendations": [
        {
            "rank":            i + 1,
            "combo":           c["combo"],
            "if_buys":         c["antecedent"],
            "also_buys":       c["consequent"],
            "support_pct":     round(c["support"] * 100, 1),
            "confidence_pct":  round(c["confidence"] * 100, 1),
            "lift":            c["lift"],
            "n_customers":     c["n_customers"],
            "interpretation":  (
                f"{c['n_customers']} customers ordered {' + '.join(c['combo'])} together. "
                f"When someone buys {' + '.join(c['antecedent'])}, there is a "
                f"{c['confidence']*100:.0f}% chance they also buy {' + '.join(c['consequent'])}. "
                f"This is {c['lift']:.1f}x more likely than by random chance."
            ),
        }
        for i, c in enumerate(unique_combos)
    ],
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n  Results saved to: {OUTPUT_FILE}")
print(f"\n{'='*65}")
print(f"  DONE. {len(unique_combos)} combos found. Charts in: {CHARTS_DIR}/")
print(f"{'='*65}")

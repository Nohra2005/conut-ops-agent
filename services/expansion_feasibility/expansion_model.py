"""
expansion_score.py
------------------
3-Pillar Expansion Scoring System for Conut branches.

Instead of relying on a single metric, this module scores each branch across
three independent pillars. Expansion is recommended only when converging
signals across all pillars justify the decision.

A closure check runs FIRST — before the expansion pillars — and can
immediately recommend closing a branch if it shows terminal decline signals.

CLOSURE CHECK (runs before expansion pillars)
    Is the branch in terminal decline and should it be closed?
    Triggers on: severe MoM crash + sustained low utilization + declining slope.

PILLAR 1 — Sustained Volume
    Is demand structurally high, not just spiking?
    Looks for consecutive months above the saturation threshold.

PILLAR 2 — Growth Trajectory
    Is demand accelerating, or was the peak a ceiling?
    Fits a linear regression on monthly sales and evaluates slope + R².

PILLAR 3 — Delivery Demand Density
    Is there a geographic pocket of loyal customers worth serving with a
    new branch?
    Evaluates customer count, repeat order rate, and average revenue.

SCORING:
    Each pillar returns:  "green" | "amber" | "red"
    Overall verdict:
        CONSIDER CLOSURE   → closure check triggered
        EXPAND             → all 3 pillars green
        MONITOR CLOSELY    → 2 green + 1 amber
        MONITOR            → all 3 amber
        DO NOT EXPAND      → any red or insufficient greens

Usage (standalone):
    python expansion_score.py
    python expansion_score.py --branch "Conut Jnah"

Usage (as LangChain Tool):
    from expansion_score import score_branch
"""

import json
import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SALES_JSON     = "monthly_sales_by_branch.json"
DEFAULT_CUSTOMERS_JSON = "customers.json"

# Pillar 1 — Volume thresholds
P1_SATURATION_THRESHOLD = 0.80   # 80% of ATH
P1_AMBER_THRESHOLD      = 0.60   # 60% of ATH
P1_MIN_CONSECUTIVE      = 2      # months in a row required

# Pillar 2 — Trajectory thresholds
P2_R2_MIN_GREEN  = 0.70   # regression must explain ≥70% of variance
P2_R2_MIN_AMBER  = 0.40
P2_MIN_MONTHS    = 3      # need at least 3 months to fit a meaningful line

# Pillar 3 — Delivery density thresholds
P3_MIN_CUSTOMERS_GREEN  = 150   # delivery customers
P3_MIN_CUSTOMERS_AMBER  = 50
P3_REPEAT_RATE_GREEN    = 0.15  # ≥15% of customers ordered more than once
P3_REPEAT_RATE_AMBER    = 0.05

# Closure check thresholds
# A branch triggers CONSIDER CLOSURE when ALL THREE conditions are met:
#   1. Last month utilization < 20% of ATH  (revenue has collapsed)
#   2. MoM change of last 2 months < -50%   (severe recent drop)
#   3. Linear regression slope is negative   (trend is structurally declining)
# This multi-condition requirement prevents false positives from data gaps.
CLOSURE_MAX_LAST_UTIL   = 0.20   # last month must be below 20% of ATH
CLOSURE_MAX_MOM_CHANGE  = -0.50  # MoM drop must be worse than -50%
# slope < 0 is the third condition (computed dynamically)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_sales(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {entry["branch"]: entry for entry in raw}


def load_customers(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 1 — SUSTAINED VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def score_pillar_1(branch_data: dict) -> dict:
    """
    Evaluates whether sales volume has been sustainably high — not just a spike.

    Logic:
      - Compute each month's utilization = month_sales / all_time_high
      - Find the longest consecutive streak of months above the threshold
      - Green : streak >= 2 AND threshold = 80%
      - Amber : streak >= 2 AND threshold = 60%, OR streak = 1 at 80%
      - Red   : no month crosses 60%

    Returns a dict with score, detail, and supporting data.
    """
    ath     = branch_data["all_time_high"]
    monthly = branch_data["monthly_sales"]

    # Compute per-month utilizations at both thresholds
    utils = []
    for m in monthly:
        util = (m["total_scaled"] / ath * 100) if ath else 0.0
        utils.append({
            "month":           m["month"],
            "total_scaled":    m["total_scaled"],
            "utilization_pct": round(util, 1),
            "above_80":        util >= P1_SATURATION_THRESHOLD * 100,
            "above_60":        util >= P1_AMBER_THRESHOLD * 100,
        })

    def longest_streak(key):
        best, cur = [], []
        for m in utils:
            if m[key]:
                cur.append(m["month"])
            else:
                if len(cur) > len(best):
                    best = cur[:]
                cur = []
        if len(cur) > len(best):
            best = cur[:]
        return best

    streak_80 = longest_streak("above_80")
    streak_60 = longest_streak("above_60")

    if len(streak_80) >= P1_MIN_CONSECUTIVE:
        score  = "green"
        detail = (f"{len(streak_80)} consecutive months ≥ {P1_SATURATION_THRESHOLD*100:.0f}% ATH: "
                  f"{', '.join(streak_80)}")
    elif len(streak_60) >= P1_MIN_CONSECUTIVE:
        score  = "amber"
        detail = (f"{len(streak_60)} consecutive months ≥ {P1_AMBER_THRESHOLD*100:.0f}% ATH "
                  f"(below the 80% bar): {', '.join(streak_60)}")
    elif len(streak_80) == 1:
        score  = "amber"
        detail = (f"Only 1 month ≥ {P1_SATURATION_THRESHOLD*100:.0f}% ATH "
                  f"({streak_80[0]}) — possible seasonal spike, not confirmed trend")
    else:
        score  = "red"
        detail = f"No month has reached {P1_AMBER_THRESHOLD*100:.0f}% of ATH"

    return {
        "score":               score,
        "detail":              detail,
        "monthly_utilizations": utils,
        "streak_above_80":     streak_80,
        "streak_above_60":     streak_60,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2 — GROWTH TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def score_pillar_2(branch_data: dict) -> dict:
    """
    Fits a linear regression on monthly sales values and evaluates whether
    the underlying trend is genuinely growing.

    Logic:
      - slope > 0 AND R² ≥ 0.70 → strong consistent growth → Green
      - slope > 0 AND R² ≥ 0.40 → noisy growth → Amber
      - slope > 0 AND R² < 0.40 → too noisy to call → Amber
      - slope ≤ 0               → declining / flat → Red
      - fewer than 3 months     → insufficient data → Red

    R² measures how well a straight line fits the data. R²=1.0 means perfect
    linear growth. R²=0.0 means the sales values are random noise.
    A high slope with a low R² means there was a spike but no real trend.
    """
    monthly = branch_data["monthly_sales"]

    if len(monthly) < P2_MIN_MONTHS:
        return {
            "score":             "red",
            "detail":            f"Only {len(monthly)} months of data — need ≥ {P2_MIN_MONTHS} to fit trajectory",
            "slope":             None,
            "r_squared":         None,
            "months_used":       len(monthly),
            "insufficient_data": True,
        }

    sales  = np.array([m["total_scaled"] for m in monthly])
    months = [m["month"] for m in monthly]
    x      = np.arange(len(sales), dtype=float)

    # Linear regression: y = slope * x + intercept
    slope, intercept = np.polyfit(x, sales, 1)

    # R-squared
    y_hat  = slope * x + intercept
    ss_res = np.sum((sales - y_hat) ** 2)
    ss_tot = np.sum((sales - np.mean(sales)) ** 2)
    r2     = round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else 0.0

    # MoM change for last 2 months (raw signal)
    mom_pct = ((sales[-1] - sales[-2]) / sales[-2] * 100) if sales[-2] > 0 else 0.0

    if slope <= 0:
        score  = "red"
        detail = (f"Declining trend — slope = {slope:+,.0f} units/month, R² = {r2:.2f}. "
                  f"Sales are contracting, not growing.")
    elif r2 >= P2_R2_MIN_GREEN:
        score  = "green"
        detail = (f"Strong consistent growth — slope = +{slope:,.0f} units/month, "
                  f"R² = {r2:.2f} (trend explains {r2*100:.0f}% of variance)")
    elif r2 >= P2_R2_MIN_AMBER:
        score  = "amber"
        detail = (f"Noisy upward trend — slope = +{slope:,.0f} units/month, "
                  f"R² = {r2:.2f}. Growth exists but is inconsistent.")
    else:
        score  = "amber"
        detail = (f"Spike-driven — slope = +{slope:,.0f} units/month but R² = {r2:.2f}. "
                  f"A large single-month spike is distorting the trend line.")

    return {
        "score":             score,
        "detail":            detail,
        "slope":             round(slope, 2),
        "r_squared":         r2,
        "mom_pct":           round(mom_pct, 1),
        "months_used":       len(monthly),
        "month_labels":      months,
        "insufficient_data": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 3 — DELIVERY DEMAND DENSITY
# ─────────────────────────────────────────────────────────────────────────────

def score_pillar_3(branch_name: str, customers: list) -> dict:
    """
    Evaluates the depth of delivery demand — how many customers exist, how
    loyal they are, and how much they spend.

    A new branch is most justified when there is a large pool of delivery
    customers who order repeatedly (they are loyal to the brand but are
    currently served at a distance). Converting them to walk-in customers
    would improve margin and reduce logistics cost.

    Logic:
      - Green : customer count ≥ 150  AND  repeat rate ≥ 15%
      - Amber : customer count ≥  50  AND  repeat rate ≥  5%
               OR high count with low repeat (brand awareness without loyalty yet)
      - Red   : fewer than 50 delivery customers
    """
    branch_customers = [c for c in customers if c["branch"] == branch_name]
    n                = len(branch_customers)

    if n == 0:
        return {
            "score":          "red",
            "detail":         "No delivery customer records found for this branch.",
            "customer_count": 0,
            "repeat_count":   0,
            "repeat_rate_pct": 0.0,
            "avg_order_value": 0.0,
            "avg_orders_per_customer": 0.0,
        }

    repeat_customers = [c for c in branch_customers if (c["num_orders"] or 0) > 1]
    repeat_rate      = len(repeat_customers) / n

    revenues = [c["total_sales_scaled"] for c in branch_customers if c["total_sales_scaled"]]
    avg_rev  = sum(revenues) / len(revenues) if revenues else 0.0

    orders = [c["num_orders"] for c in branch_customers if c["num_orders"]]
    avg_orders_per_customer = sum(orders) / len(orders) if orders else 0.0

    if n >= P3_MIN_CUSTOMERS_GREEN and repeat_rate >= P3_REPEAT_RATE_GREEN:
        score  = "green"
        detail = (f"{n} delivery customers, {repeat_rate*100:.1f}% repeat rate — "
                  f"large loyal delivery base ready to convert to walk-in")
    elif n >= P3_MIN_CUSTOMERS_AMBER and repeat_rate >= P3_REPEAT_RATE_AMBER:
        score  = "amber"
        detail = (f"{n} delivery customers, {repeat_rate*100:.1f}% repeat rate — "
                  f"reasonable base but loyalty is still building")
    elif n >= P3_MIN_CUSTOMERS_GREEN and repeat_rate < P3_REPEAT_RATE_AMBER:
        score  = "amber"
        detail = (f"{n} delivery customers but only {repeat_rate*100:.1f}% repeat — "
                  f"high reach but low loyalty, may be a new market")
    else:
        score  = "red"
        detail = (f"Only {n} delivery customers — insufficient delivery demand density "
                  f"to justify a new branch in this area")

    return {
        "score":                   score,
        "detail":                  detail,
        "customer_count":          n,
        "repeat_count":            len(repeat_customers),
        "repeat_rate_pct":         round(repeat_rate * 100, 1),
        "avg_order_value_scaled":  round(avg_rev, 2),
        "avg_orders_per_customer": round(avg_orders_per_customer, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLOSURE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_closure(branch_data: dict) -> dict:
    """
    Runs BEFORE the expansion pillars to detect branches in terminal decline.

    A branch should be considered for closure when all three signals converge:
      1. Last month utilization < 20% of ATH  — revenue has collapsed
      2. MoM change (last 2 months) < -50%    — severe recent crash, not a blip
      3. Linear regression slope is negative   — the overall trend is declining

    Requiring all three together prevents false positives:
      - A data gap (missing month) looks like a crash but won't have a negative slope
      - A seasonal dip is low but won't have a negative slope if the prior trend was strong
      - A one-bad-month won't produce both a -50% MoM AND a negative overall slope

    Returns a dict with:
        should_close    : bool
        reason          : str  — human-readable explanation
        last_util_pct   : float
        mom_pct         : float
        slope           : float
        signals_fired   : list[str]  — which of the 3 conditions triggered
    """
    monthly = branch_data["monthly_sales"]
    ath     = branch_data["all_time_high"]

    if len(monthly) < 2:
        return {
            "should_close":  False,
            "reason":        "Insufficient data to evaluate closure.",
            "last_util_pct": 0.0,
            "mom_pct":       0.0,
            "slope":         0.0,
            "signals_fired": [],
        }

    sales = [m["total_scaled"] for m in monthly]

    # Signal 1 — last month utilization
    last_util_pct = (sales[-1] / ath * 100) if ath else 0.0
    sig1 = last_util_pct < (CLOSURE_MAX_LAST_UTIL * 100)

    # Signal 2 — MoM change between last 2 months
    mom_pct = ((sales[-1] - sales[-2]) / sales[-2] * 100) if sales[-2] > 0 else 0.0
    sig2 = mom_pct < (CLOSURE_MAX_MOM_CHANGE * 100)

    # Signal 3 — overall linear regression slope
    x     = np.arange(len(sales), dtype=float)
    slope, _ = np.polyfit(x, sales, 1)
    sig3  = slope < 0

    signals_fired = []
    if sig1: signals_fired.append(f"last month at {last_util_pct:.1f}% of ATH (< {CLOSURE_MAX_LAST_UTIL*100:.0f}%)")
    if sig2: signals_fired.append(f"MoM crash of {mom_pct:+.1f}% (< {CLOSURE_MAX_MOM_CHANGE*100:.0f}%)")
    if sig3: signals_fired.append(f"negative overall slope ({slope:+,.0f} units/month)")

    should_close = sig1 and sig2 and sig3

    if should_close:
        reason = (
            f"All 3 closure signals fired: {'; '.join(signals_fired)}. "
            f"The branch shows a severe recent revenue collapse on top of a "
            f"structurally declining trend. This is not a seasonal dip — "
            f"it warrants serious consideration of closure or major restructuring."
        )
    elif len(signals_fired) == 2:
        reason = (
            f"2 of 3 closure signals active ({'; '.join(signals_fired)}). "
            f"Situation is concerning but not yet terminal. Investigate urgently."
        )
    elif len(signals_fired) == 1:
        reason = f"1 closure signal: {signals_fired[0]}. Monitor closely."
    else:
        reason = "No closure signals. Branch is not in terminal decline."

    return {
        "should_close":  should_close,
        "reason":        reason,
        "last_util_pct": round(last_util_pct, 2),
        "mom_pct":       round(mom_pct, 2),
        "slope":         round(slope, 2),
        "signals_fired": signals_fired,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OVERALL VERDICT
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(p1: dict, p2: dict, p3: dict) -> tuple[str, str]:
    """
    Combines the 3 pillar scores into an overall verdict.

    Rules (in priority order):
      EXPAND         : all 3 pillars are green
      MONITOR        : exactly 2 pillars green + 1 amber
      MONITOR        : all 3 pillars amber (strong signal but nothing confirmed)
      DO NOT EXPAND  : any pillar is red
                       OR only 1 pillar is green

    Returns (verdict_label, reasoning_string)
    """
    scores = [p1["score"], p2["score"], p3["score"]]
    greens = scores.count("green")
    ambers = scores.count("amber")
    reds   = scores.count("red")

    PILLAR_NAMES = ["Sustained Volume", "Growth Trajectory", "Delivery Density"]

    red_pillars   = [PILLAR_NAMES[i] for i, s in enumerate(scores) if s == "red"]
    amber_pillars = [PILLAR_NAMES[i] for i, s in enumerate(scores) if s == "amber"]
    green_pillars = [PILLAR_NAMES[i] for i, s in enumerate(scores) if s == "green"]

    if greens == 3:
        return (
            "EXPAND",
            f"All 3 pillars are green ({', '.join(green_pillars)}). "
            f"Sustained volume, consistent growth, and strong delivery demand "
            f"all converge — expansion is justified."
        )
    elif greens == 2 and ambers == 1:
        return (
            "MONITOR CLOSELY",
            f"2 pillars green ({', '.join(green_pillars)}), "
            f"1 amber ({', '.join(amber_pillars)}). "
            f"Strong case forming — re-evaluate next month. "
            f"If the amber pillar resolves to green, expand."
        )
    elif reds == 0 and ambers == 3:
        return (
            "MONITOR",
            f"All 3 pillars amber. Signals are building but nothing is "
            f"confirmed yet. Re-evaluate in 2 months."
        )
    elif reds > 0:
        return (
            "DO NOT EXPAND",
            f"Red flag(s) on: {', '.join(red_pillars)}. "
            f"Expansion would be premature — address these weaknesses first."
        )
    else:
        return (
            "DO NOT EXPAND",
            f"Only {greens} pillar(s) green. Need all 3 aligned before expanding."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def score_branch(
    branch_name:    str,
    sales_path:     str = DEFAULT_SALES_JSON,
    customers_path: str = DEFAULT_CUSTOMERS_JSON,
) -> dict:
    """
    Runs the full 3-pillar expansion scoring for a given branch.

    Returns a structured score dict AND prints a formatted COO-ready report.

    Parameters
    ----------
    branch_name    : Branch name (must match key in monthly_sales_by_branch.json)
    sales_path     : Path to monthly_sales_by_branch.json
    customers_path : Path to customers.json

    Returns
    -------
    dict with keys: branch, pillar_1, pillar_2, pillar_3, overall, recommendation
    """
    SEP  = "=" * 65
    SEP2 = "─" * 65

    # ── Load ──────────────────────────────────────────────────────────────────
    sales_data = load_sales(sales_path)
    customers  = load_customers(customers_path)

    if branch_name not in sales_data:
        raise ValueError(f"Branch '{branch_name}' not found. "
                         f"Available: {list(sales_data.keys())}")

    # ── CLOSURE CHECK — runs before expansion pillars ────────────────────────
    closure = check_closure(sales_data[branch_name])

    # ── Score each pillar ─────────────────────────────────────────────────────
    p1 = score_pillar_1(sales_data[branch_name])
    p2 = score_pillar_2(sales_data[branch_name])
    p3 = score_pillar_3(branch_name, customers)

    # If closure is triggered, override verdict immediately
    if closure["should_close"]:
        verdict   = "CONSIDER CLOSURE"
        reasoning = closure["reason"]
    else:
        verdict, reasoning = compute_verdict(p1, p2, p3)

    # ── Build result object ───────────────────────────────────────────────────
    result = {
        "branch": branch_name,
        "closure_check": {
            "should_close":  closure["should_close"],
            "signals_fired": closure["signals_fired"],
            "detail":        closure["reason"],
            "last_util_pct": closure["last_util_pct"],
            "mom_pct":       closure["mom_pct"],
            "slope":         closure["slope"],
        },
        "pillar_1_volume": {
            "score":  p1["score"],
            "detail": p1["detail"],
        },
        "pillar_2_trajectory": {
            "score":     p2["score"],
            "detail":    p2["detail"],
            "slope":     p2.get("slope"),
            "r_squared": p2.get("r_squared"),
        },
        "pillar_3_delivery": {
            "score":            p3["score"],
            "detail":           p3["detail"],
            "customer_count":   p3["customer_count"],
            "repeat_rate_pct":  p3["repeat_rate_pct"],
            "avg_order_value":  p3["avg_order_value_scaled"],
        },
        "overall":        verdict,
        "recommendation": reasoning,
    }

    # ── Print report ──────────────────────────────────────────────────────────
    ICONS = {"green": "🟢", "amber": "🟡", "red": "🔴"}
    VERDICT_ICONS = {
        "EXPAND":           "🚀",
        "MONITOR CLOSELY":  "👀",
        "MONITOR":          "📊",
        "DO NOT EXPAND":    "🛑",
        "CONSIDER CLOSURE": "🚨",
    }

    print(f"\n{SEP}")
    print(f"  EXPANSION SCORE CARD — {branch_name.upper()}")
    print(f"{SEP}\n")

    # ── Closure check block ───────────────────────────────────────────────────
    closure_icon = "🚨" if closure["should_close"] else ("⚠️ " if closure["signals_fired"] else "✅")
    print(f"  {SEP2}")
    print(f"  CLOSURE CHECK")
    print(f"  {SEP2}")
    print(f"  {closure_icon}  Last month utilization : {closure['last_util_pct']:.1f}% of ATH  "
          f"(threshold < {CLOSURE_MAX_LAST_UTIL*100:.0f}%)  "
          f"{'← TRIGGERED' if closure['last_util_pct'] < CLOSURE_MAX_LAST_UTIL*100 else 'OK'}")
    print(f"  {closure_icon}  MoM change (last 2 mo) : {closure['mom_pct']:+.1f}%  "
          f"(threshold < {CLOSURE_MAX_MOM_CHANGE*100:.0f}%)  "
          f"{'← TRIGGERED' if closure['mom_pct'] < CLOSURE_MAX_MOM_CHANGE*100 else 'OK'}")
    print(f"  {closure_icon}  Overall slope          : {closure['slope']:+,.0f} units/month  "
          f"{'← TRIGGERED (negative)' if closure['slope'] < 0 else 'OK (positive)'}")
    if closure["should_close"]:
        print(f"\n  🚨  ALL 3 CLOSURE SIGNALS FIRED — CONSIDER CLOSING THIS BRANCH")
    elif closure["signals_fired"]:
        print(f"\n  ⚠️   {len(closure['signals_fired'])}/3 signals active — monitor urgently")
    else:
        print(f"\n  ✅  Branch is NOT in terminal decline")
    print()

    # Per-month breakdown (Pillar 1 data)
    print(f"  Monthly Sales vs ATH  (ATH = {sales_data[branch_name]['all_time_high']:,.0f})")
    print(f"  {'─'*60}")
    print(f"  {'Month':<12}  {'Sales':>20}   {'Util %':>7}   {'≥80%':>5}   {'≥60%':>5}")
    print(f"  {'─'*12}  {'─'*20}   {'─'*7}   {'─'*5}   {'─'*5}")
    for m in p1["monthly_utilizations"]:
        print(f"  {m['month']:<12}  {m['total_scaled']:>20,.0f}   "
              f"{m['utilization_pct']:>6.1f}%   "
              f"{'  ✓' if m['above_80'] else '  ✗':>5}   "
              f"{'  ✓' if m['above_60'] else '  ✗':>5}")
    print()

    # Pillar scores
    print(f"  {SEP2}")
    print(f"  PILLAR SCORES")
    print(f"  {SEP2}")
    print(f"  {ICONS[p1['score']]}  P1 — Sustained Volume    [{p1['score'].upper()}]")
    print(f"       {p1['detail']}")
    print()
    print(f"  {ICONS[p2['score']]}  P2 — Growth Trajectory   [{p2['score'].upper()}]")
    print(f"       {p2['detail']}")
    if p2.get("slope") is not None:
        print(f"       Slope = {p2['slope']:+,.0f} units/month  |  R² = {p2['r_squared']:.3f}")
    print()
    print(f"  {ICONS[p3['score']]}  P3 — Delivery Density    [{p3['score'].upper()}]")
    print(f"       {p3['detail']}")
    print(f"       Customers: {p3['customer_count']}  |  "
          f"Repeat rate: {p3['repeat_rate_pct']}%  |  "
          f"Avg order value: {p3['avg_order_value_scaled']:,.0f}")
    print()

    # Overall verdict
    print(f"  {SEP2}")
    vi = VERDICT_ICONS.get(verdict, "")
    print(f"  {vi}  VERDICT: {verdict}")
    print(f"  {SEP2}")
    print(f"  {reasoning}")
    print(f"\n{SEP}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ALL BRANCHES AT ONCE
# ─────────────────────────────────────────────────────────────────────────────

def score_all_branches(
    sales_path:     str = DEFAULT_SALES_JSON,
    customers_path: str = DEFAULT_CUSTOMERS_JSON,
) -> list:
    """
    Runs score_branch() for every branch in the sales file and prints
    a compact summary comparison table.
    """
    sales_data = load_sales(sales_path)
    all_results = []

    for branch_name in sales_data:
        result = score_branch(branch_name, sales_path, customers_path)
        all_results.append(result)

    # Summary table
    SEP = "=" * 65
    ICONS = {"green": "🟢", "amber": "🟡", "red": "🔴"}
    VERDICT_ICONS = {
        "EXPAND":          "🚀",
        "MONITOR CLOSELY": "👀",
        "MONITOR":         "📊",
        "DO NOT EXPAND":   "🛑",
    }

    print(f"\n{SEP}")
    print("  EXPANSION SUMMARY — ALL BRANCHES")
    print(f"{SEP}\n")
    print(f"  {'Branch':<22}  {'Closure':>8}  {'P1 Vol':>8}  {'P2 Traj':>8}  {'P3 Deliv':>9}  {'Verdict'}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*22}")

    for r in all_results:
        ci  = "🚨" if r["closure_check"]["should_close"] else ("⚠️ " if r["closure_check"]["signals_fired"] else "✅")
        p1i = ICONS[r["pillar_1_volume"]["score"]]
        p2i = ICONS[r["pillar_2_trajectory"]["score"]]
        p3i = ICONS[r["pillar_3_delivery"]["score"]]
        vi  = VERDICT_ICONS.get(r["overall"], "")
        print(f"  {r['branch']:<22}  {ci:>8}  {p1i:>8}  {p2i:>8}  {p3i:>9}  {vi} {r['overall']}")

    print(f"\n{SEP}\n")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conut 3-Pillar Expansion Scoring")
    parser.add_argument("--branch",    default=None,
                        help="Branch to score. Omit to score all branches.")
    parser.add_argument("--sales",     default=DEFAULT_SALES_JSON)
    parser.add_argument("--customers", default=DEFAULT_CUSTOMERS_JSON)
    args = parser.parse_args()

    if args.branch:
        score_branch(args.branch, args.sales, args.customers)
    else:
        score_all_branches(args.sales, args.customers)
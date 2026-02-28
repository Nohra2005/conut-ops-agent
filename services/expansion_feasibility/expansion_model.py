import json, argparse, numpy as np, os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data"))

DEFAULT_SALES_JSON     = os.path.join(BASE_DIR, "monthly_sales_by_branch.json")
DEFAULT_CUSTOMERS_JSON = os.path.join(BASE_DIR, "customers.json")

P1_SATURATION_THRESHOLD = 0.80
P1_AMBER_THRESHOLD      = 0.60
P1_MIN_CONSECUTIVE      = 2
P2_R2_MIN_GREEN         = 0.70
P2_R2_MIN_AMBER         = 0.40
P2_MIN_MONTHS           = 3
P3_MIN_CUSTOMERS_GREEN  = 150
P3_MIN_CUSTOMERS_AMBER  = 50
P3_REPEAT_RATE_GREEN    = 0.15
P3_REPEAT_RATE_AMBER    = 0.05
CLOSURE_MAX_LAST_UTIL   = 0.20
CLOSURE_MAX_MOM_CHANGE  = -0.50

def load_sales(path): 
    with open(path, encoding="utf-8") as f: raw = json.load(f)
    return {e["branch"]: e for e in raw}

def load_customers(path):
    with open(path, encoding="utf-8") as f: return json.load(f)

def score_pillar_1(branch_data):
    ath = branch_data["all_time_high"]
    monthly = branch_data["monthly_sales"]
    utils = []
    for m in monthly:
        util = (m["total_scaled"] / ath * 100) if ath else 0.0
        utils.append({
            "month": m["month"], 
            "total_scaled": m["total_scaled"],
            "utilization_pct": round(float(util), 1),
            "above_80": bool(util >= P1_SATURATION_THRESHOLD*100),
            "above_60": bool(util >= P1_AMBER_THRESHOLD*100)
        })
        
    def longest_streak(key):
        best, cur = [], []
        for m in utils:
            if m[key]: cur.append(m["month"])
            else:
                if len(cur) > len(best): best = cur[:]
                cur = []
        if len(cur) > len(best): best = cur[:]
        return best
        
    streak_80 = longest_streak("above_80")
    streak_60 = longest_streak("above_60")
    
    if len(streak_80) >= P1_MIN_CONSECUTIVE:
        score  = "green"
        detail = f"{len(streak_80)} consecutive months ≥ 80% ATH: {', '.join(streak_80)}"
    elif len(streak_60) >= P1_MIN_CONSECUTIVE:
        score  = "amber"
        detail = f"{len(streak_60)} consecutive months ≥ 60% ATH (below 80% bar)"
    elif len(streak_80) == 1:
        score  = "amber"
        detail = f"Only 1 month ≥ 80% ATH ({streak_80[0]}) — possible seasonal spike"
    else:
        score  = "red"
        detail = "No month has reached 60% of ATH"
        
    return {"score": score, "detail": detail, "monthly_utilizations": utils}

def score_pillar_2(branch_data):
    monthly = branch_data["monthly_sales"]
    if len(monthly) < P2_MIN_MONTHS:
        return {"score": "red", "detail": f"Only {len(monthly)} months — need ≥ {P2_MIN_MONTHS}",
                "slope": None, "r_squared": None, "insufficient_data": True}
    
    sales  = np.array([m["total_scaled"] for m in monthly])
    months = [m["month"] for m in monthly]
    x      = np.arange(len(sales), dtype=float)
    slope, intercept = np.polyfit(x, sales, 1)
    y_hat  = slope * x + intercept
    ss_res = np.sum((sales - y_hat)**2)
    ss_tot = np.sum((sales - np.mean(sales))**2)
    r2     = round(1 - ss_res/ss_tot, 3) if ss_tot > 0 else 0.0
    mom_pct = ((sales[-1] - sales[-2]) / sales[-2] * 100) if sales[-2] > 0 else 0.0
    
    if slope <= 0:
        score  = "red";   detail = f"Declining trend — slope={slope:+,.0f}/month, R²={r2:.2f}"
    elif r2 >= P2_R2_MIN_GREEN:
        score  = "green"; detail = f"Strong consistent growth — slope=+{slope:,.0f}/month, R²={r2:.2f}"
    elif r2 >= P2_R2_MIN_AMBER:
        score  = "amber"; detail = f"Noisy upward trend — slope=+{slope:,.0f}/month, R²={r2:.2f}"
    else:
        score  = "amber"; detail = f"Spike-driven — slope=+{slope:,.0f}/month but R²={r2:.2f} (too noisy)"
        
    return {
        "score": score, 
        "detail": detail, 
        "slope": round(float(slope), 2),
        "r_squared": float(r2), 
        "mom_pct": round(float(mom_pct), 1), 
        "month_labels": months, 
        "insufficient_data": False
    }

def score_pillar_3(branch_name, customers):
    bc = [c for c in customers if c["branch"] == branch_name]
    n  = len(bc)
    if n == 0:
        return {"score":"red","detail":"No delivery customer records found.",
                "customer_count":0,"repeat_count":0,"repeat_rate_pct":0.0,
                "avg_order_value_scaled":0.0,"avg_orders_per_customer":0.0}
                
    repeat    = [c for c in bc if (c["num_orders"] or 0) > 1]
    rep_rate  = len(repeat) / n
    revenues  = [c["total_sales_scaled"] for c in bc if c["total_sales_scaled"]]
    avg_rev   = sum(revenues)/len(revenues) if revenues else 0.0
    orders    = [c["num_orders"] for c in bc if c["num_orders"]]
    avg_ord   = sum(orders)/len(orders) if orders else 0.0
    
    if n >= P3_MIN_CUSTOMERS_GREEN and rep_rate >= P3_REPEAT_RATE_GREEN:
        score  = "green"; detail = f"{n} delivery customers, {rep_rate*100:.1f}% repeat rate — strong loyal base"
    elif n >= P3_MIN_CUSTOMERS_AMBER and rep_rate >= P3_REPEAT_RATE_AMBER:
        score  = "amber"; detail = f"{n} customers, {rep_rate*100:.1f}% repeat — reasonable but still building"
    elif n >= P3_MIN_CUSTOMERS_GREEN and rep_rate < P3_REPEAT_RATE_AMBER:
        score  = "amber"; detail = f"{n} customers but only {rep_rate*100:.1f}% repeat — high reach, low loyalty"
    else:
        score  = "red";   detail = f"Only {n} delivery customers — insufficient density"
        
    return {
        "score": score, 
        "detail": detail, 
        "customer_count": int(n), 
        "repeat_count": len(repeat),
        "repeat_rate_pct": round(float(rep_rate*100), 1), 
        "avg_order_value_scaled": round(float(avg_rev), 2),
        "avg_orders_per_customer": round(float(avg_ord), 2)
    }

def check_closure(branch_data):
    monthly = branch_data["monthly_sales"]
    ath     = branch_data["all_time_high"]
    if len(monthly) < 2:
        return {"should_close": False, "reason": "Insufficient data.", "last_util_pct": 0.0,
                "mom_pct": 0.0, "slope": 0.0, "signals_fired": []}
                
    sales = [m["total_scaled"] for m in monthly]
    last_util_pct = (sales[-1]/ath*100) if ath else 0.0
    mom_pct = ((sales[-1]-sales[-2])/sales[-2]*100) if sales[-2] > 0 else 0.0
    x = np.arange(len(sales), dtype=float)
    slope, _ = np.polyfit(x, sales, 1)
    
    sig1 = last_util_pct < CLOSURE_MAX_LAST_UTIL*100
    sig2 = mom_pct < CLOSURE_MAX_MOM_CHANGE*100
    sig3 = slope < 0
    
    signals_fired = []
    if sig1: signals_fired.append(f"last month at {last_util_pct:.1f}% of ATH")
    if sig2: signals_fired.append(f"MoM crash of {mom_pct:+.1f}%")
    if sig3: signals_fired.append(f"negative slope ({slope:+,.0f}/month)")
    
    should_close = bool(sig1 and sig2 and sig3)
    reason = (f"All 3 closure signals fired: {'; '.join(signals_fired)}." if should_close
              else f"{len(signals_fired)}/3 closure signals active." if signals_fired
              else "No closure signals.")
              
    return {
        "should_close": should_close, 
        "reason": reason,
        "last_util_pct": round(float(last_util_pct), 2), 
        "mom_pct": round(float(mom_pct), 2),
        "slope": round(float(slope), 2), 
        "signals_fired": signals_fired
    }

def compute_verdict(p1, p2, p3):
    scores = [p1["score"], p2["score"], p3["score"]]
    greens = scores.count("green"); ambers = scores.count("amber"); reds = scores.count("red")
    NAMES  = ["Sustained Volume","Growth Trajectory","Delivery Density"]
    red_p   = [NAMES[i] for i,s in enumerate(scores) if s=="red"]
    amber_p = [NAMES[i] for i,s in enumerate(scores) if s=="amber"]
    green_p = [NAMES[i] for i,s in enumerate(scores) if s=="green"]
    if greens == 3:
        return "EXPAND", f"All 3 pillars green ({', '.join(green_p)}). Expansion is justified."
    elif greens == 2 and ambers == 1:
        return "MONITOR CLOSELY", f"2 green ({', '.join(green_p)}), 1 amber ({', '.join(amber_p)}). Re-evaluate next month."
    elif reds == 0 and ambers == 3:
        return "MONITOR", "All 3 pillars amber. Re-evaluate in 2 months."
    elif reds > 0:
        return "DO NOT EXPAND", f"Red flag(s) on: {', '.join(red_p)}."
    else:
        return "DO NOT EXPAND", f"Only {greens} pillar(s) green. Need all 3 aligned."

def score_branch(branch_name, sales_path=DEFAULT_SALES_JSON, customers_path=DEFAULT_CUSTOMERS_JSON):
    sales_data = load_sales(sales_path)
    customers  = load_customers(customers_path)
    if branch_name not in sales_data:
        raise ValueError(f"Branch '{branch_name}' not found. Available: {list(sales_data.keys())}")
    
    closure = check_closure(sales_data[branch_name])
    p1 = score_pillar_1(sales_data[branch_name])
    p2 = score_pillar_2(sales_data[branch_name])
    p3 = score_pillar_3(branch_name, customers)
    
    if closure["should_close"]:
        verdict   = "CONSIDER CLOSURE"
        reasoning = closure["reason"]
    else:
        verdict, reasoning = compute_verdict(p1, p2, p3)
        
    return {
        "branch": branch_name,
        "closure_check": {
            "should_close": bool(closure["should_close"]),
            "signals_fired": closure["signals_fired"],
            "detail": closure["reason"],
            "last_util_pct": float(closure["last_util_pct"]),
            "mom_pct": float(closure["mom_pct"]), 
            "slope": float(closure["slope"])
        },
        "pillar_1_volume": {
            "score": p1["score"], 
            "detail": p1["detail"],
            "monthly_utilizations": p1["monthly_utilizations"]
        },
        "pillar_2_trajectory": {
            "score": p2["score"], 
            "detail": p2["detail"],
            "slope": float(p2.get("slope")) if p2.get("slope") is not None else None, 
            "r_squared": float(p2.get("r_squared")) if p2.get("r_squared") is not None else None
        },
        "pillar_3_delivery": {
            "score": p3["score"], 
            "detail": p3["detail"],
            "customer_count": int(p3["customer_count"]),
            "repeat_rate_pct": float(p3["repeat_rate_pct"]),
            "avg_order_value": float(p3["avg_order_value_scaled"])
        },
        "overall": verdict, 
        "recommendation": reasoning,
    }

def score_all_branches(sales_path=DEFAULT_SALES_JSON, customers_path=DEFAULT_CUSTOMERS_JSON):
    sales_data = load_sales(sales_path)
    results = []
    for branch_name in sales_data:
        results.append(score_branch(branch_name, sales_path, customers_path))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=None)
    parser.add_argument("--sales",     default=DEFAULT_SALES_JSON)
    parser.add_argument("--customers", default=DEFAULT_CUSTOMERS_JSON)
    args = parser.parse_args()
    if args.branch:
        r = score_branch(args.branch, args.sales, args.customers)
        print(json.dumps(r, indent=2))
    else:
        results = score_all_branches(args.sales, args.customers)
        for r in results:
            print(f"\n{r['branch']}: {r['overall']}")
            print(f"  P1={r['pillar_1_volume']['score']}  P2={r['pillar_2_trajectory']['score']}  P3={r['pillar_3_delivery']['score']}")
            print(f"  {r['recommendation']}")
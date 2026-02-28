import os, json
from fastapi import FastAPI, HTTPException, Query

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Conut Combo Optimization Service",
    description="Returns data-driven combo recommendations based on customer basket analysis.",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _load_results():
    path = os.path.join(BASE_DIR, "combo_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError("combo_results.json not found. Run the pipeline first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _load_pricing():
    path = os.path.join(BASE_DIR, "ai_combo_pricing_results.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/")
def health():
    return {"status": "ok", "service": "combo_optimization"}


@app.get("/combos")
def get_combos(limit: int = Query(10, description="Number of combos to return")):
    """
    Returns top combo recommendations ranked by lift score.
    Each combo includes: items, support %, confidence %, lift, and a plain-English interpretation.
    """
    try:
        data = _load_results()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    combos = data["combo_recommendations"][:limit]
    return {
        "total_combos_found": data["summary"]["unique_combos_found"],
        "combos": combos
    }


@app.get("/combos/top-items")
def get_top_items(limit: int = Query(15, description="Number of top items to return")):
    """
    Returns the most frequently purchased items across all orders.
    Useful for understanding what drives the most volume.
    """
    try:
        data = _load_results()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"top_items": data["top_items"][:limit]}


@app.get("/combos/summary")
def get_summary():
    """
    Returns summary stats: total baskets, unique products, avg basket size, combos found.
    """
    try:
        data = _load_results()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return data["summary"]


@app.get("/combos/pricing")
def get_pricing():
    """
    Returns AI-suggested discount prices per combo from generate_combos.py.
    Includes original price, promo price, and target volume to break even.
    """
    pricing = _load_pricing()
    if pricing is None:
        raise HTTPException(status_code=503,
            detail="ai_combo_pricing_results.csv not found. Run generate_combos.py first.")
    return {"combo_pricing": pricing}


@app.get("/combos/full-report")
def get_full_report():
    """Full combined report — used by the AI agent."""
    try:
        data = _load_results()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    pricing = _load_pricing()
    return {
        "summary":     data["summary"],
        "top_items":   data["top_items"][:15],
        "top_combos":  data["combo_recommendations"][:10],
        "pricing":     pricing or "Run generate_combos.py to get pricing suggestions"
    }
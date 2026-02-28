from fastapi import FastAPI, Query
from growth_service import get_crosssell_opportunity, get_branch_benchmarks, get_high_value_items, get_full_strategy

app = FastAPI(title="Conut Growth Strategy Service", version="1.0.0")

@app.get("/")
def health():
    return {"status": "ok", "service": "growth"}

@app.get("/growth/crosssell")
def crosssell():
    """Returns the 83% drink gap insight + best bundles + revenue opportunity."""
    return get_crosssell_opportunity()

@app.get("/growth/benchmarks")
def benchmarks():
    """Compares each branch against the top performer for coffee, frappes, shakes."""
    return get_branch_benchmarks()

@app.get("/growth/high-value-items")
def high_value(branch: str = Query(None, description="Filter by branch name")):
    """Returns highest revenue-per-unit drinks per branch — best items to promote."""
    return get_high_value_items(branch)

@app.get("/growth/full-strategy")
def full_strategy():
    """Full combined report — used by the AI agent."""
    return get_full_strategy()
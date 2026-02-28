import os, json
from fastapi import FastAPI, HTTPException, Query
from expansion_model import score_branch, score_all_branches, DEFAULT_SALES_JSON, DEFAULT_CUSTOMERS_JSON

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Conut Expansion Feasibility Service",
    description="3-pillar scoring system to evaluate branch expansion and closure risk.",
    version="1.0.0"
)

BRANCHES = ["Conut", "Conut - Tyre", "Conut Jnah", "Main Street Coffee"]

def _check_data():
    for path in [DEFAULT_SALES_JSON, DEFAULT_CUSTOMERS_JSON]:
        if not os.path.exists(path):
            raise HTTPException(status_code=503,
                detail=f"{os.path.basename(path)} not found. Run: python run.py update")

@app.get("/")
def health():
    return {"status": "ok", "service": "expansion_feasibility"}

@app.get("/expansion/branches")
def list_branches():
    """Returns valid branch names."""
    return {"branches": BRANCHES}

@app.get("/expansion/score/{branch_name}")
def get_branch_score(branch_name: str):
    """
    Runs the 3-pillar expansion score for a single branch.
    Returns: closure check, 3 pillar scores, overall verdict and recommendation.
    
    branch_name options: Conut, Conut - Tyre, Conut Jnah, Main Street Coffee
    """
    _check_data()
    try:
        return score_branch(branch_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/expansion/score-all")
def get_all_scores():
    """
    Runs the 3-pillar expansion score for ALL branches.
    Returns a summary list — used by the AI agent for full strategic overview.
    """
    _check_data()
    results = score_all_branches()
    summary = []
    for r in results:
        summary.append({
            "branch":         r["branch"],
            "overall":        r["overall"],
            "recommendation": r["recommendation"],
            "pillar_1":       r["pillar_1_volume"]["score"],
            "pillar_2":       r["pillar_2_trajectory"]["score"],
            "pillar_3":       r["pillar_3_delivery"]["score"],
            "closure_risk":   r["closure_check"]["should_close"],
        })
    return {"branches_scored": len(summary), "results": summary}

@app.get("/expansion/closure-risks")
def get_closure_risks():
    """
    Returns only branches with active closure signals.
    Useful for urgent alerts in the AI agent.
    """
    _check_data()
    results = score_all_branches()
    at_risk = [r for r in results if r["closure_check"]["signals_fired"]]
    return {
        "total_at_risk": len(at_risk),
        "branches": [
            {"branch": r["branch"],
             "should_close": r["closure_check"]["should_close"],
             "signals_fired": r["closure_check"]["signals_fired"],
             "detail": r["closure_check"]["detail"]}
            for r in at_risk
        ]
    }

@app.get("/expansion/full-report")
def get_full_report():
    """Full combined report for all branches — used by the AI agent."""
    _check_data()
    return {"results": score_all_branches()}
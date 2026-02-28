import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from forecast_service import InventoryForecaster

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Conut Forecasting Service",
    description="Predicts daily inventory needs per branch and item using XGBoost.",
    version="1.0.0"
)

# Load the model once when the server starts
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
forecaster = InventoryForecaster(MODEL_DIR)


# ── Request models ────────────────────────────────────────────────────────────
class SingleDayRequest(BaseModel):
    branch: str   # e.g. "Conut - Tyre"
    item: str     # e.g. "FULL FAT MILK"
    date: str     # e.g. "2026-03-12"

class DateRangeRequest(BaseModel):
    branch: str   # e.g. "Conut - Tyre"
    item: str     # e.g. "FULL FAT MILK"
    start_date: str  # e.g. "2026-03-12"
    end_date: str    # e.g. "2026-03-18"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Check that the forecasting service is running."""
    return {"status": "ok", "service": "forecasting"}


@app.post("/forecast/single")
def forecast_single_day(request: SingleDayRequest):
    """
    Predict inventory needed for a single item at a branch on a specific date.

    Example:
        POST /forecast/single
        { "branch": "Conut - Tyre", "item": "FULL FAT MILK", "date": "2026-03-12" }
    """
    result = forecaster.predict_single_day(request.branch, request.item, request.date)

    # forecast_service returns an error string if something goes wrong
    if isinstance(result, str) and result.startswith("Error"):
        raise HTTPException(status_code=400, detail=result)

    return {
        "branch": request.branch,
        "item": request.item,
        "date": request.date,
        "predicted_units": result
    }


@app.post("/forecast/range")
def forecast_date_range(request: DateRangeRequest):
    """
    Predict total inventory needed for an item at a branch over a date range.

    Example:
        POST /forecast/range
        { "branch": "Conut", "item": "CLASSIC CHIMNEY",
          "start_date": "2026-03-12", "end_date": "2026-03-18" }
    """
    result = forecaster.predict_date_range(
        request.branch, request.item, request.start_date, request.end_date
    )

    if isinstance(result, str) and result.startswith("Error"):
        raise HTTPException(status_code=400, detail=result)

    return {
        "branch": request.branch,
        "item": request.item,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "total_units": result["total_requested"],
        "daily_breakdown": result["daily_breakdown"]
    }


@app.get("/branches")
def list_branches():
    """Returns the valid branch names to use in requests."""
    return {
        "branches": [
            "Conut",
            "Conut - Tyre",
            "Conut Jnah",
            "Main Street Coffee"
        ]
    }
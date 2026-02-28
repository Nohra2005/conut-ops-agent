import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from forecast_service import InventoryForecaster

app = FastAPI(
    title="Conut Forecasting Service",
    description="Predicts daily inventory needs per branch and item using seasonal weighted formula.",
    version="2.0.0"
)

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
forecaster = InventoryForecaster(DATA_DIR)


class SingleDayRequest(BaseModel):
    branch: str  # e.g. "Conut - Tyre"
    item: str    # e.g. "FULL FAT MILK"
    date: str    # e.g. "2026-03-12"

class DateRangeRequest(BaseModel):
    branch: str
    item: str
    start_date: str
    end_date: str


@app.get("/")
def health():
    return {"status": "ok", "service": "forecasting", "version": "2.0.0"}


@app.post("/forecast/single")
def forecast_single(request: SingleDayRequest):
    result = forecaster.predict_single_day(request.branch, request.item, request.date)
    if isinstance(result, str) and result.startswith("Error"):
        raise HTTPException(status_code=400, detail=result)
    return {"branch": request.branch, "item": request.item, "date": request.date, "predicted_units": result}


@app.post("/forecast/range")
def forecast_range(request: DateRangeRequest):
    result = forecaster.predict_date_range(request.branch, request.item, request.start_date, request.end_date)
    if isinstance(result, str) and result.startswith("Error"):
        raise HTTPException(status_code=400, detail=result)
    return {
        "branch": request.branch, "item": request.item,
        "start_date": request.start_date, "end_date": request.end_date,
        "total_units": result["total_predicted"],
        "daily_breakdown": result["daily_breakdown"]
    }


@app.get("/forecast/items")
def list_items(branch: str = None):
    """List all available items, optionally filtered by branch."""
    return {"items": forecaster.list_items(branch)}


@app.get("/branches")
def list_branches():
    return {"branches": ["Conut", "Conut - Tyre", "Conut Jnah", "Main Street Coffee"]}
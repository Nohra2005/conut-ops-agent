import os
import json
import mlflow
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import LabelEncoder

app = FastAPI(
    title="Conut Staffing Prediction Service",
    description="MLflow-backed service to predict daily employee requirements based on inventory demand.",
    version="1.0.0"
)

MODEL_NAME = "employee_demand_model"
CHAMPION_ALIAS = "champion"

# In-memory cache to keep API responses under 50ms
model_cache = {"model": None, "le": None, "version": None}

def load_champion_model():
    """Fetches the current champion model and its label encoder from MLflow."""
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
        
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        
        # Download and reconstruct the label encoder
        run_id = version.run_id
        le_path = client.download_artifacts(run_id, "label_encoder.json")
        with open(le_path) as f:
            classes = json.load(f)["classes"]
        le = LabelEncoder()
        le.fit(classes)
        
        model_cache["model"] = model
        model_cache["le"] = le
        model_cache["version"] = version.version
        return True
    except Exception as e:
        print(f"Warning: Could not load champion model at startup. ({e})")
        return False

@app.on_event("startup")
def startup_event():
    load_champion_model()

@app.get("/")
def health():
    status = "ok" if model_cache["model"] else "model_not_loaded"
    return {
        "status": status, 
        "service": "staffing_prediction", 
        "champion_version": model_cache["version"]
    }

@app.post("/reload-model")
def reload_model():
    """Forces the API to fetch the latest champion from MLflow without restarting."""
    success = load_champion_model()
    if not success:
        raise HTTPException(status_code=503, detail="Failed to load model. Check MLflow.")
    return {"status": "reloaded", "champion_version": model_cache["version"]}

@app.get("/staffing/predict")
def predict_staffing(branch: str, demand: float, date: str = None):
    """
    Predicts the number of employees needed.
    - branch: The branch name (e.g., 'Conut Jnah')
    - demand: Total projected inventory items needed
    - date: (Optional) YYYY-MM-DD. Defaults to today.
    """
    if not model_cache["model"]:
        success = load_champion_model()
        if not success:
            raise HTTPException(status_code=503, detail="Model not initialized. Run pipeline.py first.")
            
    le = model_cache["le"]
    model = model_cache["model"]
    
    if branch not in le.classes_:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown branch '{branch}'. Known branches: {list(le.classes_)}"
        )
        
    try:
        d = datetime.strptime(date, "%Y-%m-%d") if date else datetime.today()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    branch_enc = le.transform([branch])[0]
    
    # Construct exact feature map expected by pipeline.py
    X = pd.DataFrame([{
        "branch_enc": branch_enc,
        "total_items": demand,
        "dayofweek": d.weekday(),
        "month": d.month,
        "is_weekend": int(d.weekday() in [4, 5]),
    }])
    
    # Cast to standard python float to prevent FastAPI NumPy serialization errors
    predicted_employees = float(model.predict(X)[0])
    
    return {
        "branch": branch,
        "date": d.strftime("%Y-%m-%d"),
        "demand_items": float(demand),
        "predicted_employees_needed": round(predicted_employees, 1),
        "model_version": model_cache["version"]
    }

import logging
import json
import joblib
import hashlib
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# Configures a dual-output logging system. Every time the API runs or makes a prediction, a record is written to both the console (for active monitoring) and a file called 'api_audit.log'.
# This is a critical governance requirement in healthcare to ensure auditability of AI decisions.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("api_audit.log"), logging.StreamHandler()]
)

# Initializes the FastAPI application. This is the core web framework that will listen for incoming HTTP requests from the hospital dashboards and route them to our models.

app = FastAPI(
    title="Hospital Risk & Revenue Intelligence API",
    description="Real-time predictive models for clinical triage and claim outcome forecasting.",
    version="1.0"
)

# When the API starts up (before any requests are received), it loads the trained 
# Random Forest models (.pkl files) and their feature schemas (.json files) into memory.
# The schemas tell the API exactly which features the model expects and how to map 
# the numeric output (e.g., 0, 1, 2) back to human-readable labels (e.g., "High", "Rejected").

try:
    risk_model = joblib.load('risk_model.pkl')
    claim_model = joblib.load('claim_model.pkl')
    with open('risk_feature_schema.json', 'r') as f: risk_schema = json.load(f)
    with open('claim_feature_schema.json', 'r') as f: claim_schema = json.load(f)
    logging.info("Models and schemas loaded successfully.")
except Exception as e:
    logging.error(f"Startup Error: Could not load models/schemas. Details: {e}")

# Pydantic enforces strict data validation. If a dashboard sends a request where 
# 'age' is a string ("sixty-five") instead of a float (65.0), the API will automatically 
# reject the request with a clear error message. This prevents bad data from crashing the mode

class RiskRequest(BaseModel):
    age: float
    length_of_stay_hours: float
    visit_frequency: int
    avg_los_per_patient: float
    days_since_registration: int
    visit_month: int
    visit_dayofweek: int
    chronic_flag: int
    department_enc: int
    visit_type_enc: int
    gender_enc: int
    city_enc: int

class ClaimRequest(BaseModel):
    billed_amount: float
    provider_rejection_rate: float
    high_billed_flag: int
    department_enc: int
    insurance_provider_enc: int
    visit_type_enc: int
    age: float
    length_of_stay_hours: float
    chronic_flag: int

# Defines the standardized structure for what the API will send *back* to the user.
# Every response will contain the prediction ("High", "Rejected", etc.) and a metadata block

class PredictionResponse(BaseModel):
    prediction: str
    metadata: dict

# Creates a unique, irreversible digital fingerprint (SHA-256 hash) of the input data.
# If a doctor asks, "Why did the AI flag this patient as High Risk?", the IT team can 
# use this hash to find the exact snapshot of data the model saw at that exact millisecond.

def generate_hash(data: dict) -> str:
    """Generates a SHA-256 hash of the input features for audit traceability."""
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

# A lightweight endpoint used by hospital IT load balancers (like AWS or NGINX) 
# to constantly check if the API is alive and ready to accept traffic

@app.get("/health")
def health_check():
    """IT Ops Health Check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# The core endpoint for Model A (Clinical Risk). It accepts a JSON payload validated 
# against RiskRequest, converts it into a Pandas DataFrame in the exact column order 
# defined by the schema, runs the Random Forest prediction, logs the event, and returns the result.

@app.post("/predict/risk", response_model=PredictionResponse)
def predict_risk(request: RiskRequest):
    """Predicts Clinical/Operational Risk (High, Medium, Low)"""
    input_data = request.dict()
    input_hash = generate_hash(input_data)
    
    # Format for model
    df = pd.DataFrame([input_data])[risk_schema['features']]
    
    # Predict & Map
    pred_idx = int(risk_model.predict(df)[0])
    pred_label = risk_schema['target_mapping'].get(str(pred_idx), "Unknown")
    
    # Audit Log
    logging.info(f"Model: Risk | Hash: {input_hash} | Pred: {pred_label}")
    
    return {
        "prediction": pred_label,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0",
            "input_hash": input_hash
        }
    }


# The core endpoint for Model B (Claim Outcome). It functions identically to the risk endpoint,
# but routes data to the claim_model instead, predicting Paid, Pending, or Rejected.


@app.post("/predict/claim", response_model=PredictionResponse)
def predict_claim(request: ClaimRequest):
    """Predicts Claim Outcome (Paid, Pending, Rejected)"""
    input_data = request.dict()
    input_hash = generate_hash(input_data)
    
    # Format for model
    df = pd.DataFrame([input_data])[claim_schema['features']]
    
    # Predict & Map
    pred_idx = int(claim_model.predict(df)[0])
    pred_label = claim_schema['target_mapping'].get(str(pred_idx), "Unknown")
    
    # Audit Log
    logging.info(f"Model: Claim | Hash: {input_hash} | Pred: {pred_label}")
    
    return {
        "prediction": pred_label,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0",
            "input_hash": input_hash
        }
    }

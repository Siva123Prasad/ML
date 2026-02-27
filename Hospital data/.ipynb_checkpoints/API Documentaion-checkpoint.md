# API Documentation: Hospital Intelligence Endpoints

## 1. System Health Check
**Endpoint:** `GET /health`  
**Purpose:** Used by load balancers and IT ops to verify the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-23T15:30:00.123456"
}


## 2. Visit Risk Prediction
Endpoint: POST /predict/risk
Purpose: Predicts whether a visit is 'High', 'Medium', or 'Low' risk.

Sample curl Request:


curl -X POST "http://localhost:8000/predict/risk" \
     -H "Content-Type: application/json" \
     -d '{
           "age": 65.5,
           "length_of_stay_hours": 48.0,
           "visit_frequency": 6,
           "avg_los_per_patient": 42.5,
           "days_since_registration": 150,
           "visit_month": 11,
           "visit_dayofweek": 2,
           "chronic_flag": 1,
           "department_enc": 2,
           "visit_type_enc": 1,
           "gender_enc": 0,
           "city_enc": 1
         }'

Sample response:

{
  "prediction": "High",
  "metadata": {
    "timestamp": "2026-02-23T15:32:11.890",
    "model_version": "1.0",
    "input_hash": "a4b9c8d7... (SHA-256 hash)"
  }
}

3. Claim Outcome Prediction
Endpoint: POST /predict/claim
Purpose: Predicts if a claim will be 'Paid', 'Pending', or 'Rejected' before submission.

Sample curl Request
curl -X POST "http://localhost:8000/predict/claim" \
     -H "Content-Type: application/json" \
     -d '{
           "billed_amount": 75000.00,
           "provider_rejection_rate": 0.156,
           "high_billed_flag": 1,
           "department_enc": 3,
           "insurance_provider_enc": 2,
           "visit_type_enc": 0,
           "age": 45.0,
           "length_of_stay_hours": 12.0,
           "chronic_flag": 0
         }'

Sample response:
{
  "prediction": "Rejected",
  "metadata": {
    "timestamp": "2026-02-23T15:35:44.112",
    "model_version": "1.0",
    "input_hash": "f8e7d6c5... (SHA-256 hash)"
  }
}

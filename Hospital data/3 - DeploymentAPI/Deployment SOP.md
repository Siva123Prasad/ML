## Start the Development Server
Launch the API using Uvicorn with the `--reload` flag. This flag enables automatic restarts whenever you edit `main.py`, which is useful during development:
```bash
uvicorn main:app --reload
```

You should see the following output confirming the server is running:
```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Verify the API is Running
Open your web browser and navigate to:
```text
http://127.0.0.1:8000/health
```

You should receive this JSON response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-23T15:30:00.123456"
}
```

### Interactive Testing via Swagger UI
FastAPI automatically generates an interactive testing dashboard. Navigate to:
```text
http://127.0.0.1:8000/docs
```

You will see the full Swagger UI with all three endpoints listed:
- `GET /health` — Health check
- `POST /predict/risk` — Visit Risk prediction
- `POST /predict/claim` — Claim Outcome prediction

To test an endpoint:
1. Click on the green endpoint bar (e.g., `POST /predict/risk`) to expand it.
2. Click the **"Try it out"** button on the right.
3. A pre-filled JSON payload will appear in the text box.
4. Click the blue **"Execute"** button.
5. Scroll down to the **"Server response"** section to view the prediction and metadata.

### Testing via curl (Optional)
To test from the command line without a browser:

**Visit Risk Prediction:**
```bash
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
```

**Claim Outcome Prediction:**
```bash
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
```

### To Stop the Server
Press `CTRL + C` in the terminal at any time to shut down the server gracefully.

---

## Production Deployment

### Multi-Worker Production Server
For hospital production environments that receive concurrent requests from multiple dashboards and billing systems, run the API behind **Gunicorn** with Uvicorn workers:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

**What this means:**
- `-w 4` — Runs 4 parallel worker processes to handle concurrent requests.
- `-k uvicorn.workers.UvicornWorker` — Uses Uvicorn as the async worker type inside Gunicorn.
- `-b 0.0.0.0:8000` — Binds to all network interfaces on port 8000.

Install Gunicorn before using it:
```bash
pip install gunicorn
```

### Security Considerations
- The API should be deployed **behind the hospital's internal firewall or VPN** only. It must not be exposed to the public internet, as it processes sensitive clinical and financial feature combinations.
- API keys or OAuth2 token authentication should be added in a future version before connecting to live EMR (Electronic Medical Record) systems.
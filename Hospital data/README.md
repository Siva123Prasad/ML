# Hospital risk and revenue intelligence platform

An end-to-end machine learning and data engineering pipeline designed to predict clinical patient risk at triage and intercept insurance claim rejections before submission, recovering an estimated ₹4–7 crore in annual revenue leakage.

This project was developed as a comprehensive capstone project demonstrating full-lifecycle data science skills: SQL analytics, exploratory data analysis, predictive modeling, API deployment, and automated drift monitoring.

---

## Table of contents
1. [Business problem](#business-problem)
2. [Project architecture](#project-architecture)
3. [Key outcomes and ROI](#key-outcomes-and-roi)
4. [Deployed AI models](#deployed-ai-models)
5. [Repository structure](#repository-structure)
6. [Local setup and installation](#local-setup-and-installation)
7. [API usage and endpoints](#api-usage-and-endpoints)

---

## Business problem
Hospitals face two critical challenges that operate in silos:
1. Clinical bottlenecks: Unpredictable patient acuity leads to inefficient ICU bed allocation and reactive nursing assignments, driving up average length of stay.
2. Revenue leakage: An average 15.2% insurance claim rejection rate creates a massive financial deficit. High-value claims are frequently denied due to complex payer thresholds that human coders cannot manually catch at scale.

The solution: A dual-model AI platform integrated into the hospital's electronic medical record and billing systems via a real-time REST API.

---

## Project architecture
This project was built across 6 distinct phases:

1. SQL analytics layer: Computed operational KPIs like length of stay and high-risk concentration alongside financial metrics like provider rejection rates and realization gaps.
2. Data engineering: Cleaned 25,000 visit records, engineered 12 novel features including patient tenure and provider hostility metrics, and handled missing values via statistical imputation.
3. Model development: Built two distinct classifiers using time-based splitting to prevent data leakage. Applied SMOTE and balanced class weighting to prioritize recall on minority, high-impact classes like high risk and rejected claims.
4. Evaluation and fairness: Validated models via classification reports, feature importance extraction, and rigorous demographic/payer fairness audits to ensure compliance.
5. Production API: Deployed the models via an asynchronous REST API with strict input validation schemas and SHA-256 tamper-evident audit logging.
6. Governance and MLOps: Automated data validation and Kolmogorov-Smirnov statistical tests for feature drift detection to trigger retraining schedules.

---

## Key outcomes and ROI

### Operational impact
* Targeted triage: Identified the ICU (20.8%) and ER (20.7%) as the primary centers of high-risk volatility. Model A now allows proactive bed allocation before the patient reaches the floor.
* Length of stay reduction: Highlighted neurology (19.72 hrs) and orthopedics as critical throughput bottlenecks, largely driven by the 50.3% of visits belonging to chronic patients.

### Financial impact (ROI)
* ₹521.7 million revenue exposure: Mapped the hospital's total financial exposure across four major payers, with SecureLife carrying the highest rejection risk at 15.69%.
* ₹4–7 crore interception potential: By routing claims predicted as rejected by model B to a specialized coder pre-submission, the hospital can bypass the payer denial process entirely. 
* Zero-approval audit: Automatically flagged 190 top-decile visits billed over ₹37k that suffered total revenue leakage, generating an immediate high-value queue for collections.

---

## Deployed AI models

### Model A: Visit risk classifier
* Goal: Predict clinical risk at the time of patient registration.
* Algorithm: Random forest classifier with max depth of 10.
* Top drivers: Length of stay, average patient length of stay, age, and chronic condition flag.
* Leakage control: All post-triage financial data was strictly excluded from training.

### Model B: Claim outcome classifier
* Goal: Predict claim status prior to payer submission.
* Algorithm: Random forest classifier with max depth of 12.
* Top drivers: Billed amount, high-billed flag, and provider rejection rate.
* Leakage control: Post-adjudication variables like approved amount and payment days were excluded to simulate real-world pre-submission data states.

---

## Repository structure

```text
├── 1_analytics_and_eda/
│   ├── 00_sql_queries.ipynb          
│   ├── 01_eda.ipynb                  
│   └── data_quality_report.md        
├── 2_model_development/
│   ├── 02_risk_model.ipynb           
│   ├── 03_claim_model.ipynb          
│   ├── 04_evaluation.ipynb           
│   └── model_card.md                 
├── 3_deployment_api/
│   ├── main.py                       
│   ├── risk_model.pkl                
│   ├── claim_model.pkl               
│   ├── feature_schema.json           
│   ├── deployment_guide.md           
│   └── api_audit.log                 
├── 4_monitoring_mlops/
│   ├── monitoring.py                 
│   ├── drift_detection_report.html   
│   └── governance.md                 
└── healthcare_insights_report.docx   
```

---

## Local setup and installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hospital-risk-intelligence.git
cd hospital-risk-intelligence/3_deployment_api
```

2. Install dependencies (requires python 3.9+):
```bash
pip install fastapi uvicorn pandas scikit-learn joblib pydantic scipy
```

3. Launch the API server:
```bash
uvicorn main:app --reload
```

4. Access the interactive dashboard:
Open your browser and navigate to http://127.0.0.1:8000/docs to test the live models via the Swagger UI.

---

## API usage and endpoints

### 1. Health check
`GET /health`
Returns the status of the API for load balancers.

### 2. Predict clinical risk
`POST /predict/risk`
```json
{
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
}
```

### 3. Predict claim rejection
`POST /predict/claim`
```json
{
  "billed_amount": 75000.00,
  "provider_rejection_rate": 0.156,
  "high_billed_flag": 1,
  "department_enc": 3,
  "insurance_provider_enc": 2,
  "visit_type_enc": 0,
  "age": 45.0,
  "length_of_stay_hours": 12.0,
  "chronic_flag": 0
}
```


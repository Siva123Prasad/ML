# AI Governance and Compliance Policy
## Hospital Risk & Revenue Intelligence System

## 1. System purpose & scope
This AI system provides decision-support predictions for:
1. **Clinical Operations:** Predicting triage risk (`High`, `Medium`, `Low`).
2. **Revenue Cycle Management (RCM):** Predicting insurance claim outcomes (`Paid`, `Pending`, `Rejected`) prior to submission.
**Scope Limitation:** These models are advisory. They **do not** replace clinical judgment or override human medical coders.

## 2. Audit logging and traceability
In compliance with healthcare IT standards, every prediction made by the FastAPI service is permanently recorded in `api_audit.log`. 
- **Traceability:** Each log entry includes a precise timestamp, the model version utilized, the exact prediction label, and an irreversible SHA-256 hash (`input_hash`) of the patient/claim data. 
- **Dispute Resolution:** If an insurer disputes a rejected claim prediction, or a physician questions a clinical risk label, IT can instantly trace the exact feature inputs used by matching the hash.

## 3. Limitations and assumptions
- **Assumption of Payer Stability:** The claim prediction model (Model B) assumes historical rejection patterns remain constant. It cannot predict rejections based on *new* corporate policies enacted by insurers after the training date.
- **Missing Clinical Context:** The risk model (Model A) relies on high-level operational features (age, department, frequency). It does not ingest unstructured physician notes or live vital signs. Therefore, it is an operational proxy for risk, not a diagnostic medical device.
- **Data Capture Delays:** Realization rate features assume accurate ETL pipelines. If `approved_amount` data is structurally delayed by the payer, the data pipeline must handle imputation appropriately.

## 4. Retraining strategy & Model lifecycle
To prevent degradation, the models are governed by the following lifecycle policy:
- **Continuous Monitoring:** The `06_monitoring.py` script runs weekly via cron job, generating Evidently AI drift reports.
- **Threshold for Retraining:** If data drift is detected in more than 20% of the top 5 driving features (e.g., a massive spike in ER visits), or if overall recall for minority classes drops below 70%, the Data Science team is alerted automatically.
- **Scheduled Refresh:** Regardless of drift, both models must be retrained semi-annually (every 6 months) using the trailing 12 months of hospital data to ensure seasonal trends and updated pricing are captured.
- **Version Control:** All new models will iterate sequentially (e.g., v1.1, v2.0) and be pushed to the FastAPI service with zero downtime using Uvicorn worker restarts.

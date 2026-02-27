# Data & Concept Drift Report Summary

## 1. Executive summary
An automated drift detection script (`monitoring.py`) was executed comparing the baseline training dataset (Reference) against the most recent 5,000 hospital visits (Current). The purpose is to ensure the AI models remain accurate as hospital operations evolve.

## 2. Automated Validation Checks
Before assessing statistical drift, hard boundary rules were checked on the incoming data stream:
- **Missing Values Check:** ✅ Passed. No missing critical variables (e.g., `billed_amount`).
- **Numeric Range Violations:** ✅ Passed. No negative ages or impossible lengths of stay.
- **Unseen Categories:** ✅ Passed. No new, untrained department codes or insurance providers were detected.

## 3. Statistical Feature Drift (Covariate Shift)
The system evaluated key features for statistical distribution changes using the Kolmogorov-Smirnov test (p-value threshold = 0.05).
- **`age`:** No drift detected. Patient demographic distributions remain stable.
- **`billed_amount`:** Minor right-skew shift observed. Average procedure costs appear to be rising slightly over the last quarter, likely due to inflation. *Action: Monitor next month; if drift exceeds 10%, Model B (Claims) must be retrained.*
- **`department_enc`:** No drift detected. Volume across ER, ICU, and Cardiology remains consistent.

## 4. Concept Drift Risk (Target Shift)
- **Claim Outcomes:** The underlying rejection rates (`provider_rejection_rate`) have remained stable at ~15%. However, if an insurer issues a new policy rule next year, this will trigger severe concept drift. Model B's performance will plummet. 
- **Risk Profiles:** The ratio of High vs. Low risk visits is stable. No concept drift observed in clinical triage.

*A detailed, interactive visual dashboard of these metrics is available locally at `drift_detection_report.html`.*

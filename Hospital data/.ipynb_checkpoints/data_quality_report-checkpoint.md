# Data Quality report

## 1. Dataset overview

| Table | Rows | Columns | Granularity | Date Range |
|---|---|---|---|---|
| patients.csv | 5,000 | 7 | One row per patient | Jan 2025 – Jan 2026 |
| visits.csv | 25,000 | 8 | One row per hospital visit | Jan 2025 – Jan 2026 |
| billing.csv | 25,000 | 7 | One row per billing event | Jan 2025 – Jan 2026 |
| **Combined** | **25,000** | **30** | Visit-level analytical record | Jan 2025 – Jan 2026 |

---

## 2. Missing value analysis

### Summary

| Column | Missing | % | Severity |
|---|---|---|---|
| approved_amount | 1,318 | 5.27% | Medium |
| payment_days | 790 | 3.16% | Low–Medium |
| length_of_stay_hours | 0 | 0.00% | None |

### approved_amount — by claim status

| Claim Status | Missing Count | % of Missing |
|---|---|---|
| Paid | 817 | 62.0% |
| Pending | 301 | 22.8% |
| Rejected | 200 | 15.2% |


- Paid (61.9%): Approved amount exists in payer system but was not written back to hospital DB — ETL gap
- Pending (22.8%): Claim in-flight; approval not yet received — expected (Missing Not At Random)
- Rejected (15.2%): No approval issued — structurally correct

Treatment:
- Rejected → impute approved_amount = 0
- Paid/Pending → impute with median grouped by department + insurance_provider

### approved_amount — by department

| Department | Missing Count |
|---|---|
| ER | 230 |
| Cardiology | 224 |
| General | 224 |
| Neurology | 215 |
| Orthopedics | 215 |
| ICU | 210 |

Spread is uniform across departments — confirms a systemic ETL issue, not department-specific.

### payment_days — by claim status

| Claim Status | Missing Count | % of Missing |
|---|---|---|
| Paid | 459 | 58.1% |
| Pending | 208 | 26.3% |
| Rejected | 123 | 15.6% |

---

## 3. Referential integrity checks

| Check | Result | Status |
|---|---|---|
| Duplicate patient_id | 0 | PASS |
| Visits with no patient record | 0 | PASS |
| Visits without billing record | 0 | PASS |
| Billing without visit record | 0 | PASS |
| Negative length_of_stay_hours | 0 | PASS |
| Negative billed_amount | 0 | PASS |
| Negative payment_days | 0 | PASS |
| Invalid risk_score values | 0 | PASS |
| Invalid claim_status values | 0 | PASS |

---

## 4. Outlier detection 

### billed_amount

| Metric | Value |
|---|---|
| Range | Rs.500 to Rs.88,539 |
| Mean | Rs.20,871 |
| Median | Rs.19,645 |
| Std Dev | Rs.12,606 |
| IQR Upper Fence | Rs.53,621 |
| Outliers above fence | 373 rows (1.49%) |

Mean is much higher than median — right-skewed distribution from high-value
procedures. Outliers are genuine (complex surgeries, multi-day procedures), not data errors.

Add high_billed_flag (top 10%). Cap at Rs.53,621 for logistic regression only.

### payment_days

| Metric | Value |
|---|---|
| Range | 1 to 55 days |
| Mean / Median | 13.0 / 13.0 days |
| IQR Upper Fence | 30 days |
| Outliers above fence | 509 rows (2.04%) |

509 visits had payment delays beyond 30 days. These represent
genuine slow-paying or disputing insurers — high financial risk events, not errors.

Add long_payment_flag = (payment_days > 30). Keep raw values for tree models.

### length_of_stay_hours

| Metric | Value |
|---|---|
| Range | 0.5 to 78.4 hours |
| Mean / Median | 19.6 / 18.2 hours |
| IQR Upper Fence | 53.3 hours |
| Outliers above fence | 256 rows (1.02%) |

Stays above 53 hours (~2.2 days) are genuine extended inpatient
admissions (complex surgery, ICU). Not data errors.

Add long_stay_flag = (length_of_stay_hours > 53).

---

## 5. Distribution Analysis

### By department

| Department | Visits | Avg LOS (hrs) | Avg Billed (Rs) | High Risk % |
|---|---|---|---|---|
| Cardiology | 4,159 | 19.6 | 20,695.18 | 18.99% |
| ER | 4,220 | 19.53 | 21,015.87 | 20.66% |
| General | 4,228 | 19.43 | 20,608.2 | 19.84% |
| ICU | 4,064 | 19.36 | 20,855.75 | 20.79% |
| Neurology | 4,165 | 19.72 | 20,962.8 | 20.31% |
| Orthopedics | 4,164 | 19.66 | 21,088.25 | 20.22% |

### By visit type

| Visit Type | Count | % |
|---|---|---|
| ER | 8,382 | 33.5% |
| OPD | 8,381 | 33.5% |
| ICU | 8,237 | 32.9% |

Near-perfect balance (ER/ICU/OPD ~33% each). visit_type will be a stable feature.

### By insurance provider

| Provider | Visits | Avg Billed (Rs) | Rejection Rate | Avg Payment Days |
|---|---|---|---|---|
| CareOne | 6,283 | 20,803.44 | 14.87% | 13.03 |
| HealthPlus | 6,220 | 20,929.38 | 14.97% | 13.08 |
| MediCareX | 6,532 | 20,604.89 | 15.25% | 13.01 |
| SecureLife | 5,965 | 21,171.67 | 15.69% | 13.08 |

SecureLife has the highest avg bill AND rejection rate — highest net revenue risk.
All providers similar on payment speed (~13 days); bottleneck is approval quality not collections.

### By City

| City | Visits | Avg Billed (Rs) |
|---|---|---|
| Bangalore | 4,205 | 21,044.48 |
| Chennai | 3,975 | 20,778.88 |
| Delhi | 4,107 | 20,891.34 |
| Hyderabad | 4,370 | 20,846.17 |
| Mumbai | 4,122 | 20,962.58 |
| Pune | 4,221 | 20,699.97 |

### Patient demographics

| Metric | Value |
|---|---|
| Age Range | 1 – 90 years |
| Mean Age | 44.6 years |
| Chronic Condition Patients | 2,524 (50.5%) |

---

## 6. Target Variable Distribution

### Model A — risk_score

| Class | Count | % | Note |
|---|---|---|---|
| Low | 12,470 | 49.88% | Acceptable |
| Medium | 7,496 | 29.98% | Acceptable |
| High | 5,034 | 20.14% | Minority — apply SMOTE or class_weight |

### Model B — claim_status

| Class | Count | % | Note |
|---|---|---|---|
| Paid | 14,940 | 59.76% | Acceptable |
| Pending | 6,263 | 25.05% | Acceptable |
| Rejected | 3,797 | 15.19% | Minority — highest business priority |

---

## 7. Feature engineering summary

| Feature | Source | Business Rationale |
|---|---|---|
| visit_frequency | visits | High-frequency = likely chronic/recurring patient |
| avg_los_per_patient | visits | Patient-level LOS profile; flags heavy resource users |
| days_since_registration | patients+visits | Patient tenure; negative = pre-registration visit (data gap) |
| provider_rejection_rate | billing+patients | Insurer behaviour signal; predictive of claim outcome |
| visit_month | visits | Seasonal demand patterns |
| visit_dayofweek | visits | Weekend vs weekday admission patterns |
| visit_quarter | visits | Quarterly operational/financial cycles |
| realization_rate | billing | approved/billed ratio; measures revenue capture per visit |
| chronic_flag | patients | Clinical risk factor; chronic patients = higher risk score |
| high_billed_flag | billing | Top-decile billing; revenue leakage risk marker |
| long_stay_flag | visits | LOS > 53 hrs; proxy for clinical complexity |
| long_payment_flag | billing | payment_days > 30; proxy for collections difficulty |

---

## 8. Overall risk summary

| Risk | Severity | Action |
|---|---|---|
| approved_amount missing 5.27% | Medium | Impute by claim_status + dept median |
| payment_days missing 3.16% | Low | Add binary flag + impute for Paid only |
| Negative days_since_registration | Low | Flag as pre_registration_visit |
| High Risk class only 20% | Medium | class_weight=balanced + SMOTE if needed |
| Rejected claim only 15% | High | SMOTE + monitor recall closely in Phase 4 |
| billed_amount right-skewed | Low | high_billed_flag + cap for linear models |
| payment_days outliers 2.1% | Low | long_payment_flag + keep raw for trees |

--- 
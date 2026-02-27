import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import datetime

print("Loading data for Drift Analysis...")
try:
    df = pd.read_csv('model_table.csv')
except FileNotFoundError:
    print("❌ Error: model_table.csv not found.")
    exit()

# Simulate Reference (Train) vs Current (Production)
df_reference = df.head(15000)
df_current = df.tail(5000)

features_to_monitor = [
    'age', 'length_of_stay_hours', 'billed_amount', 
    'visit_frequency', 'avg_los_per_patient', 'realization_rate'
]

html_content = f"""
<html>
<head>
    <title>Hospital AI Drift Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #34495e; color: white; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .alert {{ color: #e74c3c; font-weight: bold; }}
        .safe {{ color: #27ae60; font-weight: bold; }}
        .header-meta {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Hospital AI Drift Detection Report</h1>
    <p class="header-meta">Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p class="header-meta">Phase 6: Monitoring and Governance</p>

    <h2>1. Data Validation Checks (Current Stream)</h2>
    <ul>
"""

print("Running validation checks...")
issues = 0
nulls = df_current[features_to_monitor].isnull().sum().sum()
if nulls > 0:
    html_content += f"<li class='alert'>⚠️ Found {nulls} missing values in current data stream!</li>"
    issues += 1
else:
    html_content += "<li class='safe'>✅ No missing values detected in critical features.</li>"

if (df_current['age'] < 0).any() or (df_current['length_of_stay_hours'] < 0).any():
    html_content += "<li class='alert'>⚠️ Negative clinical values detected (Age or LOS)!</li>"
    issues += 1
else:
    html_content += "<li class='safe'>✅ All numeric ranges are within logical clinical bounds.</li>"

html_content += """
    </ul>
    <h2>2. Statistical Feature Drift (Kolmogorov-Smirnov Test)</h2>
    <p>Compares the distribution of the original training data (Reference) against the latest 5,000 hospital visits (Current). A p-value < 0.05 indicates statistically significant drift.</p>
    <table>
        <tr>
            <th>Feature Name</th>
            <th>Reference Mean</th>
            <th>Current Mean</th>
            <th>KS p-value</th>
            <th>Status</th>
        </tr>
"""


print("Calculating Statistical Drift...")
for feature in features_to_monitor:
    ref_data = df_reference[feature].dropna()
    cur_data = df_current[feature].dropna()
    
    # Calculate means
    ref_mean = round(ref_data.mean(), 2)
    cur_mean = round(cur_data.mean(), 2)
    
    # Kolmogorov-Smirnov test for distribution drift
    stat, p_value = ks_2samp(ref_data, cur_data)
    
    # Determine Status
    if p_value < 0.05:
        status = "<span class='alert'>DRIFT DETECTED</span>"
    else:
        status = "<span class='safe'>Stable</span>"
        
    html_content += f"""
        <tr>
            <td>{feature}</td>
            <td>{ref_mean}</td>
            <td>{cur_mean}</td>
            <td>{p_value:.4f}</td>
            <td>{status}</td>
        </tr>
    """

html_content += """
    </table>
    
    <h2>3. Governance Recommendations</h2>
    <p>If significant drift is detected in high-impact features (e.g., <b>billed_amount</b> or <b>length_of_stay_hours</b>), Model B (Claim Outcomes) should be flagged for retraining using the latest trailing 12 months of hospital data to prevent revenue leakage due to shifting payer behaviors.</p>
</body>
</html>
"""

with open('drift_detection_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ SUCCESS: drift_detection_report.html generated!")

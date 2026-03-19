import json
import re
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from groq import Groq

client = Groq(api_key="GROQ_API_KEY")
rf_model = joblib.load(r'C:\Users\ADMIN\Documents\ML Self learn\ML\Hospital data\3 - DeploymentAPI\risk_model.pkl')

# --- Rebuild X_test (same as your training notebook) ---
df = pd.read_csv(r'C:\Users\ADMIN\Documents\ML Self learn\ML\Hospital data\1 - Analytics and EDA\model_table.csv')
df['visit_date'] = pd.to_datetime(df['visit_date'])

le = LabelEncoder()
y = pd.Series(le.fit_transform(df['risk_score']))

features = [
    'age', 'length_of_stay_hours', 'visit_frequency', 'avg_los_per_patient',
    'days_since_registration', 'visit_month', 'visit_dayofweek', 'chronic_flag',
    'department_enc', 'visit_type_enc', 'gender_enc', 'city_enc'
]

X = df[['visit_date'] + features].copy()
X.fillna(X.median(numeric_only=True), inplace=True)
X = X.sort_values('visit_date')

split_idx = int(len(X) * 0.8)
X_test = X.iloc[split_idx:].drop(columns=['visit_date'])
# -------------------------------------------------------

def get_triage_decision(patient_dict, risk_score, predicted_class):
    prompt = f"""
You are an emergency department triage assistant.

Patient data: {patient_dict}
Predicted risk class: {predicted_class}
Probability of High risk: {risk_score:.3f}

Respond ONLY with valid JSON in this exact format:
{{
  "urgency_level": "emergent or urgent or non_urgent",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "triage_note": "1-2 sentence explanation of urgency and key risk factors."
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
    return json.loads(raw)

results = []
class_map = {0: "High", 1: "Low", 2: "Medium"}

for i in range(20):
    row = X_test.iloc[i]
    # Use a one-row DataFrame so sklearn sees feature names
    row_df = row.to_frame().T

    patient_dict = row.to_dict()

    proba = rf_model.predict_proba(row_df)[0]     # array like [p_high, p_low, p_med]
    risk_score = float(proba[0])                  # probability of High risk

    predicted_idx = rf_model.predict(row_df)[0]   # 0,1,2
    predicted_class = class_map[predicted_idx]
    # patient_dict = row.to_dict()
   # proba = rf_model.predict_proba(row.values.reshape(1, -1))
    # risk_score = float(proba)
    # predicted_class = class_map[rf_model.predict(row.values.reshape(1, -1))]

    decision = get_triage_decision(patient_dict, risk_score, predicted_class)

    results.append({
        "patient_id": f"P{1000 + i}",
        "features": {k: round(v, 2) if isinstance(v, float) else int(v) for k, v in patient_dict.items()},
        "risk_score": round(risk_score, 3),
        "predicted_class": predicted_class,
        "urgency_level": decision["urgency_level"],
        "recommended_actions": decision["recommended_actions"],
        "triage_note": decision["triage_note"],
        "status": "pending"
    })
    print(f"✓ Patient P{1000+i} done — {predicted_class} / {decision['urgency_level']}")

with open('triage_decisions.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ triage_decisions.json saved with 20 patients.")
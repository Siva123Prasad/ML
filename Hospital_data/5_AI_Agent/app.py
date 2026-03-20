import json
import streamlit as st

st.set_page_config(page_title="Hospital Triage Agent", layout="wide")

# Load pre-computed decisions
with open('triage_decisions.json', 'r') as f:
    patients = json.load(f)

# Sort by urgency priority
urgency_order = {"Emergency": 0, "Urgent": 1, "Non_urgent": 2}
patients = sorted(patients, key=lambda x: urgency_order.get(x["urgency_level"], 3))

# Badge styling
def urgency_badge(level):
    colors = {"Emergency": "🔴", "Urgent": "🟡", "Non_urgent": "🟢"}
    return colors.get(level, "⚪")

st.title("🏥 Hospital Triage Agent")
st.caption("ML-powered risk classification + AI-generated triage decisions")

col1, col2 = st.columns([1, 2])

# Left panel: Patient queue
with col1:
    st.subheader("Patient Queue")
    selected_id = st.session_state.get("selected", patients[0]["patient_id"])

    for p in patients:
        badge = urgency_badge(p["urgency_level"])
        label = f"{badge} {p['patient_id']} — {p['predicted_class']} Risk ({p['risk_score']:.2f})"
        if st.button(label, key=p["patient_id"], use_container_width=True):
            st.session_state["selected"] = p["patient_id"]
            selected_id = p["patient_id"]

# Right panel: Triage decision card
with col2:
    patient = next(p for p in patients if p["patient_id"] == selected_id)

    st.subheader(f"Triage Decision — {patient['patient_id']}")

    badge = urgency_badge(patient["urgency_level"])
    st.markdown(f"### {badge} {patient['urgency_level'].replace('_', ' ').title()}")
    st.metric("ML Risk Score", f"{patient['risk_score']:.3f}", delta=patient["predicted_class"] + " Risk")

    st.markdown("**Triage Note**")
    st.info(patient["triage_note"])

    st.markdown("**Recommended Actions**")
    for action in patient["recommended_actions"]:
        st.checkbox(action, key=f"{patient['patient_id']}_{action}")

    st.markdown("**Patient Features**")
    st.json(patient["features"], expanded=False)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Approve Triage Plan", use_container_width=True, type="primary"):
            st.success(f"Triage plan approved for {patient['patient_id']}")
    with col_b:
        if st.button("🔺 Escalate to Senior", use_container_width=True):
            st.warning(f"{patient['patient_id']} escalated for senior review")
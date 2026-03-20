"""
main_agent.py
=============
Extended FastAPI server — drops into your existing 3 - DeploymentAPI folder
alongside the original main.py.

New endpoints added (all under /agent/):
  POST /agent/triage   — ML risk score + LLM triage decision
  POST /agent/claim    — ML claim outcome + LLM denial prevention
  POST /agent/visit    — Full admit-to-bill: triage + claim in one call

Original endpoints retained:
  GET  /health
  POST /predict/risk
  POST /predict/claim

Run from the 3 - DeploymentAPI folder:
  uvicorn main_agent:app --reload --port 8001

Environment variables:
  LLM_BACKEND     = mock | openai | anthropic   (default: mock)
  OPENAI_API_KEY  = sk-...
  ANTHROPIC_API_KEY = sk-ant-...
"""

import logging
import json
import joblib
import hashlib
import sys
import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# ── Allow importing agent modules from 5 - AI Agent folder ──
_AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "5 - AI Agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from triage_agent import TriageAgent
from claim_agent import ClaimAgent
from orchestrator import HospitalAgentOrchestrator, CombinedVisitReport

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("api_audit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Hospital Risk & Revenue Intelligence API — v2 (AI Agent)",
    description=(
        "Dual-model ML platform extended with LLM-powered agents for "
        "explainable triage decisions and denial-prevention claim reviews. "
        "v1 prediction endpoints retained for backward compatibility."
    ),
    version="2.0",
)

# ── Load models (v1 + v2 share the same pkl files) ───────────
try:
    _base = os.path.dirname(os.path.abspath(__file__))
    # If running from DeploymentAPI folder, look locally; else look in 3 - DeploymentAPI
    _api_dir = _base if os.path.exists(os.path.join(_base, "risk_model.pkl")) else \
        os.path.join(_base, "..", "3 - DeploymentAPI")

    risk_model  = joblib.load(os.path.join(_api_dir, "risk_model.pkl"))
    claim_model = joblib.load(os.path.join(_api_dir, "claim_model.pkl"))

    with open(os.path.join(_api_dir, "risk_feature_schema.json"))  as f: risk_schema  = json.load(f)
    with open(os.path.join(_api_dir, "claim_feature_schema.json")) as f: claim_schema = json.load(f)

    # Build Agent layer
    orchestrator = HospitalAgentOrchestrator(
        triage_agent=TriageAgent(
            ml_model=risk_model,
            feature_names=risk_schema["features"],
            target_mapping=risk_schema["target_mapping"],
        ),
        claim_agent=ClaimAgent(
            ml_model=claim_model,
            feature_names=claim_schema["features"],
            target_mapping=claim_schema["target_mapping"],
        ),
    )
    logger.info("Models, schemas, and AI agents loaded successfully.")
except Exception as e:
    logger.error(f"Startup Error: {e}")
    orchestrator = None


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

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


class FullVisitRequest(BaseModel):
    visit_id: Optional[str] = None
    patient: RiskRequest
    claim: ClaimRequest
    skip_claim_if_low_risk: bool = False


class PredictionResponse(BaseModel):
    prediction: str
    metadata: dict


class TriageResponse(BaseModel):
    visit_id: Optional[str]
    urgency_level: str
    risk_score: float
    risk_label: str
    top_factors: List[str]
    recommended_actions: List[str]
    triage_note: str
    metadata: dict


class ClaimAgentResponse(BaseModel):
    claim_id: Optional[str]
    outcome_label: str
    confidence: float
    denial_reasons: List[str]
    corrective_actions: List[str]
    rewritten_note: str
    compliance_flags: List[str]
    escalate_to_coder: bool
    metadata: dict


class FullVisitResponse(BaseModel):
    visit_id: Optional[str]
    priority_score: float
    triage: Optional[TriageResponse]
    claim: Optional[ClaimAgentResponse]
    metadata: dict


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def generate_hash(data: dict) -> str:
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()


def _meta(input_hash: str, model_version: str = "2.0") -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "model_version": model_version,
        "input_hash": input_hash,
        "llm_backend": os.getenv("LLM_BACKEND", "mock"),
    }


def _check_orchestrator():
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent not initialised. Check startup logs.")


# ─────────────────────────────────────────────
# v1 Endpoints (backward compatible)
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_ready": orchestrator is not None,
    }


@app.post("/predict/risk", response_model=PredictionResponse)
def predict_risk(request: RiskRequest):
    """v1 endpoint: returns raw ML prediction label only."""
    input_data = request.dict()
    input_hash = generate_hash(input_data)
    df = pd.DataFrame([input_data])[risk_schema["features"]]
    pred_idx = int(risk_model.predict(df)[0])
    pred_label = risk_schema["target_mapping"].get(str(pred_idx), "Unknown")
    logger.info(f"v1 | Model: Risk | Hash: {input_hash} | Pred: {pred_label}")
    return {"prediction": pred_label, "metadata": _meta(input_hash, "1.0")}


@app.post("/predict/claim", response_model=PredictionResponse)
def predict_claim(request: ClaimRequest):
    """v1 endpoint: returns raw ML prediction label only."""
    input_data = request.dict()
    input_hash = generate_hash(input_data)
    df = pd.DataFrame([input_data])[claim_schema["features"]]
    pred_idx = int(claim_model.predict(df)[0])
    pred_label = claim_schema["target_mapping"].get(str(pred_idx), "Unknown")
    logger.info(f"v1 | Model: Claim | Hash: {input_hash} | Pred: {pred_label}")
    return {"prediction": pred_label, "metadata": _meta(input_hash, "1.0")}


# ─────────────────────────────────────────────
# v2 Agent Endpoints
# ─────────────────────────────────────────────

@app.post("/agent/triage", response_model=TriageResponse)
def agent_triage(request: RiskRequest):
    """
    AI Triage Agent endpoint.
    Returns ML risk score + LLM-generated urgency classification,
    recommended clinical actions, and triage note.
    """
    _check_orchestrator()
    input_data = request.dict()
    input_hash = generate_hash(input_data)
    features = [input_data[f] for f in risk_schema["features"]]

    report = orchestrator.run_triage_only(features)
    d = report.triage

    logger.info(
        f"v2 | Agent: Triage | Hash: {input_hash} | "
        f"Urgency: {d.urgency_level} | Risk: {d.risk_label}"
    )

    return TriageResponse(
        visit_id=None,
        urgency_level=d.urgency_level,
        risk_score=d.risk_score,
        risk_label=d.risk_label,
        top_factors=d.top_factors,
        recommended_actions=d.recommended_actions,
        triage_note=d.triage_note,
        metadata=_meta(input_hash),
    )


@app.post("/agent/claim", response_model=ClaimAgentResponse)
def agent_claim(request: ClaimRequest):
    """
    AI Claim Agent endpoint.
    Returns ML outcome prediction + LLM denial-prevention analysis,
    corrective actions, compliance flags, and a rewritten claim narrative.
    """
    _check_orchestrator()
    input_data = request.dict()
    input_hash = generate_hash(input_data)
    features = [input_data[f] for f in claim_schema["features"]]

    report = orchestrator.run_claim_only(features)
    d = report.claim

    logger.info(
        f"v2 | Agent: Claim | Hash: {input_hash} | "
        f"Outcome: {d.outcome_label} | Escalate: {d.escalate_to_coder}"
    )

    return ClaimAgentResponse(
        claim_id=None,
        outcome_label=d.outcome_label,
        confidence=d.confidence,
        denial_reasons=d.denial_reasons,
        corrective_actions=d.corrective_actions,
        rewritten_note=d.rewritten_note,
        compliance_flags=d.compliance_flags,
        escalate_to_coder=d.escalate_to_coder,
        metadata=_meta(input_hash),
    )


@app.post("/agent/visit", response_model=FullVisitResponse)
def agent_full_visit(request: FullVisitRequest):
    """
    Full admit-to-bill pipeline.
    Runs Triage Agent + Claim Agent in sequence and returns a
    combined priority-scored visit report.
    """
    _check_orchestrator()
    visit_id = request.visit_id
    patient_data = request.patient.dict()
    claim_data   = request.claim.dict()
    input_hash   = generate_hash({"patient": patient_data, "claim": claim_data})

    patient_features = [patient_data[f] for f in risk_schema["features"]]
    claim_features   = [claim_data[f]   for f in claim_schema["features"]]

    report: CombinedVisitReport = orchestrator.run_full_visit(
        patient_features=patient_features,
        claim_features=claim_features,
        visit_id=visit_id,
        skip_claim_if_low_risk=request.skip_claim_if_low_risk,
    )

    logger.info(f"v2 | Agent: FullVisit | {report.summary()} | Hash: {input_hash}")

    t = report.triage
    c = report.claim

    triage_resp = TriageResponse(
        visit_id=visit_id,
        urgency_level=t.urgency_level,
        risk_score=t.risk_score,
        risk_label=t.risk_label,
        top_factors=t.top_factors,
        recommended_actions=t.recommended_actions,
        triage_note=t.triage_note,
        metadata=_meta(input_hash),
    ) if t else None

    claim_resp = ClaimAgentResponse(
        claim_id=visit_id,
        outcome_label=c.outcome_label,
        confidence=c.confidence,
        denial_reasons=c.denial_reasons,
        corrective_actions=c.corrective_actions,
        rewritten_note=c.rewritten_note,
        compliance_flags=c.compliance_flags,
        escalate_to_coder=c.escalate_to_coder,
        metadata=_meta(input_hash),
    ) if c else None

    return FullVisitResponse(
        visit_id=visit_id,
        priority_score=report.priority_score,
        triage=triage_resp,
        claim=claim_resp,
        metadata=_meta(input_hash),
    )

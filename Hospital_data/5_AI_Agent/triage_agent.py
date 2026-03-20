"""
triage_agent.py
================
AI Triage Agent — Layer 2 on top of the Risk Model (Model A).

Flow:
  patient_row  →  risk_model.predict_proba()  →  risk_score
                                                        ↓
  LLM(system_prompt + patient_dict + risk_score + SHAP importances)
                                                        ↓
  TriageDecision(urgency_level, risk_score, recommended_actions, triage_note)

Supports two LLM back-ends (set env var LLM_BACKEND):
  • "openai"    — requires OPENAI_API_KEY
  • "anthropic" — requires ANTHROPIC_API_KEY  (default)
  • "mock"      — returns deterministic stub; useful for offline testing
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class TriageDecision:
    """Output produced by the Triage Agent for a single patient row."""
    urgency_level: str                      # "non_urgent" | "urgent" | "emergent"
    risk_score: float                       # raw probability from ML model (0–1)
    risk_label: str                         # "High" | "Medium" | "Low"  (model mapping)
    top_factors: List[str]                  # top feature names driving the score
    recommended_actions: List[str]          # 2-5 clinical next steps
    triage_note: str                        # 1-3 sentence LLM rationale
    patient_id: Optional[str] = None       # optional passthrough identifier
    raw_llm_response: Optional[str] = None # for debugging / audit trail


# ──────────────────────────────────────────────
# LLM Client Wrappers
# ──────────────────────────────────────────────

class _MockLLMClient:
    """Deterministic stub — no API key required. Use LLM_BACKEND=mock."""
    def generate(self, prompt: str) -> str:
        # Parse risk score from prompt to produce semi-realistic mock output
        import re
        m = re.search(r"risk.*?:\s*([\d.]+)", prompt, re.IGNORECASE)
        score = float(m.group(1)) if m else 0.5
        if score >= 0.7:
            urgency = "emergent"
            actions = ["Immediate physician assessment", "ECG + ABG stat", "IV access and monitoring"]
        elif score >= 0.3:
            urgency = "urgent"
            actions = ["Nurse assessment within 30 min", "Basic labs ordered", "Vitals every 15 min"]
        else:
            urgency = "non_urgent"
            actions = ["Standard intake process", "Scheduled physician review"]
        return json.dumps({
            "urgency_level": urgency,
            "recommended_actions": actions,
            "triage_note": f"[MOCK] Risk score {score:.3f} → {urgency}. Clinical team should verify."
        })


class _OpenAIClient:
    """Thin wrapper around openai>=1.0."""
    def __init__(self):
        from openai import OpenAI  # type: ignore
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content


class _AnthropicClient:
    """Thin wrapper around anthropic>=0.25."""
    def __init__(self):
        import anthropic  # type: ignore
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


def _build_llm_client():
    backend = os.getenv("LLM_BACKEND", "mock").lower()
    if backend == "openai":
        return _OpenAIClient()
    elif backend == "anthropic":
        return _AnthropicClient()
    else:
        logger.warning("LLM_BACKEND not set or set to 'mock'. Using deterministic stub.")
        return _MockLLMClient()


# ──────────────────────────────────────────────
# Triage Agent
# ──────────────────────────────────────────────

class TriageAgent:
    """
    Wraps a trained risk classifier and an LLM to produce
    explainable triage decisions.

    Usage
    -----
    agent = TriageAgent.from_pkl("risk_model.pkl", "risk_feature_schema.json")
    decision = agent.triage(patient_series)       # pd.Series or 1-D array
    """

    # Maps the Random Forest's numeric output → human label (from schema)
    _TARGET_MAP = {"0": "High", "1": "Low", "2": "Medium"}

    # Maps risk_label → urgency bucket
    _RISK_TO_URGENCY = {
        "High":   "emergent",
        "Medium": "urgent",
        "Low":    "non_urgent",
    }

    def __init__(
        self,
        ml_model,
        feature_names: List[str],
        target_mapping: Optional[Dict[str, str]] = None,
        llm_client=None,
    ):
        self.ml_model = ml_model
        self.feature_names = feature_names
        self.target_mapping = target_mapping or self._TARGET_MAP
        self.llm_client = llm_client or _build_llm_client()

    # ── Factory ──────────────────────────────

    @classmethod
    def from_pkl(cls, model_path: str, schema_path: str, llm_client=None) -> "TriageAgent":
        """Load agent directly from .pkl and schema .json paths."""
        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)
        features = schema["features"]
        target_mapping = schema.get("target_mapping", cls._TARGET_MAP)
        return cls(model, features, target_mapping, llm_client)

    # ── Internal helpers ─────────────────────

    def _predict_risk(self, patient_features: np.ndarray) -> tuple[float, str]:
        """Returns (probability_of_high_risk_class, mapped_label)."""
        proba = self.ml_model.predict_proba([patient_features])[0]
        pred_idx = int(np.argmax(proba))
        risk_score = float(proba[pred_idx])
        risk_label = self.target_mapping.get(str(pred_idx), "Unknown")
        return risk_score, risk_label

    def _top_features(self, n: int = 4) -> List[str]:
        """Return top-n feature names by RF feature_importances_."""
        importances = getattr(self.ml_model, "feature_importances_", None)
        if importances is None:
            return self.feature_names[:n]
        ranked = sorted(
            zip(self.feature_names, importances), key=lambda x: x[1], reverse=True
        )
        return [name for name, _ in ranked[:n]]

    def _build_prompt(
        self,
        patient_dict: Dict[str, Any],
        risk_score: float,
        risk_label: str,
        top_factors: List[str],
    ) -> str:
        return f"""You are a clinical emergency department triage assistant.

PATIENT DATA:
{json.dumps(patient_dict, indent=2)}

ML MODEL OUTPUT:
- Risk score (confidence): {risk_score:.3f}
- Risk label: {risk_label}
- Top contributing features: {', '.join(top_factors)}

TASK:
Based on the patient data and model output, provide a structured triage assessment.

Return ONLY valid JSON with exactly these keys:
{{
  "urgency_level": "<one of: non_urgent | urgent | emergent>",
  "recommended_actions": ["<action 1>", "<action 2>", "..."],
  "triage_note": "<1-3 sentence clinical rationale>"
}}

Rules:
- urgency_level must match the severity implied by risk_label: High→emergent, Medium→urgent, Low→non_urgent (you may escalate if clinical context warrants it)
- recommended_actions: 2-5 specific, actionable steps (labs, monitoring, consults, bed assignment)
- triage_note: concise clinical justification citing the top contributing factors
- Return ONLY the JSON object, no markdown fencing
"""

    def _parse_llm_response(self, raw: str) -> Dict[str, Any]:
        """Safely parse JSON from LLM output, stripping markdown fences if present."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())

    # ── Public API ───────────────────────────

    def triage(
        self,
        patient_features,
        patient_id: Optional[str] = None,
    ) -> TriageDecision:
        """
        Run the full triage pipeline for one patient.

        Parameters
        ----------
        patient_features : array-like or pd.Series
            Must align with self.feature_names.
        patient_id : str, optional
            Passthrough identifier (visit_id, patient_id, etc.)

        Returns
        -------
        TriageDecision
        """
        # Normalise input
        if isinstance(patient_features, pd.Series):
            arr = patient_features[self.feature_names].values
        else:
            arr = np.asarray(patient_features, dtype=float)

        risk_score, risk_label = self._predict_risk(arr)
        top_factors = self._top_features(n=4)

        patient_dict = dict(zip(self.feature_names, arr.tolist()))
        prompt = self._build_prompt(patient_dict, risk_score, risk_label, top_factors)

        raw_response = self.llm_client.generate(prompt)
        llm_output = self._parse_llm_response(raw_response)

        return TriageDecision(
            urgency_level=llm_output["urgency_level"],
            risk_score=risk_score,
            risk_label=risk_label,
            top_factors=top_factors,
            recommended_actions=llm_output["recommended_actions"],
            triage_note=llm_output["triage_note"],
            patient_id=patient_id,
            raw_llm_response=raw_response,
        )

    def triage_batch(self, df: pd.DataFrame, id_col: Optional[str] = None) -> List[TriageDecision]:
        """Run triage for every row in a DataFrame. Returns list of TriageDecisions."""
        decisions = []
        for idx, row in df.iterrows():
            pid = str(row[id_col]) if id_col and id_col in row else str(idx)
            try:
                decisions.append(self.triage(row, patient_id=pid))
            except Exception as e:
                logger.error(f"Triage failed for patient {pid}: {e}")
        return decisions

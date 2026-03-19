"""
claim_agent.py
==============
AI Claim Agent — Layer 2 on top of the Claim Outcome Model (Model B).

Flow:
  claim_row  →  claim_model.predict_proba()  →  outcome_label
                                                       ↓
  LLM(system_prompt + claim_dict + outcome + RAG-style insurer rules)
                                                       ↓
  ClaimDecision(outcome, risk_score, denial_reasons, corrective_actions,
                rewritten_note, compliance_flags)

For "Rejected" or "Pending" claims the LLM acts as a specialist medical coder
and produces a fully rewritten, compliant claim submission.
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
class ClaimDecision:
    """Output produced by the Claim Agent for a single claim row."""
    outcome_label: str              # "Paid" | "Pending" | "Rejected"
    confidence: float               # model probability for the predicted class
    denial_reasons: List[str]       # key risk factors flagged by LLM
    corrective_actions: List[str]   # specific steps to fix/pre-empt rejection
    rewritten_note: str             # LLM-drafted compliant claim narrative
    compliance_flags: List[str]     # ICD/CPT/documentation gaps noted
    escalate_to_coder: bool         # True if human review recommended
    claim_id: Optional[str] = None
    raw_llm_response: Optional[str] = None


# ──────────────────────────────────────────────
# Insurer Policy Mini-RAG
# ──────────────────────────────────────────────

# In production this would be populated from a vector store of real insurer
# policy documents. This version hard-codes representative rules per provider
# and is keyed by insurance_provider_enc integer.

INSURER_POLICY_SNIPPETS: Dict[int, str] = {
    0: (
        "HealthFirst Policy: Claims >₹50,000 require pre-authorization. "
        "Missing ICD-10 code or diagnosis mismatch triggers automatic denial. "
        "Re-admission within 30 days classified as same episode — separate billing not allowed."
    ),
    1: (
        "SecureLife Policy: Highest rejection rate (15.69%). "
        "All surgical claims need operative notes attached. "
        "Chronic condition visits must reference most recent HbA1c/BP values."
    ),
    2: (
        "MedShield Policy: Requires itemised bill for claims >₹30,000. "
        "LOS > 7 days triggers mandatory utilisation review. "
        "Outpatient pharmacy billed separately; do not bundle."
    ),
    3: (
        "CareUnity Policy: Bundled payment model — avoid unbundling procedure codes. "
        "Emergency admissions require ER physician sign-off within 6 hours."
    ),
}

DEFAULT_POLICY = (
    "Standard insurer rules apply: accurate ICD-10/CPT coding, itemised billing "
    "for amounts over ₹25,000, and documented medical necessity are required."
)


def get_insurer_policy(insurance_provider_enc: int) -> str:
    return INSURER_POLICY_SNIPPETS.get(int(insurance_provider_enc), DEFAULT_POLICY)


# ──────────────────────────────────────────────
# LLM Client Wrappers (shared pattern with triage_agent)
# ──────────────────────────────────────────────

class _MockLLMClient:
    def generate(self, prompt: str) -> str:
        import re
        m = re.search(r"outcome.*?:\s*(\w+)", prompt, re.IGNORECASE)
        outcome = m.group(1).strip() if m else "Pending"
        escalate = outcome != "Paid"
        return json.dumps({
            "denial_reasons": ["[MOCK] High billed amount relative to provider norms"],
            "corrective_actions": ["Attach prior-auth documents", "Verify ICD-10 coding"],
            "rewritten_note": "[MOCK] Rewritten claim narrative. Patient presented with documented chronic condition. All procedures medically necessary per attached clinical notes.",
            "compliance_flags": ["Prior auth required for amounts >₹50k"],
            "escalate_to_coder": escalate,
        })


class _OpenAIClient:
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
    def __init__(self):
        import anthropic  # type: ignore
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            max_tokens=768,
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
# Claim Agent
# ──────────────────────────────────────────────

class ClaimAgent:
    """
    Wraps the trained claim outcome classifier and an LLM to produce
    denial-prevention intelligence and compliant rewritten claims.

    Usage
    -----
    agent = ClaimAgent.from_pkl("claim_model.pkl", "claim_feature_schema.json")
    decision = agent.review(claim_series)
    """

    _TARGET_MAP = {"0": "Paid", "1": "Pending", "2": "Rejected"}

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

    @classmethod
    def from_pkl(cls, model_path: str, schema_path: str, llm_client=None) -> "ClaimAgent":
        model = joblib.load(model_path)
        with open(schema_path) as f:
            schema = json.load(f)
        features = schema["features"]
        target_mapping = schema.get("target_mapping", cls._TARGET_MAP)
        return cls(model, features, target_mapping, llm_client)

    def _predict_outcome(self, claim_features: np.ndarray) -> tuple[float, str]:
        proba = self.ml_model.predict_proba([claim_features])[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        label = self.target_mapping.get(str(pred_idx), "Unknown")
        return confidence, label

    def _top_features(self, n: int = 4) -> List[str]:
        importances = getattr(self.ml_model, "feature_importances_", None)
        if importances is None:
            return self.feature_names[:n]
        ranked = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked[:n]]

    def _build_prompt(
        self,
        claim_dict: Dict[str, Any],
        outcome_label: str,
        confidence: float,
        top_factors: List[str],
        insurer_policy: str,
    ) -> str:
        needs_fix = outcome_label in ("Rejected", "Pending")
        fix_instruction = (
            "Since the claim is predicted to be REJECTED or PENDING, you must also:\n"
            "- Identify the specific denial reasons based on insurer policy\n"
            "- Provide corrective actions the billing coder should take BEFORE resubmission\n"
            "- Draft a rewritten_note: a compliant claim narrative citing medical necessity\n"
            "- Flag any compliance gaps (missing ICD-10 codes, missing prior auth, etc.)\n"
        ) if needs_fix else (
            "The claim is predicted to be PAID. Provide a brief confirmation note and any "
            "pre-emptive compliance flags to maintain this status.\n"
        )

        return f"""You are a specialist medical billing compliance officer and AI claim reviewer.

CLAIM DATA:
{json.dumps(claim_dict, indent=2)}

ML MODEL PREDICTION:
- Predicted outcome: {outcome_label}
- Model confidence: {confidence:.3f}
- Top risk features: {', '.join(top_factors)}

INSURER POLICY (retrieved):
{insurer_policy}

TASK:
{fix_instruction}
Return ONLY valid JSON with exactly these keys:
{{
  "denial_reasons": ["<reason 1>", ...],
  "corrective_actions": ["<action 1>", ...],
  "rewritten_note": "<compliant claim narrative>",
  "compliance_flags": ["<flag 1>", ...],
  "escalate_to_coder": true or false
}}

Rules:
- denial_reasons: 1-4 specific reasons based on claim data + insurer policy (empty list if Paid)
- corrective_actions: concrete steps (attach document X, update ICD code, add prior auth ref)
- rewritten_note: professional billing language, cite medical necessity, reference policy compliance
- compliance_flags: specific documentation gaps (e.g., "Missing ICD-10 E11.9 for Type 2 Diabetes")
- escalate_to_coder: true if the claim needs human expert review before resubmission
- Return ONLY the JSON object, no markdown fencing
"""

    def _parse_llm_response(self, raw: str) -> Dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())

    def review(
        self,
        claim_features,
        claim_id: Optional[str] = None,
    ) -> ClaimDecision:
        """
        Run the full claim review pipeline for one claim row.

        Parameters
        ----------
        claim_features : array-like or pd.Series
            Must align with self.feature_names.
        claim_id : str, optional
            Passthrough identifier.

        Returns
        -------
        ClaimDecision
        """
        if isinstance(claim_features, pd.Series):
            arr = claim_features[self.feature_names].values
        else:
            arr = np.asarray(claim_features, dtype=float)

        confidence, outcome_label = self._predict_outcome(arr)
        top_factors = self._top_features(n=4)

        claim_dict = dict(zip(self.feature_names, arr.tolist()))

        # Retrieve insurer policy (mini-RAG)
        enc_key = "insurance_provider_enc"
        insurer_enc = int(claim_dict.get(enc_key, -1))
        insurer_policy = get_insurer_policy(insurer_enc)

        prompt = self._build_prompt(claim_dict, outcome_label, confidence, top_factors, insurer_policy)
        raw_response = self.llm_client.generate(prompt)
        llm_output = self._parse_llm_response(raw_response)

        return ClaimDecision(
            outcome_label=outcome_label,
            confidence=confidence,
            denial_reasons=llm_output.get("denial_reasons", []),
            corrective_actions=llm_output.get("corrective_actions", []),
            rewritten_note=llm_output.get("rewritten_note", ""),
            compliance_flags=llm_output.get("compliance_flags", []),
            escalate_to_coder=bool(llm_output.get("escalate_to_coder", False)),
            claim_id=claim_id,
            raw_llm_response=raw_response,
        )

    def review_batch(self, df: pd.DataFrame, id_col: Optional[str] = None) -> List[ClaimDecision]:
        """Run review for every row in a DataFrame."""
        decisions = []
        for idx, row in df.iterrows():
            cid = str(row[id_col]) if id_col and id_col in row else str(idx)
            try:
                decisions.append(self.review(row, claim_id=cid))
            except Exception as e:
                logger.error(f"Claim review failed for {cid}: {e}")
        return decisions

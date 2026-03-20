"""
orchestrator.py
================
Hospital AI Agent Orchestrator.

Coordinates the Triage Agent and Claim Agent into a unified
"admit-to-bill" workflow for a single patient visit.

Architecture
------------
                    ┌─────────────────────┐
     patient_row ──▶│   TriageAgent        │──▶ TriageDecision
                    └─────────────────────┘
                              │
             (if urgency != non_urgent AND claim data present)
                              ▼
                    ┌─────────────────────┐
     claim_row   ──▶│   ClaimAgent         │──▶ ClaimDecision
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   CombinedVisitReport│
                    └─────────────────────┘

The orchestrator can also run each agent independently if only
one data type is provided.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import pandas as pd

from triage_agent import TriageAgent, TriageDecision
from claim_agent import ClaimAgent, ClaimDecision

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Combined Output
# ──────────────────────────────────────────────

@dataclass
class CombinedVisitReport:
    """
    Unified output for a patient visit that has both clinical
    and billing dimensions.
    """
    visit_id: Optional[str]
    triage: Optional[TriageDecision]
    claim: Optional[ClaimDecision]

    @property
    def has_triage(self) -> bool:
        return self.triage is not None

    @property
    def has_claim(self) -> bool:
        return self.claim is not None

    @property
    def priority_score(self) -> float:
        """
        Composite priority for queue sorting:
          - emergent = 1.0, urgent = 0.6, non_urgent = 0.2
          - Claim escalation adds 0.1 penalty (revenue risk)
        """
        urgency_weight = {"emergent": 1.0, "urgent": 0.6, "non_urgent": 0.2}
        base = urgency_weight.get(
            self.triage.urgency_level if self.triage else "non_urgent", 0.2
        )
        claim_penalty = 0.1 if (self.claim and self.claim.escalate_to_coder) else 0.0
        return round(base + claim_penalty, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visit_id": self.visit_id,
            "priority_score": self.priority_score,
            "triage": asdict(self.triage) if self.triage else None,
            "claim": asdict(self.claim) if self.claim else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Human-readable one-liner for logging / UI display."""
        parts = [f"Visit {self.visit_id or 'unknown'}"]
        if self.triage:
            parts.append(
                f"Urgency={self.triage.urgency_level.upper()} "
                f"(score={self.triage.risk_score:.2f})"
            )
        if self.claim:
            flag = " [ESCALATE]" if self.claim.escalate_to_coder else ""
            parts.append(f"Claim={self.claim.outcome_label}{flag}")
        parts.append(f"Priority={self.priority_score}")
        return " | ".join(parts)


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

class HospitalAgentOrchestrator:
    """
    Top-level controller that loads both agents from disk and
    coordinates them for single or batch processing.

    Usage (single patient)
    ----------------------
    orch = HospitalAgentOrchestrator.from_defaults()

    # Clinical-only triage
    report = orch.run_triage_only(patient_row, visit_id="V001")

    # Billing-only claim review
    report = orch.run_claim_only(claim_row, claim_id="C001")

    # Full admit-to-bill workflow
    report = orch.run_full_visit(patient_row, claim_row, visit_id="V001")
    """

    # Default paths relative to the 3 - DeploymentAPI folder
    DEFAULT_RISK_MODEL   = os.path.join("..", "3 - DeploymentAPI", "risk_model.pkl")
    DEFAULT_RISK_SCHEMA  = os.path.join("..", "3 - DeploymentAPI", "risk_feature_schema.json")
    DEFAULT_CLAIM_MODEL  = os.path.join("..", "3 - DeploymentAPI", "claim_model.pkl")
    DEFAULT_CLAIM_SCHEMA = os.path.join("..", "3 - DeploymentAPI", "claim_feature_schema.json")

    def __init__(self, triage_agent: TriageAgent, claim_agent: ClaimAgent):
        self.triage_agent = triage_agent
        self.claim_agent = claim_agent

    @classmethod
    def from_defaults(cls, llm_client=None) -> "HospitalAgentOrchestrator":
        """
        Load both agents from the default pkl/schema paths.
        Resolves paths relative to the location of this file.
        """
        base = os.path.dirname(os.path.abspath(__file__))
        api_dir = os.path.join(base, "..", "3 - DeploymentAPI")

        triage_agent = TriageAgent.from_pkl(
            model_path=os.path.join(api_dir, "risk_model.pkl"),
            schema_path=os.path.join(api_dir, "risk_feature_schema.json"),
            llm_client=llm_client,
        )
        claim_agent = ClaimAgent.from_pkl(
            model_path=os.path.join(api_dir, "claim_model.pkl"),
            schema_path=os.path.join(api_dir, "claim_feature_schema.json"),
            llm_client=llm_client,
        )
        return cls(triage_agent, claim_agent)

    # ── Single visit runners ─────────────────

    def run_triage_only(
        self,
        patient_features,
        visit_id: Optional[str] = None,
    ) -> CombinedVisitReport:
        triage_decision = self.triage_agent.triage(patient_features, patient_id=visit_id)
        return CombinedVisitReport(visit_id=visit_id, triage=triage_decision, claim=None)

    def run_claim_only(
        self,
        claim_features,
        claim_id: Optional[str] = None,
    ) -> CombinedVisitReport:
        claim_decision = self.claim_agent.review(claim_features, claim_id=claim_id)
        return CombinedVisitReport(visit_id=claim_id, triage=None, claim=claim_decision)

    def run_full_visit(
        self,
        patient_features,
        claim_features,
        visit_id: Optional[str] = None,
        skip_claim_if_low_risk: bool = False,
    ) -> CombinedVisitReport:
        """
        Full admit-to-bill pipeline.

        Parameters
        ----------
        patient_features : array-like or pd.Series
            Risk model features.
        claim_features : array-like or pd.Series
            Claim model features.
        visit_id : str, optional
        skip_claim_if_low_risk : bool
            If True and triage comes back non_urgent, skip claim review
            (useful for high-throughput ED settings to save LLM calls).
        """
        triage_decision = self.triage_agent.triage(patient_features, patient_id=visit_id)
        logger.info(f"[{visit_id}] Triage complete: {triage_decision.urgency_level}")

        if skip_claim_if_low_risk and triage_decision.urgency_level == "non_urgent":
            logger.info(f"[{visit_id}] Skipping claim review (low risk, skip flag set).")
            return CombinedVisitReport(visit_id=visit_id, triage=triage_decision, claim=None)

        claim_decision = self.claim_agent.review(claim_features, claim_id=visit_id)
        logger.info(
            f"[{visit_id}] Claim review complete: {claim_decision.outcome_label} "
            f"(escalate={claim_decision.escalate_to_coder})"
        )

        return CombinedVisitReport(
            visit_id=visit_id, triage=triage_decision, claim=claim_decision
        )

    # ── Batch runners ────────────────────────

    def run_batch_triage(
        self,
        df: pd.DataFrame,
        id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Process all rows in df through the Triage Agent.
        Returns a DataFrame with urgency_level, risk_score, risk_label,
        recommended_actions, triage_note columns appended.
        """
        decisions = self.triage_agent.triage_batch(df, id_col=id_col)
        results = []
        for d in decisions:
            results.append({
                "visit_id": d.patient_id,
                "urgency_level": d.urgency_level,
                "risk_score": d.risk_score,
                "risk_label": d.risk_label,
                "top_factors": ", ".join(d.top_factors),
                "recommended_actions": " | ".join(d.recommended_actions),
                "triage_note": d.triage_note,
            })
        return pd.DataFrame(results)

    def run_batch_claims(
        self,
        df: pd.DataFrame,
        id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Process all rows in df through the Claim Agent.
        Returns a DataFrame with claim review columns appended.
        """
        decisions = self.claim_agent.review_batch(df, id_col=id_col)
        results = []
        for d in decisions:
            results.append({
                "claim_id": d.claim_id,
                "outcome_label": d.outcome_label,
                "confidence": d.confidence,
                "denial_reasons": " | ".join(d.denial_reasons),
                "corrective_actions": " | ".join(d.corrective_actions),
                "compliance_flags": " | ".join(d.compliance_flags),
                "escalate_to_coder": d.escalate_to_coder,
                "rewritten_note": d.rewritten_note,
            })
        return pd.DataFrame(results)

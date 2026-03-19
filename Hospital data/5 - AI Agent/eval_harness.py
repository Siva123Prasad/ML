"""
eval_harness.py
===============
Evaluation framework for the Hospital AI Agent.

Three experiments:
  1. Urgency classification accuracy  — confusion matrix of agent urgency_level
     vs. ground-truth severity label derived from the dataset.
  2. Queue simulation                 — does sorting by agent priority get critical
     patients seen EARLIER than FIFO or raw risk-score ordering?
  3. Claim intervention ROI           — how many Rejected claims does the agent
     correctly flag, and what is the estimated revenue protected?

Usage
-----
# From the 5 - AI Agent folder:
python eval_harness.py --patients ../1\ -\ Analytics\ and\ EDA/visits.csv \
                       --billing  ../1\ -\ Analytics\ and\ EDA/billing.csv \
                       --n_rows 200

All results are saved to ./eval_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow imports from this folder
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from orchestrator import HospitalAgentOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(_HERE) / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Ground-truth urgency mapping
# ─────────────────────────────────────────────

def map_to_true_urgency(row: pd.Series) -> str:
    """
    Derive ground-truth urgency from available dataset columns.
    Adjust column names to match your actual schema if needed.
    """
    # Try common column names in priority order
    risk_col = None
    for c in ["risk_label", "risk_category", "visit_risk"]:
        if c in row.index:
            risk_col = c
            break

    if risk_col:
        val = str(row[risk_col]).strip().lower()
        if val in ("high", "emergent", "3", "critical"):
            return "emergent"
        elif val in ("medium", "moderate", "urgent", "2"):
            return "urgent"
        else:
            return "non_urgent"

    # Fallback: derive from LOS
    los_col = None
    for c in ["length_of_stay_hours", "los_hours", "length_of_stay"]:
        if c in row.index:
            los_col = c
            break

    chronic = row.get("chronic_flag", 0)
    los = float(row[los_col]) if los_col else 0

    if chronic == 1 and los > 72:
        return "emergent"
    elif chronic == 1 or los > 24:
        return "urgent"
    else:
        return "non_urgent"


# ─────────────────────────────────────────────
# Experiment 1: Urgency Classification
# ─────────────────────────────────────────────

def experiment_urgency_accuracy(
    orch: HospitalAgentOrchestrator,
    patient_df: pd.DataFrame,
    n_rows: int = 100,
) -> pd.DataFrame:
    """
    Compare agent urgency_level predictions vs. ground-truth urgency.
    Returns a confusion-matrix DataFrame and saves to eval_results/.
    """
    logger.info(f"Experiment 1: Urgency Accuracy on {n_rows} rows")
    sample = patient_df.head(n_rows).copy()

    # Build ground truth
    sample["true_urgency"] = sample.apply(map_to_true_urgency, axis=1)

    # Run batch triage
    batch_results = orch.run_batch_triage(sample)
    sample = sample.reset_index(drop=True)
    batch_results = batch_results.reset_index(drop=True)
    sample["pred_urgency"] = batch_results["urgency_level"]

    # Confusion matrix
    labels = ["non_urgent", "urgent", "emergent"]
    conf = pd.crosstab(
        sample["true_urgency"],
        sample["pred_urgency"],
        rownames=["True"],
        colnames=["Predicted"],
    ).reindex(index=labels, columns=labels, fill_value=0)

    correct = (sample["true_urgency"] == sample["pred_urgency"]).sum()
    accuracy = correct / len(sample)
    logger.info(f"Urgency Accuracy: {accuracy:.1%} ({correct}/{len(sample)})")

    # Save
    conf.to_csv(RESULTS_DIR / "urgency_confusion_matrix.csv")
    sample[["true_urgency", "pred_urgency", "risk_score", "triage_note"]].to_csv(
        RESULTS_DIR / "urgency_predictions.csv", index=False
    )

    print("\n── Experiment 1: Urgency Confusion Matrix ──")
    print(conf.to_string())
    print(f"Overall Accuracy: {accuracy:.1%}\n")
    return conf


# ─────────────────────────────────────────────
# Experiment 2: Queue Simulation
# ─────────────────────────────────────────────

def _top_k_recall(sorted_df: pd.DataFrame, k: int, critical_label: str = "emergent") -> float:
    """Fraction of critical patients found in the top-k ranked positions."""
    top_k = sorted_df.head(k)
    critical_in_top = (top_k["true_urgency"] == critical_label).sum()
    total_critical = (sorted_df["true_urgency"] == critical_label).sum()
    if total_critical == 0:
        return 0.0
    return critical_in_top / total_critical


def experiment_queue_simulation(
    orch: HospitalAgentOrchestrator,
    patient_df: pd.DataFrame,
    n_rows: int = 200,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Simulate three queue orderings and compare how many critical patients
    are seen in the top-k positions:
      A) FIFO (arrival order)
      B) ML risk score only (no LLM)
      C) Agent urgency level (LLM-assisted)
    """
    logger.info(f"Experiment 2: Queue Simulation on {n_rows} rows, top-k={top_k}")
    sample = patient_df.head(n_rows).copy().reset_index(drop=True)
    sample["arrival_order"] = range(len(sample))
    sample["true_urgency"] = sample.apply(map_to_true_urgency, axis=1)

    # Run triage
    batch_results = orch.run_batch_triage(sample).reset_index(drop=True)
    sample["agent_urgency"] = batch_results["urgency_level"]
    sample["agent_risk_score"] = batch_results["risk_score"]

    # Map urgency to numeric priority
    urgency_rank = {"emergent": 3, "urgent": 2, "non_urgent": 1}
    sample["urgency_priority"] = sample["agent_urgency"].map(urgency_rank)

    # Queue A: FIFO
    fifo_df = sample.sort_values("arrival_order")
    recall_fifo = _top_k_recall(fifo_df, k=top_k)

    # Queue B: ML risk score (descending)
    ml_df = sample.sort_values("agent_risk_score", ascending=False)
    recall_ml = _top_k_recall(ml_df, k=top_k)

    # Queue C: Agent urgency priority (descending), then risk score as tiebreaker
    agent_df = sample.sort_values(
        ["urgency_priority", "agent_risk_score"], ascending=[False, False]
    )
    recall_agent = _top_k_recall(agent_df, k=top_k)

    results = pd.DataFrame([
        {"ordering": "FIFO",          "recall_critical_top_k": recall_fifo},
        {"ordering": "ML Score Only", "recall_critical_top_k": recall_ml},
        {"ordering": "AI Agent",      "recall_critical_top_k": recall_agent},
    ])

    results["top_k"] = top_k
    results["n_patients"] = n_rows
    results.to_csv(RESULTS_DIR / "queue_simulation_results.csv", index=False)

    print("\n── Experiment 2: Queue Simulation ──")
    print(results.to_string(index=False))
    print(
        f"\nAI Agent lifts critical-patient recall in top-{top_k} by "
        f"{recall_agent - recall_fifo:+.1%} vs FIFO, "
        f"{recall_agent - recall_ml:+.1%} vs ML-only.\n"
    )
    return results


# ─────────────────────────────────────────────
# Experiment 3: Claim Intervention ROI
# ─────────────────────────────────────────────

def experiment_claim_roi(
    orch: HospitalAgentOrchestrator,
    billing_df: pd.DataFrame,
    n_rows: int = 200,
    avg_recovery_rate: float = 0.65,
) -> pd.DataFrame:
    """
    Estimate revenue protected by routing agent-flagged claims to coders
    before submission.

    avg_recovery_rate: fraction of flagged Rejected claims successfully
    corrected and paid after coder review (literature: 60-70%).
    """
    logger.info(f"Experiment 3: Claim ROI on {n_rows} rows")
    sample = billing_df.head(n_rows).copy().reset_index(drop=True)

    batch_results = orch.run_batch_claims(sample).reset_index(drop=True)
    sample["pred_outcome"]      = batch_results["outcome_label"]
    sample["escalate_to_coder"] = batch_results["escalate_to_coder"]

    # Identify agent-flagged rejections
    flagged_rejected = sample[
        (sample["pred_outcome"] == "Rejected") & (sample["escalate_to_coder"])
    ]

    # Revenue at risk
    billed_col = next((c for c in ["billed_amount", "billed", "amount"] if c in sample.columns), None)
    if billed_col:
        revenue_at_risk = flagged_rejected[billed_col].sum()
        recoverable     = revenue_at_risk * avg_recovery_rate
    else:
        revenue_at_risk = None
        recoverable     = None

    n_total    = len(sample)
    n_flagged  = len(flagged_rejected)
    flag_rate  = n_flagged / n_total if n_total > 0 else 0

    summary = {
        "total_claims":       n_total,
        "flagged_for_review": n_flagged,
        "flag_rate":          round(flag_rate, 4),
        "revenue_at_risk_inr": revenue_at_risk,
        "recoverable_inr":    recoverable,
        "assumed_recovery_rate": avg_recovery_rate,
    }

    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "claim_roi_summary.csv", index=False)
    sample[["pred_outcome", "escalate_to_coder"]].to_csv(
        RESULTS_DIR / "claim_predictions.csv", index=False
    )

    print("\n── Experiment 3: Claim ROI ──")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    return pd.DataFrame([summary])


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hospital AI Agent Evaluation Harness")
    parser.add_argument("--patients", type=str, default=None, help="Path to patient/visits CSV")
    parser.add_argument("--billing",  type=str, default=None, help="Path to billing CSV")
    parser.add_argument("--n_rows",   type=int, default=100,  help="Number of rows to evaluate")
    parser.add_argument("--top_k",    type=int, default=20,   help="k for queue recall@k")
    args = parser.parse_args()

    # Resolve data paths
    data_dir = Path(_HERE) / ".." / "1 - Analytics and EDA"
    patient_path = args.patients or (data_dir / "visits.csv")
    billing_path = args.billing  or (data_dir / "billing.csv")

    # Load data
    try:
        patient_df = pd.read_csv(patient_path)
        logger.info(f"Loaded patients: {len(patient_df)} rows from {patient_path}")
    except FileNotFoundError:
        logger.error(f"Patients file not found: {patient_path}")
        patient_df = None

    try:
        billing_df = pd.read_csv(billing_path)
        logger.info(f"Loaded billing: {len(billing_df)} rows from {billing_path}")
    except FileNotFoundError:
        logger.error(f"Billing file not found: {billing_path}")
        billing_df = None

    # Load orchestrator
    orch = HospitalAgentOrchestrator.from_defaults()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments
    if patient_df is not None:
        # Fill only the feature columns that the risk model expects
        risk_features = orch.triage_agent.feature_names
        available = [c for c in risk_features if c in patient_df.columns]
        missing   = [c for c in risk_features if c not in patient_df.columns]
        if missing:
            logger.warning(f"Adding zero-fill for missing columns: {missing}")
            for c in missing:
                patient_df[c] = 0

        experiment_urgency_accuracy(orch, patient_df, n_rows=args.n_rows)
        experiment_queue_simulation(orch, patient_df, n_rows=args.n_rows * 2, top_k=args.top_k)

    if billing_df is not None:
        claim_features = orch.claim_agent.feature_names
        for c in claim_features:
            if c not in billing_df.columns:
                billing_df[c] = 0
        experiment_claim_roi(orch, billing_df, n_rows=args.n_rows)

    print(f"\nAll results saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()

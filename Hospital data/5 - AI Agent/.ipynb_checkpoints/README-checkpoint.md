# Phase 5 — AI Agent Layer

Upgrades the pure ML pipeline (Phases 1–4) from **prediction → action** by
placing an LLM reasoning layer on top of both trained Random Forest models.

---

## Architecture

```
                 ┌─────────────────────────────────────────┐
Patient row ───▶ │  TriageAgent                            │
                 │  risk_model.pkl  →  risk_score           │
                 │  LLM prompt      →  urgency_level        │
                 │                     recommended_actions  │
                 │                     triage_note          │
                 └─────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────────┐
Claim row ─────▶ │  ClaimAgent                             │
                 │  claim_model.pkl →  outcome_label        │
                 │  LLM + RAG rules →  denial_reasons       │
                 │                     corrective_actions   │
                 │                     rewritten_note       │
                 │                     compliance_flags     │
                 └─────────────────────────────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────────┐
                 │  HospitalAgentOrchestrator               │
                 │  → CombinedVisitReport                   │
                 │  → priority_score (queue ordering)       │
                 └─────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `triage_agent.py` | Wraps risk model + LLM → explainable triage decisions |
| `claim_agent.py` | Wraps claim model + LLM + insurer policy RAG → denial prevention |
| `orchestrator.py` | Coordinates both agents; single and batch processing |
| `main_agent.py` | Extended FastAPI server (v2 endpoints + v1 backward-compat) |
| `eval_harness.py` | 3 evaluation experiments: accuracy, queue simulation, claim ROI |

## New API Endpoints (v2)

| Endpoint | Description |
|----------|-------------|
| `POST /agent/triage` | ML risk + LLM triage decision |
| `POST /agent/claim` | ML claim outcome + LLM denial prevention + rewritten note |
| `POST /agent/visit` | Full admit-to-bill pipeline (triage + claim combined) |
| `GET  /health` | Health check (now includes `agent_ready` flag) |
| `POST /predict/risk` | v1 backward-compatible ML-only risk prediction |
| `POST /predict/claim` | v1 backward-compatible ML-only claim prediction |

## Setup

```bash
cd "Hospital data/5 - AI Agent"
pip install -r requirements.txt
```

Set LLM backend (default is `mock` — no API key needed):

```bash
# OpenAI
export LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...

# Anthropic
export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Offline testing (deterministic stub)
export LLM_BACKEND=mock
```

Run the server (from the `3 - DeploymentAPI` folder or adjust paths):

```bash
# From Hospital data root
cd "3 - DeploymentAPI"
uvicorn "$(pwd)/../5 - AI Agent/main_agent":app --reload --port 8001
```

Or the simpler approach — copy `main_agent.py` into `3 - DeploymentAPI/`:

```bash
cp "5 - AI Agent/main_agent.py" "3 - DeploymentAPI/"
cd "3 - DeploymentAPI"
uvicorn main_agent:app --reload --port 8001
```

Open http://localhost:8001/docs for the Swagger UI.

## Evaluation

```bash
cd "5 - AI Agent"
python eval_harness.py --patients "../1 - Analytics and EDA/visits.csv" \
                       --billing  "../1 - Analytics and EDA/billing.csv" \
                       --n_rows 200 --top_k 20
```

Results saved to `5 - AI Agent/eval_results/`:
- `urgency_confusion_matrix.csv`
- `urgency_predictions.csv`
- `queue_simulation_results.csv`
- `claim_roi_summary.csv`
- `claim_predictions.csv`

## Example Usage (Python)

```python
from orchestrator import HospitalAgentOrchestrator

orch = HospitalAgentOrchestrator.from_defaults()

# Single visit — full pipeline
report = orch.run_full_visit(patient_row, claim_row, visit_id="V001")
print(report.summary())
# Visit V001 | Urgency=EMERGENT (score=0.81) | Claim=Rejected [ESCALATE] | Priority=1.1

print(report.triage.triage_note)
print(report.claim.rewritten_note)
```

## LLM Backend Notes

- **`mock`** (default): Zero-dependency, deterministic stub. Use for development
  and offline testing. Output is labelled `[MOCK]`.
- **`openai`**: Uses `gpt-4o-mini` by default. Override with `OPENAI_MODEL` env var.
- **`anthropic`**: Uses `claude-3-5-haiku-20241022` by default. Override with
  `ANTHROPIC_MODEL` env var. Cheapest production option for high-throughput ED use.

## Insurer Policy RAG

`claim_agent.py` ships with hard-coded policy snippets for the 4 insurers in your
dataset (HealthFirst, SecureLife, MedShield, CareUnity). In production, replace the
`INSURER_POLICY_SNIPPETS` dict with a proper vector store (e.g., ChromaDB, Pinecone)
populated from the real insurer PDF policy documents.

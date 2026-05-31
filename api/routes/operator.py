"""
Operator Dashboard API — Human-in-the-Loop management.
Stores pending review cases and allows operators to approve/reject/override.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/operator", tags=["Operator Dashboard"])

PENDING_PATH = Path("memory/data/pending_reviews.json")


# ── Storage helpers ──────────────────────────────────────

def _load() -> list:
    if PENDING_PATH.exists():
        with open(PENDING_PATH) as f:
            return json.load(f)
    return []


def _save(cases: list):
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PENDING_PATH, "w") as f:
        json.dump(cases, f, indent=2, default=str)


def add_pending_case(
    session_id: str,
    user_id: str,
    user_message: str,
    agent_response: str,
    intent: str,
    confidence: float,
    trace_id: str,
) -> dict:
    """Called by orchestrator when confidence < threshold."""
    case = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
        "user_message": user_message,
        "agent_response": agent_response,
        "intent": intent,
        "confidence": confidence,
        "trace_id": trace_id,
        "status": "pending",
        "operator_response": None,
        "decision": None,
        "created_at": datetime.now().isoformat(),
        "resolved_at": None,
    }
    cases = _load()
    cases.append(case)
    _save(cases)
    return case


# ── Schemas ──────────────────────────────────────────────

class ReviewDecision(BaseModel):
    decision: str  # "approve" | "reject" | "override"
    operator_response: Optional[str] = None


# ── Routes ───────────────────────────────────────────────

@router.get("/")
async def operator_dashboard():
    """Serve the operator dashboard HTML."""
    return FileResponse("web/operator.html")


@router.get("/cases")
async def list_cases(status: Optional[str] = None):
    """List all review cases, optionally filtered by status."""
    cases = _load()
    if status:
        cases = [c for c in cases if c["status"] == status]
    return {
        "cases": sorted(cases, key=lambda c: c["created_at"], reverse=True),
        "total": len(cases),
        "pending": sum(1 for c in cases if c["status"] == "pending"),
    }


@router.post("/cases/{case_id}/review")
async def review_case(case_id: str, body: ReviewDecision):
    """Approve, reject, or override an agent response."""
    cases = _load()
    for case in cases:
        if case["id"] == case_id:
            if case["status"] != "pending":
                raise HTTPException(400, "Case already reviewed")

            case["decision"] = body.decision
            case["operator_response"] = body.operator_response
            case["status"] = "resolved"
            case["resolved_at"] = datetime.now().isoformat()
            _save(cases)
            return {"status": "resolved", "case": case}

    raise HTTPException(404, f"Case {case_id} not found")


@router.get("/stats")
async def operator_stats():
    """Operator dashboard statistics."""
    cases = _load()
    total = len(cases)
    pending = sum(1 for c in cases if c["status"] == "pending")
    resolved = sum(1 for c in cases if c["status"] == "resolved")

    decisions: dict = {}
    for c in cases:
        d = c.get("decision")
        if d:
            decisions[d] = decisions.get(d, 0) + 1

    avg_confidence = (
        sum(c.get("confidence", 0) for c in cases) / total if total else 0
    )

    return {
        "total_cases": total,
        "pending": pending,
        "resolved": resolved,
        "decisions": decisions,
        "avg_confidence_at_escalation": round(avg_confidence, 3),
    }

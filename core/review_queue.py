"""
Human-in-the-loop review queue.

The orchestrator previously reached into `api.routes.operator` to enqueue a
low-confidence case, which made the agent layer depend on the web layer. The
storage lives here instead; the operator route is a thin HTTP surface over it.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core import config
from core.storage import lock_for, read_json, write_json

PENDING_PATH = config.data_path("pending_reviews.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cases() -> List[dict]:
    cases = read_json(PENDING_PATH, default=[])
    return cases if isinstance(cases, list) else []


def save_cases(cases: List[dict]) -> None:
    write_json(PENDING_PATH, cases)


def add_pending_case(
    session_id: str,
    user_id: str,
    user_message: str,
    agent_response: str,
    intent: str,
    confidence: float,
    trace_id: str,
) -> dict:
    """Enqueue a case for operator review. Called when confidence is too low."""
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
        "created_at": _now(),
        "resolved_at": None,
    }

    # Read-modify-write must be atomic against a concurrent escalation, or one
    # of the two cases is lost when the second write overwrites the document.
    with lock_for(PENDING_PATH):
        cases = load_cases()
        cases.append(case)
        save_cases(cases)

    return case


def resolve_case(
    case_id: str,
    decision: str,
    operator_response: Optional[str] = None,
) -> Optional[dict]:
    """
    Resolve a pending case.

    Returns the updated case, or None when the ID is unknown. Raises
    ValueError when the case has already been reviewed.
    """
    with lock_for(PENDING_PATH):
        cases = load_cases()
        for case in cases:
            if case["id"] != case_id:
                continue
            if case["status"] != "pending":
                raise ValueError("Case already reviewed")

            case["decision"] = decision
            case["operator_response"] = operator_response
            case["status"] = "resolved"
            case["resolved_at"] = _now()
            save_cases(cases)
            return case

    return None


def get_stats() -> Dict:
    cases = load_cases()
    total = len(cases)
    decisions: Dict[str, int] = {}
    for case in cases:
        decision = case.get("decision")
        if decision:
            decisions[decision] = decisions.get(decision, 0) + 1

    avg_confidence = (
        sum(c.get("confidence", 0) for c in cases) / total if total else 0
    )

    return {
        "total_cases": total,
        "pending": sum(1 for c in cases if c.get("status") == "pending"),
        "resolved": sum(1 for c in cases if c.get("status") == "resolved"),
        "decisions": decisions,
        "avg_confidence_at_escalation": round(avg_confidence, 3),
    }

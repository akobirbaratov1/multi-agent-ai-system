"""
Human Feedback Collection — Collects thumbs up/down and comments on agent responses.
Used for continuous improvement and fine-tuning signal.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core import config
from core.storage import append_jsonl, read_jsonl, tail_jsonl

FEEDBACK_PATH = config.data_path("feedback.jsonl")


def _append(record: dict):
    append_jsonl(FEEDBACK_PATH, record)


def _load_all() -> List[dict]:
    return read_jsonl(FEEDBACK_PATH)


def submit_feedback(
    trace_id: str,
    session_id: str,
    rating: int,
    comment: Optional[str] = None,
    agent: Optional[str] = None,
) -> dict:
    """
    Submit human feedback for a response.

    Args:
        trace_id: The trace ID from the agent response
        session_id: The session this feedback belongs to
        rating: 1 (thumbs up) or -1 (thumbs down)
        comment: Optional free-text comment
        agent: The agent that produced the response

    Returns:
        Feedback record
    """
    if rating not in (1, -1):
        raise ValueError("rating must be 1 (positive) or -1 (negative)")

    record = {
        "id": str(uuid.uuid4()),
        "trace_id": trace_id,
        "session_id": session_id,
        "rating": rating,
        "sentiment": "positive" if rating == 1 else "negative",
        "comment": comment,
        "agent": agent,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    _append(record)
    return record


def get_feedback_stats() -> Dict:
    """Aggregate feedback statistics."""
    records = _load_all()

    if not records:
        return {"total_feedback": 0}

    positive = sum(1 for r in records if r.get("rating") == 1)
    negative = sum(1 for r in records if r.get("rating") == -1)
    total = len(records)

    by_agent: Dict[str, Dict] = {}
    for r in records:
        agent = r.get("agent", "unknown")
        if agent not in by_agent:
            by_agent[agent] = {"positive": 0, "negative": 0}
        if r.get("rating") == 1:
            by_agent[agent]["positive"] += 1
        else:
            by_agent[agent]["negative"] += 1

    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": round(positive / total, 3) if total else 0,
        "by_agent": by_agent,
    }


def get_recent_feedback(n: int = 10) -> List[dict]:
    """Return n most recent feedback entries."""
    return tail_jsonl(FEEDBACK_PATH, n)

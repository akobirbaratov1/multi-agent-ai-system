"""
Follow-up System — Scheduled follow-ups for CRM leads.
Tracks follow-up queue and dispatches emails at the right time.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

FOLLOWUP_PATH = Path("memory/data/followups.json")


def _load() -> List[dict]:
    if FOLLOWUP_PATH.exists():
        with open(FOLLOWUP_PATH) as f:
            return json.load(f)
    return []


def _save(items: List[dict]):
    FOLLOWUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FOLLOWUP_PATH, "w") as f:
        json.dump(items, f, indent=2, default=str)


def schedule_followup(
    lead_id: str,
    user_id: str,
    email: Optional[str],
    reason: str,
    delay_hours: int = 24,
    template: str = "follow_up",
) -> dict:
    """
    Schedule a follow-up for a lead.

    Args:
        lead_id: CRM lead ID
        user_id: User identifier
        email: Recipient email address
        reason: Why this follow-up was scheduled
        delay_hours: Hours until follow-up should be sent
        template: Email template to use

    Returns:
        Follow-up record
    """
    send_at = datetime.now() + timedelta(hours=delay_hours)

    record = {
        "id": str(uuid.uuid4()),
        "lead_id": lead_id,
        "user_id": user_id,
        "email": email,
        "reason": reason,
        "template": template,
        "delay_hours": delay_hours,
        "scheduled_at": datetime.now().isoformat(),
        "send_at": send_at.isoformat(),
        "status": "scheduled",
        "sent_at": None,
    }

    items = _load()
    items.append(record)
    _save(items)
    return record


def get_due_followups() -> List[dict]:
    """Return all follow-ups that are due to be sent now."""
    now = datetime.now()
    items = _load()
    return [
        f for f in items
        if f["status"] == "scheduled"
        and datetime.fromisoformat(f["send_at"]) <= now
    ]


def mark_sent(followup_id: str) -> Optional[dict]:
    """Mark a follow-up as sent."""
    items = _load()
    for item in items:
        if item["id"] == followup_id:
            item["status"] = "sent"
            item["sent_at"] = datetime.now().isoformat()
            _save(items)
            return item
    return None


def process_due_followups() -> List[dict]:
    """
    Process all due follow-ups — send emails and mark as sent.
    Call this on a schedule (e.g. every hour via cron or background task).
    """
    from tools.email import send_email

    due = get_due_followups()
    processed = []

    for followup in due:
        if followup.get("email"):
            send_email(
                to=followup["email"],
                template=followup.get("template", "follow_up"),
                variables={"name": followup.get("user_id", "there")},
                lead_id=followup.get("lead_id"),
            )
        mark_sent(followup["id"])
        processed.append(followup)

    return processed


def get_stats() -> Dict:
    items = _load()
    return {
        "total": len(items),
        "scheduled": sum(1 for f in items if f["status"] == "scheduled"),
        "sent": sum(1 for f in items if f["status"] == "sent"),
        "overdue": len([
            f for f in items
            if f["status"] == "scheduled"
            and datetime.fromisoformat(f["send_at"]) < datetime.now()
        ]),
    }

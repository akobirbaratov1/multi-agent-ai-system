"""
Follow-up System — Scheduled follow-ups for CRM leads.
Tracks follow-up queue and dispatches emails at the right time.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from core import config
from core.logging_config import get_logger
from core.storage import lock_for, read_json, write_json

logger = get_logger(__name__)

FOLLOWUP_PATH = config.data_path("followups.json")

MAX_ATTEMPTS = 3


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str) -> Optional[datetime]:
    """
    Parse a stored timestamp, tolerating both naive (pre-1.1) and aware values.

    Records written by earlier builds carry naive local timestamps; comparing
    those against an aware `now` raises TypeError and would break the whole
    queue sweep on upgrade.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _load() -> List[dict]:
    items = read_json(FOLLOWUP_PATH, default=[])
    return items if isinstance(items, list) else []


def _save(items: List[dict]) -> None:
    write_json(FOLLOWUP_PATH, items)


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
        email: Recipient email address (None schedules a queue entry an
            operator can complete later)
        reason: Why this follow-up was scheduled
        delay_hours: Hours until follow-up should be sent
        template: Email template to use

    Returns:
        Follow-up record
    """
    now = _now()
    record = {
        "id": str(uuid.uuid4()),
        "lead_id": lead_id,
        "user_id": user_id,
        "email": email,
        "reason": reason,
        "template": template,
        "delay_hours": delay_hours,
        "scheduled_at": now.isoformat(),
        "send_at": (now + timedelta(hours=delay_hours)).isoformat(),
        "status": "scheduled",
        "sent_at": None,
        "attempts": 0,
        "last_error": None,
    }

    with lock_for(FOLLOWUP_PATH):
        items = _load()
        items.append(record)
        _save(items)

    return record


def get_due_followups() -> List[dict]:
    """Return all follow-ups that are due to be sent now."""
    now = _now()
    return [
        f for f in _load()
        if f.get("status") == "scheduled"
        and (_parse_ts(f.get("send_at", "")) or now) <= now
    ]


def _update(followup_id: str, **fields) -> Optional[dict]:
    with lock_for(FOLLOWUP_PATH):
        items = _load()
        for item in items:
            if item.get("id") == followup_id:
                item.update(fields)
                _save(items)
                return item
    return None


def mark_sent(followup_id: str) -> Optional[dict]:
    """Mark a follow-up as sent."""
    return _update(followup_id, status="sent", sent_at=_now().isoformat())


def process_due_followups() -> Dict:
    """
    Process all due follow-ups — send emails and mark as sent.

    Call this on a schedule; the API runs it from a background task when
    FOLLOWUP_INTERVAL_SECONDS is configured.

    Returns a summary rather than the raw list so callers can log outcomes.
    """
    from tools.email import EmailError, is_valid_email, send_email

    sent, skipped, failed = 0, 0, 0

    for followup in get_due_followups():
        followup_id = followup["id"]
        email = followup.get("email")

        # No address on file — the lead came in through a channel that never
        # collected one. Park it for an operator instead of retrying forever.
        if not is_valid_email(email or ""):
            _update(
                followup_id,
                status="needs_contact_info",
                last_error="No valid email address on file",
            )
            skipped += 1
            continue

        try:
            send_email(
                to=email,
                template=followup.get("template", "follow_up"),
                variables={"name": followup.get("user_id", "there")},
                lead_id=followup.get("lead_id"),
            )
        except (EmailError, OSError) as exc:
            attempts = followup.get("attempts", 0) + 1
            _update(
                followup_id,
                attempts=attempts,
                last_error=str(exc),
                status="failed" if attempts >= MAX_ATTEMPTS else "scheduled",
            )
            logger.warning(
                "Follow-up dispatch failed",
                extra={"followup_id": followup_id, "attempts": attempts},
            )
            failed += 1
            continue

        mark_sent(followup_id)
        sent += 1

    return {"sent": sent, "skipped": skipped, "failed": failed}


def get_stats() -> Dict:
    items = _load()
    now = _now()
    return {
        "total": len(items),
        "scheduled": sum(1 for f in items if f.get("status") == "scheduled"),
        "sent": sum(1 for f in items if f.get("status") == "sent"),
        "failed": sum(1 for f in items if f.get("status") == "failed"),
        "needs_contact_info": sum(
            1 for f in items if f.get("status") == "needs_contact_info"
        ),
        "overdue": sum(
            1 for f in items
            if f.get("status") == "scheduled"
            and (_parse_ts(f.get("send_at", "")) or now) < now
        ),
    }

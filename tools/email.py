"""
Email Dispatch — Mock SMTP integration
Simulates real email sending (SendGrid/SES compatible structure).
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from core import config
from core.storage import lock_for, read_json, write_json

EMAIL_LOG_PATH = config.data_path("email_log.json")

TEMPLATES = {
    "welcome": {
        "subject": "Welcome to Our Platform!",
        "body": "Hi {name},\n\nThank you for your interest. Our team will reach out within 24 hours.\n\nBest regards,\nThe Team",
    },
    "demo_confirmation": {
        "subject": "Your Demo is Confirmed",
        "body": "Hi {name},\n\nYour demo has been scheduled. A calendar invite will follow shortly.\n\nBest regards,\nThe Team",
    },
    "follow_up": {
        "subject": "Following up on your inquiry",
        "body": "Hi {name},\n\nJust following up on your recent conversation with our AI assistant.\n\nWould you like to schedule a call?\n\nBest regards,\nThe Team",
    },
    "support_resolution": {
        "subject": "Your Support Ticket has been Resolved",
        "body": "Hi {name},\n\nWe're glad we could help resolve your issue. Please don't hesitate to reach out if you need further assistance.\n\nBest regards,\nSupport Team",
    },
    "default": {
        "subject": "{subject}",
        "body": "{body}",
    },
}

# Sensible stand-ins so a template placeholder the caller didn't supply renders
# as neutral text instead of raising. `"{name}".format()` with no `name` raised
# KeyError and took the whole follow-up run down.
_PLACEHOLDER_DEFAULTS = {
    "name": "there",
    "subject": "A message from our team",
    "body": "",
}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class EmailError(ValueError):
    """Raised when an email cannot be constructed or addressed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_log() -> list:
    log = read_json(EMAIL_LOG_PATH, default=[])
    return log if isinstance(log, list) else []


def _save_log(log: list) -> None:
    write_json(EMAIL_LOG_PATH, log)


def _render(template_text: str, variables: dict) -> str:
    """Fill a template, tolerating placeholders the caller didn't provide."""
    merged = {**_PLACEHOLDER_DEFAULTS, **variables}
    try:
        return template_text.format(**merged)
    except (KeyError, IndexError):
        # An unexpected placeholder: substitute the ones we know and leave the
        # rest verbatim rather than failing the send.
        def replace(match: re.Match) -> str:
            key = match.group(1)
            return str(merged.get(key, match.group(0)))

        return re.sub(r"\{(\w+)\}", replace, template_text)


def is_valid_email(address: str) -> bool:
    return bool(address and _EMAIL_RE.match(address.strip()))


def send_email(
    to: str,
    template: str = "default",
    variables: Optional[dict] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    lead_id: Optional[str] = None,
) -> dict:
    """
    Send an email (mock). Returns an email record.

    Args:
        to: Recipient email address
        template: Template name from TEMPLATES dict
        variables: Dict of variables to substitute in template (e.g. {"name": "John"})
        subject: Override subject (used with 'default' template)
        body: Override body (used with 'default' template)
        lead_id: Associated CRM lead ID

    Raises:
        EmailError: when the recipient address is missing or malformed.
    """
    if not is_valid_email(to):
        raise EmailError(f"Invalid recipient address: {to!r}")

    variables = dict(variables or {})
    if subject is not None:
        variables.setdefault("subject", subject)
    if body is not None:
        variables.setdefault("body", body)

    tmpl = TEMPLATES.get(template, TEMPLATES["default"])

    record = {
        "id": str(uuid.uuid4()),
        "to": to.strip(),
        "subject": subject or _render(tmpl["subject"], variables),
        "body": body or _render(tmpl["body"], variables),
        "template": template if template in TEMPLATES else "default",
        "lead_id": lead_id,
        "status": "sent",
        "sent_at": _now(),
    }

    # Read-modify-write under the path lock so concurrent sends don't drop
    # each other's records.
    with lock_for(EMAIL_LOG_PATH):
        log = _load_log()
        log.append(record)
        _save_log(log)

    return record


def get_email_stats() -> dict:
    log = _load_log()
    return {
        "total_sent": len(log),
        "by_template": {
            tmpl: sum(1 for e in log if e.get("template") == tmpl)
            for tmpl in TEMPLATES
        },
    }

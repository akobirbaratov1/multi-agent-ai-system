"""
Email Dispatch — Mock SMTP integration
Simulates real email sending (SendGrid/SES compatible structure).
"""

import uuid
from datetime import datetime
from pathlib import Path
import json
from typing import Optional

EMAIL_LOG_PATH = Path("memory/data/email_log.json")

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


def _load_log() -> list:
    if EMAIL_LOG_PATH.exists():
        with open(EMAIL_LOG_PATH) as f:
            return json.load(f)
    return []


def _save_log(log: list):
    import os
    os.makedirs(EMAIL_LOG_PATH.parent, exist_ok=True)
    with open(EMAIL_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)


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
    """
    variables = variables or {}
    tmpl = TEMPLATES.get(template, TEMPLATES["default"])

    resolved_subject = subject or tmpl["subject"].format(**variables)
    resolved_body = body or tmpl["body"].format(**variables)

    record = {
        "id": str(uuid.uuid4()),
        "to": to,
        "subject": resolved_subject,
        "body": resolved_body,
        "template": template,
        "lead_id": lead_id,
        "status": "sent",
        "sent_at": datetime.now().isoformat(),
    }

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

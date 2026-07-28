"""
Mock CRM — Lead Management, Email Dispatch, Follow-up
Simulates real CRM integration (HubSpot, Salesforce compatible structure)
"""

import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core import config
from core.storage import read_json, write_json

CRM_DATA_PATH = config.data_path("crm_data.json")

_EMPTY: Dict[str, list] = {"leads": [], "activities": [], "emails": []}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MockCRM:
    """
    Mock CRM system simulating real CRM functionality.
    Data structure compatible with HubSpot/Salesforce.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        data = read_json(CRM_DATA_PATH, default=None)
        if not isinstance(data, dict):
            return {key: list(value) for key, value in _EMPTY.items()}
        # Tolerate a partial document written by an older build.
        for key in _EMPTY:
            data.setdefault(key, [])
        return data

    def _save_data(self) -> None:
        write_json(CRM_DATA_PATH, self.data)

    def create_lead(
        self,
        user_id: str,
        stage: str,
        score: int,
        insights: List[str],
        contact_info: Optional[Dict] = None,
    ) -> Dict:
        """Create a new lead in CRM"""

        timestamp = _now()
        lead = {
            "id": f"LEAD-{str(uuid.uuid4())[:8].upper()}",
            "user_id": user_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "stage": stage,
            "score": score,
            "insights": insights,
            "contact_info": contact_info or {},
            "status": "active",
            "source": "ai_agent",
            "tags": ["ai_qualified"],
        }

        with self._lock:
            self.data["leads"].append(lead)
            self._save_data()

        return lead

    def update_lead(self, lead_id: str, updates: Dict) -> Optional[Dict]:
        """Update existing lead"""

        with self._lock:
            for lead in self.data["leads"]:
                if lead["id"] == lead_id:
                    lead.update(updates)
                    lead["updated_at"] = _now()
                    self._save_data()
                    return lead
        return None

    def log_activity(
        self,
        lead_id: str,
        activity: str,
        notes: str = "",
        agent: str = "ai",
    ) -> Dict:
        """Log CRM activity"""

        activity_record = {
            "id": str(uuid.uuid4()),
            "lead_id": lead_id,
            "activity": activity,
            "notes": notes,
            "agent": agent,
            "timestamp": _now(),
        }

        with self._lock:
            self.data["activities"].append(activity_record)
            self._save_data()

        return activity_record

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        template: str = "default",
        lead_id: Optional[str] = None,
    ) -> Dict:
        """Mock email dispatch"""

        email_record = {
            "id": str(uuid.uuid4()),
            "to": to,
            "subject": subject,
            "body": body,
            "template": template,
            "lead_id": lead_id,
            "status": "sent",  # Mock: always succeeds
            "sent_at": _now(),
        }

        with self._lock:
            self.data["emails"].append(email_record)
            self._save_data()

        return email_record

    def get_lead(self, lead_id: str) -> Optional[Dict]:
        """Get lead by ID"""
        with self._lock:
            for lead in self.data["leads"]:
                if lead["id"] == lead_id:
                    return dict(lead)
        return None

    def get_leads_by_stage(self, stage: str) -> List[Dict]:
        """Get all leads in a stage"""
        with self._lock:
            return [dict(lead) for lead in self.data["leads"] if lead.get("stage") == stage]

    def list_leads(self) -> List[Dict]:
        """Return a snapshot of all leads."""
        with self._lock:
            return [dict(lead) for lead in self.data["leads"]]

    def list_activities(self) -> List[Dict]:
        """Return a snapshot of all logged activities."""
        with self._lock:
            return [dict(a) for a in self.data["activities"]]

    def get_stats(self) -> Dict:
        """CRM statistics dashboard"""

        with self._lock:
            leads = list(self.data["leads"])
            activity_count = len(self.data["activities"])
            email_count = len(self.data["emails"])

        stages: Dict[str, int] = {}
        for lead in leads:
            stage = lead.get("stage", "unknown")
            stages[stage] = stages.get(stage, 0) + 1

        avg_score = (
            sum(lead.get("score", 0) for lead in leads) / len(leads)
            if leads else 0
        )

        return {
            "total_leads": len(leads),
            "total_activities": activity_count,
            "total_emails": email_count,
            "leads_by_stage": stages,
            "average_lead_score": round(avg_score, 1),
            "active_leads": sum(1 for lead in leads if lead.get("status") == "active"),
        }


# ── shared instance ──────────────────────────────────────
#
# Each importer used to build its own MockCRM. Because the whole document is
# read once at construction and rewritten wholesale on every save, two
# instances silently clobbered each other: a lead created by the sales agent
# never appeared in /crm/leads, and the next write from either instance dropped
# the other's records. One process-wide instance removes the lost-update window.

_crm: Optional[MockCRM] = None
_crm_guard = threading.Lock()


def get_crm() -> MockCRM:
    """Return the process-wide CRM instance, constructing it on first use."""
    global _crm
    if _crm is None:
        with _crm_guard:
            if _crm is None:
                _crm = MockCRM()
    return _crm


def reset_crm() -> None:
    """Drop the shared instance. Used by tests that redirect the data paths."""
    global _crm
    with _crm_guard:
        _crm = None

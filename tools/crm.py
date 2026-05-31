"""
Mock CRM — Lead Management, Email Dispatch, Follow-up
Simulates real CRM integration (HubSpot, Salesforce compatible structure)
"""

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

CRM_DATA_PATH = Path("memory/data/crm_data.json")


class MockCRM:
    """
    Mock CRM system simulating real CRM functionality.
    Data structure compatible with HubSpot/Salesforce.
    """

    def __init__(self):
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        if CRM_DATA_PATH.exists():
            with open(CRM_DATA_PATH) as f:
                return json.load(f)
        return {"leads": [], "activities": [], "emails": []}

    def _save_data(self):
        import os
        os.makedirs(CRM_DATA_PATH.parent, exist_ok=True)
        with open(CRM_DATA_PATH, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def create_lead(
        self,
        user_id: str,
        stage: str,
        score: int,
        insights: List[str],
        contact_info: Optional[Dict] = None,
    ) -> Dict:
        """Create a new lead in CRM"""

        lead = {
            "id": f"LEAD-{str(uuid.uuid4())[:8].upper()}",
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "stage": stage,
            "score": score,
            "insights": insights,
            "contact_info": contact_info or {},
            "status": "active",
            "source": "ai_agent",
            "tags": ["ai_qualified"],
        }

        self.data["leads"].append(lead)
        self._save_data()

        return lead

    def update_lead(self, lead_id: str, updates: Dict) -> Optional[Dict]:
        """Update existing lead"""

        for lead in self.data["leads"]:
            if lead["id"] == lead_id:
                lead.update(updates)
                lead["updated_at"] = datetime.now().isoformat()
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
            "timestamp": datetime.now().isoformat(),
        }

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
            "sent_at": datetime.now().isoformat(),
        }

        self.data["emails"].append(email_record)
        self._save_data()

        return email_record

    def get_lead(self, lead_id: str) -> Optional[Dict]:
        """Get lead by ID"""
        for lead in self.data["leads"]:
            if lead["id"] == lead_id:
                return lead
        return None

    def get_leads_by_stage(self, stage: str) -> List[Dict]:
        """Get all leads in a stage"""
        return [l for l in self.data["leads"] if l["stage"] == stage]

    def get_stats(self) -> Dict:
        """CRM statistics dashboard"""

        leads = self.data["leads"]
        stages = {}
        for lead in leads:
            stage = lead.get("stage", "unknown")
            stages[stage] = stages.get(stage, 0) + 1

        avg_score = (
            sum(l.get("score", 0) for l in leads) / len(leads)
            if leads else 0
        )

        return {
            "total_leads": len(leads),
            "total_activities": len(self.data["activities"]),
            "total_emails": len(self.data["emails"]),
            "leads_by_stage": stages,
            "average_lead_score": round(avg_score, 1),
            "active_leads": sum(1 for l in leads if l.get("status") == "active"),
        }

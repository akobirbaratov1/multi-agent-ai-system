"""CRM management routes."""

from fastapi import APIRouter, HTTPException
from tools.crm import MockCRM

router = APIRouter(prefix="/crm", tags=["CRM"])

_crm = MockCRM()


@router.get("/stats")
async def crm_stats():
    """CRM dashboard statistics."""
    return _crm.get_stats()


@router.get("/leads")
async def get_leads():
    """Get all CRM leads."""
    return {"leads": _crm.data["leads"]}


@router.get("/leads/{lead_id}")
async def get_lead(lead_id: str):
    """Get a specific lead by ID."""
    lead = _crm.get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} not found")
    return lead


@router.get("/activities")
async def get_activities():
    """Get all logged CRM activities."""
    return {"activities": _crm.data["activities"]}

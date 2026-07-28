"""CRM management routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import require_admin
from tools.crm import get_crm

router = APIRouter(prefix="/crm", tags=["CRM"])


@router.get("/stats")
async def crm_stats():
    """CRM dashboard statistics."""
    return get_crm().get_stats()


@router.get("/leads", dependencies=[Depends(require_admin)])
async def get_leads(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Get CRM leads (paginated).

    Admin-only: lead records carry user identifiers and qualification notes.
    """
    leads = get_crm().list_leads()
    return {
        "leads": leads[offset:offset + limit],
        "total": len(leads),
        "limit": limit,
        "offset": offset,
    }


@router.get("/leads/{lead_id}", dependencies=[Depends(require_admin)])
async def get_lead(lead_id: str):
    """Get a specific lead by ID."""
    lead = get_crm().get_lead(lead_id)
    if not lead:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Lead {lead_id} not found"
        )
    return lead


@router.get("/activities", dependencies=[Depends(require_admin)])
async def get_activities(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get logged CRM activities (paginated)."""
    activities = get_crm().list_activities()
    return {
        "activities": activities[offset:offset + limit],
        "total": len(activities),
        "limit": limit,
        "offset": offset,
    }

"""
Operator Dashboard API — Human-in-the-Loop management.
Serves pending review cases and lets operators approve/reject/override.

The queue itself lives in `core.review_queue` so the orchestrator can enqueue
cases without importing the web layer.
"""

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from api.deps import require_admin
from core import config
from core.review_queue import PENDING_PATH, add_pending_case, get_stats, load_cases, resolve_case

router = APIRouter(prefix="/operator", tags=["Operator Dashboard"])

__all__ = ["router", "add_pending_case", "PENDING_PATH"]


class ReviewDecision(BaseModel):
    decision: Literal["approve", "reject", "override"]
    operator_response: Optional[str] = Field(None, max_length=5000)


@router.get("/", include_in_schema=False)
async def operator_dashboard():
    """Serve the operator dashboard HTML."""
    dashboard = config.BASE_DIR / "web" / "operator.html"
    if not dashboard.is_file():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Operator dashboard is not bundled with this build"},
        )
    return FileResponse(str(dashboard))


@router.get("/cases", dependencies=[Depends(require_admin)])
async def list_cases(
    review_status: Optional[Literal["pending", "resolved"]] = Query(
        None, alias="status"
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    List review cases, optionally filtered by status.

    Admin-only: cases contain the user's verbatim message and the agent's
    draft reply.
    """
    cases = load_cases()
    pending_total = sum(1 for c in cases if c.get("status") == "pending")

    if review_status:
        cases = [c for c in cases if c.get("status") == review_status]

    ordered = sorted(cases, key=lambda c: c.get("created_at") or "", reverse=True)

    return {
        "cases": ordered[offset:offset + limit],
        "total": len(ordered),
        "pending": pending_total,
        "limit": limit,
        "offset": offset,
    }


@router.post("/cases/{case_id}/review", dependencies=[Depends(require_admin)])
async def review_case(case_id: str, body: ReviewDecision):
    """Approve, reject, or override an agent response."""
    try:
        case = resolve_case(
            case_id=case_id,
            decision=body.decision,
            operator_response=body.operator_response,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Case {case_id} not found"
        )

    return {"status": "resolved", "case": case}


@router.get("/stats", dependencies=[Depends(require_admin)])
async def operator_stats():
    """Operator dashboard statistics."""
    return get_stats()

"""Monitoring & feedback routes."""

from datetime import date

from fastapi import APIRouter, Depends, Query

from api.deps import require_admin
from api.schemas import FeedbackRequest, FeedbackResponse
from core.review_queue import load_cases
from evaluation.feedback import get_feedback_stats, submit_feedback
from monitoring.langsmith import get_trace_status
from monitoring.metrics import get_summary

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/metrics")
async def metrics():
    """Aggregated system metrics."""
    return get_summary()


@router.get("/tracing")
async def tracing_status():
    """LangSmith tracing status."""
    return get_trace_status()


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """Submit human feedback on an agent response."""
    record = submit_feedback(
        trace_id=request.trace_id,
        session_id=request.session_id,
        rating=request.rating,
        comment=request.comment,
        agent=request.agent,
    )
    return {"id": record["id"], "status": "recorded", "sentiment": record["sentiment"]}


@router.get("/feedback/stats")
async def feedback_stats():
    """Feedback aggregation statistics."""
    return get_feedback_stats()


# The reporting endpoints below scan the full metrics/feedback history and are
# an easy way to burn CPU, so they sit behind the admin guard.

@router.get("/analysis", dependencies=[Depends(require_admin)])
async def performance_analysis(days: int = Query(7, ge=1, le=90)):
    """System performance analysis for the past N days."""
    from feedback_loop.analyzer import analyze_performance

    return analyze_performance(days=days)


@router.get("/improvements", dependencies=[Depends(require_admin)])
async def improvement_suggestions():
    """AI-generated improvement suggestions based on recent performance."""
    from feedback_loop.improver import run_improvement_cycle

    return run_improvement_cycle()


@router.get("/report", dependencies=[Depends(require_admin)])
async def weekly_report():
    """Full weekly performance report."""
    from feedback_loop.reporter import generate_weekly_report

    return generate_weekly_report()


@router.get("/followups", dependencies=[Depends(require_admin)])
async def followup_stats():
    """Follow-up queue statistics."""
    from tools.followup import get_stats

    return get_stats()


@router.get("/operator/summary", dependencies=[Depends(require_admin)])
async def operator_summary():
    """Human-in-the-loop queue summary."""
    cases = load_cases()
    today = date.today().isoformat()
    return {
        "pending": sum(1 for c in cases if c.get("status") == "pending"),
        "resolved_today": sum(
            1 for c in cases
            if c.get("status") == "resolved"
            and (c.get("resolved_at") or "")[:10] == today
        ),
    }

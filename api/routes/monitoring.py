"""Monitoring & feedback routes."""

from fastapi import APIRouter
from api.schemas import FeedbackRequest, FeedbackResponse
from monitoring.metrics import get_summary
from monitoring.langsmith import get_trace_status
from evaluation.feedback import submit_feedback, get_feedback_stats

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


@router.get("/analysis")
async def performance_analysis(days: int = 7):
    """System performance analysis for the past N days."""
    from feedback_loop.analyzer import analyze_performance
    return analyze_performance(days=days)


@router.get("/improvements")
async def improvement_suggestions():
    """AI-generated improvement suggestions based on recent performance."""
    from feedback_loop.improver import run_improvement_cycle
    return run_improvement_cycle()


@router.get("/report")
async def weekly_report():
    """Full weekly performance report."""
    from feedback_loop.reporter import generate_weekly_report
    return generate_weekly_report()


@router.get("/followups")
async def followup_stats():
    """Follow-up queue statistics."""
    from tools.followup import get_stats
    return get_stats()


@router.get("/operator/summary")
async def operator_summary():
    """Human-in-the-loop queue summary."""
    from api.routes.operator import _load
    cases = _load()
    return {
        "pending": sum(1 for c in cases if c["status"] == "pending"),
        "resolved_today": sum(
            1 for c in cases
            if c["status"] == "resolved"
            and c.get("resolved_at", "")[:10] == __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        ),
    }

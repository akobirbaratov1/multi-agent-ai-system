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

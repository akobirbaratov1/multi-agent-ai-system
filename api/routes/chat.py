"""Chat route — main agent interaction endpoint."""

from fastapi import APIRouter, HTTPException, status

from agents.orchestrator import process_message
from api.schemas import ChatRequest, ChatResponse
from core.logging_config import get_logger
from monitoring.metrics import record_request

router = APIRouter(tags=["Chat"])
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user message through the multi-agent system.

    Routes to Sales, Support, or Research agent based on intent classification.
    Applies RAG retrieval, confidence scoring, and optional CRM logging.
    """
    try:
        result = process_message(
            user_id=request.user_id,
            message=request.message,
            session_id=request.session_id,
            interface=request.interface,
            conversation_history=[m.model_dump() for m in request.conversation_history],
        )
    except Exception as exc:
        # `process_message` already degrades internally; reaching here means
        # something outside the graph failed. Log the detail and return an
        # opaque message — the previous handler echoed str(e) to the caller,
        # leaking internal errors to anyone who could trigger a fault.
        logger.exception("Chat request failed", extra={"user_id": request.user_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The assistant is temporarily unavailable. Please try again.",
        ) from exc

    # Metrics must never break the response the user is waiting on.
    try:
        record_request(
            agent=result.get("agent", "unknown"),
            intent=result.get("intent", "unknown"),
            latency_ms=result.get("latency_ms", 0),
            rag_used=result.get("rag_used", False),
            confidence=result.get("confidence", 0),
            error=result.get("error"),
        )
    except Exception:
        logger.warning("Failed to record request metrics", exc_info=True)

    return result

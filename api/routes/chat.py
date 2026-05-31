"""Chat route — main agent interaction endpoint."""

from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse
from agents.orchestrator import process_message
from monitoring.metrics import record_request

router = APIRouter(tags=["Chat"])


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
            conversation_history=request.conversation_history or [],
        )

        record_request(
            agent=result.get("agent", "unknown"),
            intent=result.get("intent", "unknown"),
            latency_ms=result.get("latency_ms", 0),
            rag_used=result.get("rag_used", False),
            confidence=result.get("confidence", 0),
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

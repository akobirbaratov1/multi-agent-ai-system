"""
Main Orchestrator — LangGraph State Machine
Coordinates all agents in the Multi-Agent AI System
"""

import time
import uuid
from typing import Optional

from langgraph.graph import END, StateGraph

from agents.research_agent import research_agent
from agents.sales_agent import sales_agent
from agents.support_agent import support_agent
from core import config
from core.logging_config import get_logger
from core.review_queue import add_pending_case
from core.router import classify_intent, route_to_agent
from core.security import sanitize_input, security_check
from core.state import AgentState

logger = get_logger(__name__)

SECURITY_PREFIX = "Security: "


def security_check_node(state: AgentState) -> AgentState:
    """Security Check — rate limiting, injection detection, sanitization."""
    raw_message = state.get("user_message") or ""
    clean_message = sanitize_input(raw_message)

    passed, reason = security_check(
        user_id=state.get("user_id") or "anonymous",
        message=clean_message,
    )

    if not passed:
        return {
            **state,
            "user_message": clean_message,
            "is_valid": False,
            "is_spam": False,
            "validation_reason": f"{SECURITY_PREFIX}{reason}",
        }

    return {**state, "user_message": clean_message}


def validate_input(state: AgentState) -> AgentState:
    """Input validation — spam filter, basic checks"""

    message = (state.get("user_message") or "").strip()

    if len(message) < 2:
        return {**state, "is_valid": False, "is_spam": False,
                "validation_reason": "Message too short"}

    if len(message) > config.MAX_MESSAGE_LENGTH:
        return {**state, "is_valid": False, "is_spam": False,
                "validation_reason": "Message too long"}

    spam_patterns = ["buy now!!!", "click here!!!", "free money"]
    is_spam = any(pattern in message.lower() for pattern in spam_patterns)

    if is_spam:
        return {**state, "is_valid": False, "is_spam": True,
                "validation_reason": "Spam detected"}

    return {**state, "is_valid": True, "is_spam": False}


def reject_message(state: AgentState) -> AgentState:
    """Handle invalid/spam messages"""

    reason = state.get("validation_reason") or "Invalid message"

    if state.get("is_spam"):
        response = "Your message was flagged as spam. Please rephrase."
    elif reason.startswith(SECURITY_PREFIX):
        # Don't echo which rule tripped — that tells a prober how to evade it.
        response = "Your request could not be processed. Please try again shortly."
    else:
        response = f"Invalid message: {reason}"

    return {**state, "final_response": response}


def human_review_node(state: AgentState) -> AgentState:
    """Human-in-the-loop — low confidence cases stored for operator review."""
    try:
        add_pending_case(
            session_id=state.get("session_id") or "",
            user_id=state.get("user_id") or "",
            user_message=state.get("user_message") or "",
            agent_response=state.get("agent_response") or "",
            intent=str(state.get("intent") or "unknown"),
            confidence=state.get("intent_confidence") or 0.0,
            trace_id=state.get("trace_id") or "",
        )
    except Exception:
        # Never fail the user's turn because the review queue is unwritable —
        # but do surface it, otherwise escalations vanish silently.
        logger.exception("Failed to enqueue human review case")

    return {
        **state,
        "final_response": (
            "I'm not fully confident in my response. "
            "A human operator will review and respond shortly. "
            f"Your session ID: {state.get('session_id')}"
        ),
        "requires_human": True,
    }


def should_continue(state: AgentState) -> str:
    """Conditional edge — check validation result"""

    if not state.get("is_valid") or state.get("is_spam"):
        return "reject"
    return "classify"


def after_security(state: AgentState) -> str:
    """Conditional edge after security check."""
    # `validation_reason` is initialised to None, so `state.get(key, "")` returns
    # None rather than the default — the previous `.startswith` on it raised
    # AttributeError on every single request and turned /chat into a hard 500.
    reason = state.get("validation_reason") or ""
    if not state.get("is_valid") and reason.startswith(SECURITY_PREFIX):
        return "reject"
    return "validate"


def build_graph() -> StateGraph:
    """Build the LangGraph workflow"""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("security", security_check_node)
    workflow.add_node("validate", validate_input)
    workflow.add_node("classify", classify_intent)
    workflow.add_node("sales_agent", sales_agent)
    workflow.add_node("support_agent", support_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("reject", reject_message)

    # Entry point — security first
    workflow.set_entry_point("security")

    # Security → validate or reject
    workflow.add_conditional_edges(
        "security",
        after_security,
        {"reject": "reject", "validate": "validate"},
    )

    # Edges
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "reject": "reject",
            "classify": "classify",
        }
    )

    workflow.add_conditional_edges(
        "classify",
        route_to_agent,
        {
            "sales_agent": "sales_agent",
            "support_agent": "support_agent",
            "research_agent": "research_agent",
            "human_review": "human_review",
            "reject": "reject",
        }
    )

    # All agents lead to END
    workflow.add_edge("sales_agent", END)
    workflow.add_edge("support_agent", END)
    workflow.add_edge("research_agent", END)
    workflow.add_edge("human_review", END)
    workflow.add_edge("reject", END)

    return workflow.compile()


# Global graph instance
graph = build_graph()


def _enum_value(value, fallback: str = "unknown") -> str:
    """Render an enum/None state field as the plain string the API returns."""
    if value is None:
        return fallback
    return getattr(value, "value", str(value))


def process_message(
    user_id: str,
    message: str,
    session_id: Optional[str] = None,
    interface: str = "web",
    conversation_history: Optional[list] = None,
) -> dict:
    """Main entry point for processing user messages"""

    start_time = time.time()
    resolved_session_id = session_id or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())

    initial_state: AgentState = {
        "user_id": user_id,
        "session_id": resolved_session_id,
        "user_message": message,
        "interface": interface,
        "is_valid": False,
        "is_spam": False,
        "validation_reason": None,
        "user_context": {},
        "conversation_history": conversation_history or [],
        "intent": None,
        "intent_confidence": 0.0,
        "assigned_agent": None,
        "agent_response": None,
        "agent_metadata": {},
        "retrieved_documents": [],
        "rag_used": False,
        "confidence_level": None,
        "requires_human": False,
        "human_approved": None,
        "human_feedback": None,
        "lead_created": False,
        "lead_id": None,
        "email_sent": False,
        "crm_actions": [],
        "final_response": "",
        "response_metadata": {},
        "trace_id": trace_id,
        "latency_ms": 0.0,
        "tokens_used": 0,
        "error": None,
    }

    try:
        result = graph.invoke(initial_state)
        error: Optional[str] = result.get("error")
    except Exception as exc:
        # A failure anywhere in the graph should still produce a usable turn.
        logger.exception(
            "Orchestration failed",
            extra={"trace_id": trace_id, "session_id": resolved_session_id},
        )
        result = {
            **initial_state,
            "final_response": (
                "Something went wrong while processing your request. "
                "Please try again in a moment."
            ),
            "error": type(exc).__name__,
        }
        error = type(exc).__name__

    latency = (time.time() - start_time) * 1000

    # `assigned_agent` / `intent` stay None on the reject and error paths, and
    # the response schema types both as `str` — returning None there failed
    # response validation and turned a clean rejection into a 500.
    return {
        "response": result.get("final_response") or "I could not process your request.",
        "session_id": result.get("session_id") or resolved_session_id,
        "agent": _enum_value(result.get("assigned_agent")),
        "intent": _enum_value(result.get("intent")),
        "confidence": float(result.get("intent_confidence") or 0.0),
        "rag_used": bool(result.get("rag_used")),
        "requires_human": bool(result.get("requires_human")),
        "crm_actions": result.get("crm_actions") or [],
        "latency_ms": round(latency, 2),
        "trace_id": result.get("trace_id") or trace_id,
        "metadata": result.get("agent_metadata") or {},
        "error": error,
    }

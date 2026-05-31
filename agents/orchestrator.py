"""
Main Orchestrator — LangGraph State Machine
Coordinates all agents in the Multi-Agent AI System
"""

import uuid
import time
from langgraph.graph import StateGraph, END
from core.state import AgentState, IntentType
from core.router import classify_intent, route_to_agent
from agents.sales_agent import sales_agent
from agents.support_agent import support_agent
from agents.research_agent import research_agent


def validate_input(state: AgentState) -> AgentState:
    """Input validation — spam filter, basic checks"""

    message = state.get("user_message", "").strip()

    # Basic validation
    if not message or len(message) < 2:
        return {**state, "is_valid": False, "is_spam": False,
                "validation_reason": "Message too short"}

    if len(message) > 5000:
        return {**state, "is_valid": False, "is_spam": False,
                "validation_reason": "Message too long"}

    # Simple spam detection
    spam_patterns = ["buy now!!!", "click here!!!", "free money"]
    is_spam = any(pattern in message.lower() for pattern in spam_patterns)

    if is_spam:
        return {**state, "is_valid": False, "is_spam": True,
                "validation_reason": "Spam detected"}

    return {**state, "is_valid": True, "is_spam": False}


def reject_message(state: AgentState) -> AgentState:
    """Handle invalid/spam messages"""

    reason = state.get("validation_reason", "Invalid message")
    response = (
        "Your message was flagged as spam. Please rephrase."
        if state.get("is_spam")
        else f"Invalid message: {reason}"
    )

    return {**state, "final_response": response}


def human_review_node(state: AgentState) -> AgentState:
    """Human-in-the-loop — low confidence cases"""

    return {
        **state,
        "final_response": (
            "I'm not fully confident in my response. "
            "A human agent will review and respond shortly. "
            f"Your session ID: {state.get('session_id')}"
        ),
        "requires_human": True,
    }


def should_continue(state: AgentState) -> str:
    """Conditional edge — check validation result"""

    if not state.get("is_valid") or state.get("is_spam"):
        return "reject"
    return "classify"


def build_graph() -> StateGraph:
    """Build the LangGraph workflow"""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("validate", validate_input)
    workflow.add_node("classify", classify_intent)
    workflow.add_node("sales_agent", sales_agent)
    workflow.add_node("support_agent", support_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("reject", reject_message)

    # Entry point
    workflow.set_entry_point("validate")

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


def process_message(
    user_id: str,
    message: str,
    session_id: str = None,
    interface: str = "web",
    conversation_history: list = None,
) -> dict:
    """Main entry point for processing user messages"""

    start_time = time.time()

    initial_state: AgentState = {
        "user_id": user_id,
        "session_id": session_id or str(uuid.uuid4()),
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
        "trace_id": str(uuid.uuid4()),
        "latency_ms": 0.0,
        "tokens_used": 0,
        "error": None,
    }

    result = graph.invoke(initial_state)

    latency = (time.time() - start_time) * 1000
    result["latency_ms"] = latency

    return {
        "response": result.get("final_response", "I could not process your request."),
        "session_id": result["session_id"],
        "agent": result.get("assigned_agent", "unknown"),
        "intent": result.get("intent", "unknown"),
        "confidence": result.get("intent_confidence", 0.0),
        "rag_used": result.get("rag_used", False),
        "requires_human": result.get("requires_human", False),
        "crm_actions": result.get("crm_actions", []),
        "latency_ms": round(latency, 2),
        "trace_id": result["trace_id"],
        "metadata": result.get("agent_metadata", {}),
    }

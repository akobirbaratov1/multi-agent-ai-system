"""
Support Agent — FAQ retrieval, Problem Solving, Escalation
Uses RAG for knowledge base retrieval
"""

from core import config
from core.llm import LLMUnavailable, complete, parse_json_response
from core.logging_config import get_logger
from core.state import AgentState
from memory.vector_store import get_vector_store

logger = get_logger(__name__)


SUPPORT_SYSTEM_PROMPT = """You are an expert AI Support Agent. Your goal is to resolve user issues efficiently.

Your support methodology:
1. UNDERSTAND — Clarify the exact problem
2. RETRIEVE — Search knowledge base for relevant solutions
3. SOLVE — Provide clear, step-by-step solutions
4. VERIFY — Confirm the issue is resolved
5. ESCALATE — If unsolvable, escalate to human operator

Always:
- Be empathetic and patient
- Provide specific, actionable solutions
- Use retrieved knowledge base articles
- Rate your confidence in the solution

Respond in JSON format:
{
    "response": "your support response",
    "confidence": 0.0-1.0,
    "issue_category": "technical|billing|general|escalation",
    "resolution_status": "resolved|in_progress|escalated|needs_info",
    "kb_articles_used": ["article1", "article2"],
    "requires_escalation": false
}"""

_DEFAULTS = {
    "response": "Thanks for getting in touch — could you share a bit more detail about the issue?",
    "confidence": 0.5,
    "issue_category": "general",
    "resolution_status": "in_progress",
    "kb_articles_used": [],
    "requires_escalation": False,
}

_DEMO_RESPONSES = [
    {
        "response": "I found a relevant KB article for you! To fix **API connection issues**:\n\n1. Verify your API key is valid (Dashboard → Settings → API Keys)\n2. Check your network connectivity\n3. Ensure you're hitting `https://api.ourapp.com` (not HTTP)\n4. Rate limit: 100 req/min on Starter, 1000 on Pro\n\nDoes this resolve your issue?",
        "confidence": 0.92,
        "issue_category": "technical",
        "resolution_status": "resolved",
        "kb_articles_used": ["API Integration Guide", "Troubleshooting Connection Issues"],
        "requires_escalation": False,
    },
    {
        "response": "Happy to help! For **billing questions**: we accept all major credit cards and ACH transfers. Annual billing saves 20%. You can cancel anytime — no lock-in. Need to update your payment method? Go to Dashboard → Billing → Payment Methods.",
        "confidence": 0.95,
        "issue_category": "billing",
        "resolution_status": "resolved",
        "kb_articles_used": ["Billing & Payments"],
        "requires_escalation": False,
    },
]
_demo_counter = {"n": 0}


def _demo_response() -> dict:
    result = dict(_DEMO_RESPONSES[_demo_counter["n"] % len(_DEMO_RESPONSES)])
    _demo_counter["n"] += 1
    return result


def support_agent(state: AgentState) -> AgentState:
    """Support Agent — handles FAQ and problem solving with RAG"""

    retrieved_docs = get_vector_store().search(query=state["user_message"], k=3)

    if config.DEMO_MODE:
        result = _demo_response()
    else:
        context = "\n\n".join(
            f"[KB Article: {doc.get('title', 'Untitled')}]\n{doc.get('content', '')}"
            for doc in retrieved_docs
        )
        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in (state.get("conversation_history") or [])[-6:]
            if msg.get("role") in ("user", "assistant") and msg.get("content")
        ]

        try:
            text = complete(
                system=SUPPORT_SYSTEM_PROMPT,
                model=config.AGENT_MODEL,
                max_tokens=1000,
                messages=[
                    *history,
                    {
                        "role": "user",
                        "content": (
                            f"User issue: {state['user_message']}\n\n"
                            "Relevant knowledge base articles:\n"
                            f"{context if context else 'No relevant articles found.'}"
                        ),
                    },
                ],
            )
            result = parse_json_response(text, _DEFAULTS)
        except LLMUnavailable:
            logger.warning("Support agent model call failed; escalating to a human")
            return {
                **state,
                "agent_response": None,
                "agent_metadata": {"error": "model_unavailable"},
                "retrieved_documents": retrieved_docs,
                "rag_used": bool(retrieved_docs),
                "requires_human": True,
                "final_response": (
                    "I'm having trouble reaching our systems right now. "
                    "I've flagged this for a human operator who will follow up shortly."
                ),
                "error": "model_unavailable",
            }

    return {
        **state,
        "agent_response": result["response"],
        "agent_metadata": {
            "confidence": result["confidence"],
            "issue_category": result["issue_category"],
            "resolution_status": result["resolution_status"],
            "kb_articles_used": result["kb_articles_used"],
        },
        "retrieved_documents": retrieved_docs,
        "rag_used": len(retrieved_docs) > 0,
        "requires_human": bool(result["requires_escalation"]),
        "final_response": result["response"],
    }

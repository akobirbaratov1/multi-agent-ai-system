"""
Support Agent — FAQ retrieval, Problem Solving, Escalation
Uses RAG for knowledge base retrieval
"""

import os
import json
from anthropic import Anthropic
from core.state import AgentState
from memory.vector_store import FAISSVectorStore

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
client = Anthropic() if not DEMO_MODE else None
vector_store = FAISSVectorStore()


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


def _demo_response(docs: list) -> dict:
    result = _DEMO_RESPONSES[_demo_counter["n"] % len(_DEMO_RESPONSES)].copy()
    _demo_counter["n"] += 1
    return result


def support_agent(state: AgentState) -> AgentState:
    """Support Agent — handles FAQ and problem solving with RAG"""

    retrieved_docs = vector_store.search(query=state["user_message"], k=3)

    if DEMO_MODE:
        result = _demo_response(retrieved_docs)
    else:
        context = "\n\n".join([
            f"[KB Article: {doc['title']}]\n{doc['content']}"
            for doc in retrieved_docs
        ])

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SUPPORT_SYSTEM_PROMPT,
            messages=[
                *[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in state.get("conversation_history", [])[-6:]
                ],
                {
                    "role": "user",
                    "content": f"User issue: {state['user_message']}\n\nRelevant knowledge base articles:\n{context if context else 'No relevant articles found.'}",
                },
            ],
        )

        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            result = {
                "response": response.content[0].text,
                "confidence": 0.5,
                "issue_category": "general",
                "resolution_status": "in_progress",
                "kb_articles_used": [],
                "requires_escalation": False,
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
        "requires_human": result["requires_escalation"],
        "final_response": result["response"],
    }

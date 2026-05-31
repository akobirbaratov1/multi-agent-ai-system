"""
Intent Classification & Agent Router
Uses Claude to classify user intent and route to appropriate agent
"""

import os
import json
from anthropic import Anthropic
from core.state import AgentState, IntentType, AgentType, ConfidenceLevel

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
client = Anthropic() if not DEMO_MODE else None

_DEMO_ROUTING = {
    ("price", "pricing", "plan", "buy", "demo", "cost", "enterprise", "sales"): "sales",
    ("help", "issue", "error", "problem", "support", "fix", "broken", "api", "connect"): "support",
    ("what", "how", "tell", "explain", "research", "info", "learn", "about"): "research",
}


ROUTER_SYSTEM_PROMPT = """You are an expert intent classifier for a Multi-Agent AI System.

Your job is to analyze user messages and classify them into one of these intents:
- SALES: User wants to buy, learn about pricing, request demo, interested in products/services
- SUPPORT: User has a problem, needs help, asking about how something works, complaints
- RESEARCH: User wants information, data, reports, analysis, general knowledge

Respond ONLY with valid JSON in this exact format:
{
    "intent": "sales|support|research",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "key_signals": ["signal1", "signal2"]
}"""


def _demo_classify(message: str) -> dict:
    """Keyword-based intent classification for demo mode (no API key needed)."""
    msg = message.lower()
    for keywords, intent in _DEMO_ROUTING.items():
        if any(kw in msg for kw in keywords):
            return {"intent": intent, "confidence": 0.92, "reasoning": "demo mode", "key_signals": []}
    return {"intent": "support", "confidence": 0.75, "reasoning": "demo mode fallback", "key_signals": []}


def classify_intent(state: AgentState) -> AgentState:
    """Classify user intent using Claude (or keyword matching in demo mode)."""

    if DEMO_MODE:
        result = _demo_classify(state["user_message"])
    else:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=ROUTER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"""
User message: "{state['user_message']}"
Conversation history: {json.dumps(state.get('conversation_history', [])[-3:], ensure_ascii=False)}
User context: {json.dumps(state.get('user_context', {}), ensure_ascii=False)}
"""
                }
            ]
        )

        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            result = {"intent": "support", "confidence": 0.5, "reasoning": "parse error", "key_signals": []}

    intent_map = {
        "sales": IntentType.SALES,
        "support": IntentType.SUPPORT,
        "research": IntentType.RESEARCH,
    }

    agent_map = {
        IntentType.SALES: AgentType.SALES,
        IntentType.SUPPORT: AgentType.SUPPORT,
        IntentType.RESEARCH: AgentType.RESEARCH,
    }

    intent = intent_map.get(result["intent"], IntentType.UNKNOWN)
    confidence = result["confidence"]

    if confidence >= 0.85:
        confidence_level = ConfidenceLevel.HIGH
    elif confidence >= 0.60:
        confidence_level = ConfidenceLevel.MEDIUM
    else:
        confidence_level = ConfidenceLevel.LOW

    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "assigned_agent": agent_map.get(intent, AgentType.ORCHESTRATOR),
        "confidence_level": confidence_level,
        "requires_human": confidence_level == ConfidenceLevel.LOW,
    }


def route_to_agent(state: AgentState) -> str:
    """LangGraph conditional edge — route to appropriate agent"""

    if state.get("requires_human"):
        return "human_review"

    if not state.get("is_valid") or state.get("is_spam"):
        return "reject"

    intent = state.get("intent")

    routing = {
        IntentType.SALES: "sales_agent",
        IntentType.SUPPORT: "support_agent",
        IntentType.RESEARCH: "research_agent",
    }

    return routing.get(intent, "support_agent")

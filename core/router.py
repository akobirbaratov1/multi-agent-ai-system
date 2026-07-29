"""
Intent Classification & Agent Router
Uses Claude to classify user intent and route to appropriate agent
"""

import json

from core import config
from core.confidence import compute_confidence_level
from core.llm import LLMUnavailable, complete, parse_json_response
from core.logging_config import get_logger
from core.state import AgentState, AgentType, ConfidenceLevel, IntentType

logger = get_logger(__name__)

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

_ROUTER_DEFAULTS = {
    "intent": "support",
    "confidence": 0.5,
    "reasoning": "",
    "key_signals": [],
}

_INTENT_MAP = {
    "sales": IntentType.SALES,
    "support": IntentType.SUPPORT,
    "research": IntentType.RESEARCH,
}

_AGENT_MAP = {
    IntentType.SALES: AgentType.SALES,
    IntentType.SUPPORT: AgentType.SUPPORT,
    IntentType.RESEARCH: AgentType.RESEARCH,
}


def _demo_classify(message: str) -> dict:
    """Keyword-based intent classification for demo mode (no API key needed)."""
    msg = message.lower()
    for keywords, intent in _DEMO_ROUTING.items():
        if any(kw in msg for kw in keywords):
            return {"intent": intent, "confidence": 0.92, "reasoning": "demo mode", "key_signals": []}
    return {"intent": "support", "confidence": 0.75, "reasoning": "demo mode fallback", "key_signals": []}


def classify_intent(state: AgentState) -> AgentState:
    """Classify user intent using Claude (or keyword matching in demo mode)."""

    if config.DEMO_MODE:
        result = _demo_classify(state["user_message"])
    else:
        history = json.dumps(
            (state.get("conversation_history") or [])[-3:], ensure_ascii=False
        )
        context = json.dumps(state.get("user_context") or {}, ensure_ascii=False)

        try:
            text = complete(
                system=ROUTER_SYSTEM_PROMPT,
                model=config.ROUTER_MODEL,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": (
                        f'User message: "{state["user_message"]}"\n'
                        f"Conversation history: {history}\n"
                        f"User context: {context}\n"
                    ),
                }],
            )
            result = parse_json_response(text, _ROUTER_DEFAULTS)
        except LLMUnavailable:
            # Routing must not take the turn down. Fall back to the keyword
            # classifier with low confidence, which sends the case to a human.
            logger.warning("Intent classification unavailable; falling back to keywords")
            result = _demo_classify(state["user_message"])
            result["confidence"] = 0.0
            result["reasoning"] = "router unavailable"

    intent = _INTENT_MAP.get(str(result.get("intent", "")).lower(), IntentType.UNKNOWN)

    # A malformed confidence must not escape as a string or an out-of-range float.
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    confidence_level = compute_confidence_level(confidence)

    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "assigned_agent": _AGENT_MAP.get(intent, AgentType.ORCHESTRATOR),
        "confidence_level": confidence_level,
        "requires_human": confidence_level == ConfidenceLevel.LOW,
    }


def route_to_agent(state: AgentState) -> str:
    """LangGraph conditional edge — route to appropriate agent"""

    if not state.get("is_valid") or state.get("is_spam"):
        return "reject"

    if state.get("requires_human"):
        return "human_review"

    routing = {
        IntentType.SALES: "sales_agent",
        IntentType.SUPPORT: "support_agent",
        IntentType.RESEARCH: "research_agent",
    }

    return routing.get(state.get("intent"), "support_agent")

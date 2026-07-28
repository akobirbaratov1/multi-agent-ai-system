"""
Sales Agent — Lead Qualification, Questioning, Sales Conversion
Powered by Claude with structured sales methodology
"""

from core import config
from core.llm import LLMUnavailable, complete, parse_json_response
from core.logging_config import get_logger
from core.state import AgentState
from tools.crm import get_crm

logger = get_logger(__name__)


SALES_SYSTEM_PROMPT = """You are an expert AI Sales Agent. Your goal is to qualify leads and drive conversions.

Your sales methodology:
1. QUALIFY — Understand the prospect's needs, budget, timeline, decision authority
2. EDUCATE — Explain how our solution solves their specific problem
3. HANDLE OBJECTIONS — Address concerns professionally and empathetically
4. CONVERT — Guide toward next steps (demo, trial, purchase)

Always:
- Ask one qualifying question at a time
- Listen and adapt to responses
- Be helpful, not pushy
- Provide specific value propositions
- Track lead score based on conversation

Respond in JSON format:
{
    "response": "your sales response",
    "lead_score": 0-100,
    "qualification_stage": "awareness|interest|consideration|intent|purchase",
    "next_action": "continue_conversation|schedule_demo|create_lead|close",
    "key_insights": ["insight1", "insight2"]
}"""

# Every key the node reads below must appear here — `parse_json_response`
# layers the model's reply over these, so a partial or malformed response
# degrades instead of raising KeyError mid-request.
_DEFAULTS = {
    "response": "Thanks for reaching out! Could you tell me a bit more about what you're looking for?",
    "lead_score": 0,
    "qualification_stage": "awareness",
    "next_action": "continue_conversation",
    "key_insights": [],
}

_VALID_STAGES = {"awareness", "interest", "consideration", "intent", "purchase"}
_VALID_ACTIONS = {"continue_conversation", "schedule_demo", "create_lead", "close"}

_DEMO_RESPONSES = [
    {
        "response": "Great question! We offer three plans: **Starter** ($49/mo), **Professional** ($149/mo), and **Enterprise** (custom). All include a 14-day free trial. Which team size are you working with?",
        "lead_score": 45,
        "qualification_stage": "interest",
        "next_action": "continue_conversation",
        "key_insights": ["interested in pricing", "evaluating options"],
    },
    {
        "response": "Excellent! Based on your needs, the Professional plan at $149/mo would be a perfect fit — it includes unlimited API calls, priority support, and custom integrations. Would you like to schedule a live demo?",
        "lead_score": 75,
        "qualification_stage": "consideration",
        "next_action": "schedule_demo",
        "key_insights": ["high engagement", "budget confirmed", "decision timeline: soon"],
    },
]
_demo_counter = {"n": 0}


def _demo_response() -> dict:
    result = _DEMO_RESPONSES[_demo_counter["n"] % len(_DEMO_RESPONSES)]
    _demo_counter["n"] += 1
    return dict(result)


def _normalize(result: dict) -> dict:
    """Clamp model-supplied fields to the values the CRM logic expects."""
    try:
        score = int(result.get("lead_score", 0))
    except (TypeError, ValueError):
        score = 0
    result["lead_score"] = min(max(score, 0), 100)

    if result.get("qualification_stage") not in _VALID_STAGES:
        result["qualification_stage"] = "awareness"
    if result.get("next_action") not in _VALID_ACTIONS:
        result["next_action"] = "continue_conversation"
    if not isinstance(result.get("key_insights"), list):
        result["key_insights"] = []

    return result


def sales_agent(state: AgentState) -> AgentState:
    """Sales Agent — handles lead qualification and conversion"""

    if config.DEMO_MODE:
        result = _demo_response()
    else:
        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in (state.get("conversation_history") or [])[-6:]
            if msg.get("role") in ("user", "assistant") and msg.get("content")
        ]
        try:
            text = complete(
                system=SALES_SYSTEM_PROMPT,
                model=config.AGENT_MODEL,
                max_tokens=1000,
                messages=[*history, {"role": "user", "content": state["user_message"]}],
            )
            result = parse_json_response(text, _DEFAULTS)
        except LLMUnavailable:
            logger.warning("Sales agent model call failed; returning fallback response")
            return {
                **state,
                "agent_response": None,
                "agent_metadata": {"error": "model_unavailable"},
                "final_response": (
                    "I'm having trouble reaching our systems right now. "
                    "Please try again shortly and I'll pick up where we left off."
                ),
                "requires_human": True,
                "error": "model_unavailable",
            }

    result = _normalize(result)

    crm = get_crm()
    crm_actions = []
    lead_id = None
    lead_created = False

    if result["lead_score"] >= 60 and result["next_action"] in ("create_lead", "close"):
        lead = crm.create_lead(
            user_id=state["user_id"],
            stage=result["qualification_stage"],
            score=result["lead_score"],
            insights=result["key_insights"],
        )
        lead_id = lead["id"]
        lead_created = True
        crm_actions.append(f"Lead created: {lead_id}")

    if result["next_action"] == "schedule_demo":
        crm.log_activity(
            lead_id=lead_id or state["user_id"],
            activity="demo_requested",
            notes=result["response"],
        )
        crm_actions.append("Demo request logged")

        # Schedule a 24h follow-up
        from tools.followup import schedule_followup

        schedule_followup(
            lead_id=lead_id or state["user_id"],
            user_id=state["user_id"],
            email=(state.get("user_context") or {}).get("email"),
            reason="demo_requested",
            delay_hours=24,
        )
        crm_actions.append("Follow-up scheduled: 24h")

    return {
        **state,
        "agent_response": result["response"],
        "agent_metadata": {
            "lead_score": result["lead_score"],
            "qualification_stage": result["qualification_stage"],
            "next_action": result["next_action"],
            "key_insights": result["key_insights"],
        },
        "lead_created": lead_created,
        "lead_id": lead_id,
        "crm_actions": crm_actions,
        "final_response": result["response"],
    }

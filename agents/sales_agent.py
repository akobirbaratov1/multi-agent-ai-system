"""
Sales Agent — Lead Qualification, Questioning, Sales Conversion
Powered by Claude with structured sales methodology
"""

import os
import json
from anthropic import Anthropic
from core.state import AgentState
from tools.crm import MockCRM

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
client = Anthropic() if not DEMO_MODE else None
crm = MockCRM()


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
    return result


def sales_agent(state: AgentState) -> AgentState:
    """Sales Agent — handles lead qualification and conversion"""

    if DEMO_MODE:
        result = _demo_response()
    else:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SALES_SYSTEM_PROMPT,
            messages=[
                *[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in state.get("conversation_history", [])[-6:]
                ],
                {"role": "user", "content": state["user_message"]},
            ],
        )

        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            result = {
                "response": response.content[0].text,
                "lead_score": 0,
                "qualification_stage": "awareness",
                "next_action": "continue_conversation",
                "key_insights": [],
            }

    # CRM actions based on lead score
    crm_actions = []
    lead_id = None
    lead_created = False

    if result["lead_score"] >= 60 and result["next_action"] in ["create_lead", "close"]:
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
            lead_id=state["user_id"],
            activity="demo_requested",
            notes=result["response"],
        )
        crm_actions.append("Demo request logged")

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

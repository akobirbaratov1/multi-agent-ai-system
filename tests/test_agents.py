"""
Agent Tests — Unit tests for Sales, Support, and Research agents.
Run: pytest tests/test_agents.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


def make_state(message: str, history: list = None) -> dict:
    return {
        "user_id": "test_user",
        "session_id": "test_session",
        "user_message": message,
        "interface": "test",
        "is_valid": True,
        "is_spam": False,
        "validation_reason": None,
        "user_context": {},
        "conversation_history": history or [],
        "intent": None,
        "intent_confidence": 0.9,
        "assigned_agent": None,
        "agent_response": None,
        "agent_metadata": {},
        "retrieved_documents": [],
        "rag_used": False,
        "confidence_level": "high",
        "requires_human": False,
        "human_approved": None,
        "human_feedback": None,
        "lead_created": False,
        "lead_id": None,
        "email_sent": False,
        "crm_actions": [],
        "final_response": "",
        "response_metadata": {},
        "trace_id": "trace_test_001",
        "latency_ms": 0.0,
        "tokens_used": 0,
        "error": None,
    }


def _mock_client(json_text: str) -> MagicMock:
    """Build a mock Anthropic client that returns the given JSON text."""
    mock = MagicMock()
    mock.messages.create.return_value.content = [MagicMock(text=json_text)]
    return mock


class TestSalesAgent:
    def test_sales_agent_returns_final_response(self):
        """Sales agent must populate final_response."""
        mock_c = _mock_client('{"response": "Great! Let me tell you about our plans.", "lead_score": 40, "qualification_stage": "awareness", "next_action": "continue_conversation", "key_insights": ["interested in pricing"]}')

        with patch("agents.sales_agent.DEMO_MODE", False), \
             patch("agents.sales_agent.client", mock_c):
            from agents.sales_agent import sales_agent
            result = sales_agent(make_state("What is the price?"))

        assert result["final_response"] != ""
        assert "agent_metadata" in result
        assert "lead_score" in result["agent_metadata"]

    def test_sales_agent_creates_lead_on_high_score(self):
        """Sales agent creates a CRM lead when score >= 60 and action is create_lead."""
        mock_c = _mock_client('{"response": "Let us set up a deal.", "lead_score": 80, "qualification_stage": "intent", "next_action": "create_lead", "key_insights": ["high intent", "budget confirmed"]}')

        with patch("agents.sales_agent.DEMO_MODE", False), \
             patch("agents.sales_agent.client", mock_c):
            from agents.sales_agent import sales_agent
            result = sales_agent(make_state("I want to buy the Enterprise plan now"))

        assert result["lead_created"] is True
        assert result["lead_id"] is not None

    def test_sales_agent_handles_json_error(self):
        """Sales agent must not crash when Claude returns non-JSON."""
        mock_c = _mock_client("Sorry, I cannot help with that right now.")

        with patch("agents.sales_agent.DEMO_MODE", False), \
             patch("agents.sales_agent.client", mock_c):
            from agents.sales_agent import sales_agent
            result = sales_agent(make_state("hello"))

        assert result["final_response"] != ""

    def test_sales_agent_demo_mode(self):
        """Sales agent works without API key in demo mode."""
        with patch("agents.sales_agent.DEMO_MODE", True):
            from agents.sales_agent import sales_agent
            result = sales_agent(make_state("Tell me about pricing"))

        assert result["final_response"] != ""
        assert result["agent_metadata"]["lead_score"] >= 0


class TestSupportAgent:
    def test_support_agent_returns_response(self):
        """Support agent must return a response."""
        mock_c = _mock_client('{"response": "Try clearing your cache.", "confidence": 0.9, "issue_category": "technical", "resolution_status": "resolved", "kb_articles_used": [], "requires_escalation": false}')

        with patch("agents.support_agent.DEMO_MODE", False), \
             patch("agents.support_agent.client", mock_c), \
             patch("agents.support_agent.vector_store.search", return_value=[]):
            from agents.support_agent import support_agent
            result = support_agent(make_state("My app is crashing"))

        assert result["final_response"] != ""
        assert result["agent_metadata"]["resolution_status"] == "resolved"

    def test_support_agent_uses_rag(self):
        """Support agent should mark rag_used=True when documents retrieved."""
        mock_c = _mock_client('{"response": "Based on KB article...", "confidence": 0.85, "issue_category": "technical", "resolution_status": "resolved", "kb_articles_used": ["API Guide"], "requires_escalation": false}')
        mock_docs = [{"title": "API Guide", "content": "Check your API key.", "relevance_score": 0.9}]

        with patch("agents.support_agent.DEMO_MODE", False), \
             patch("agents.support_agent.client", mock_c), \
             patch("agents.support_agent.vector_store.search", return_value=mock_docs):
            from agents.support_agent import support_agent
            result = support_agent(make_state("API integration issue"))

        assert result["rag_used"] is True

    def test_support_agent_demo_mode(self):
        """Support agent works without API key in demo mode."""
        with patch("agents.support_agent.DEMO_MODE", True), \
             patch("agents.support_agent.vector_store.search", return_value=[]):
            from agents.support_agent import support_agent
            result = support_agent(make_state("I have a billing issue"))

        assert result["final_response"] != ""


class TestResearchAgent:
    def test_research_agent_returns_response(self):
        """Research agent must return a structured response."""
        mock_c = _mock_client('{"response": "Multi-agent AI involves...", "confidence": 0.9, "sources_used": [], "key_findings": ["finding1"], "follow_up_questions": ["q1"], "requires_more_info": false}')

        with patch("agents.research_agent.DEMO_MODE", False), \
             patch("agents.research_agent.client", mock_c), \
             patch("agents.research_agent.vector_store.search", return_value=[]):
            from agents.research_agent import research_agent
            result = research_agent(make_state("Explain multi-agent AI"))

        assert result["final_response"] != ""
        assert "key_findings" in result["agent_metadata"]

    def test_research_agent_demo_mode(self):
        """Research agent works without API key in demo mode."""
        with patch("agents.research_agent.DEMO_MODE", True), \
             patch("agents.research_agent.vector_store.search", return_value=[]):
            from agents.research_agent import research_agent
            result = research_agent(make_state("What is LangGraph?"))

        assert result["final_response"] != ""
        assert len(result["agent_metadata"]["key_findings"]) > 0

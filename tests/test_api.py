"""
API Tests — Integration tests for FastAPI endpoints.
Run: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "vector_store_stats" in data
        assert "crm_stats" in data


class TestChatEndpoint:
    def test_chat_requires_user_id_and_message(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422  # Validation error

    def test_chat_rejects_empty_message(self, client):
        mock_result = {
            "response": "Invalid message: Message too short",
            "session_id": "sess_001",
            "agent": "unknown",
            "intent": "unknown",
            "confidence": 0.0,
            "rag_used": False,
            "requires_human": False,
            "crm_actions": [],
            "latency_ms": 10.0,
            "trace_id": "trace_001",
            "metadata": {},
        }
        with patch("api.routes.chat.process_message", return_value=mock_result):
            response = client.post("/chat", json={"user_id": "u1", "message": "x"})
        assert response.status_code == 200

    def test_chat_returns_correct_schema(self, client):
        mock_result = {
            "response": "Hello! How can I help?",
            "session_id": "sess_abc",
            "agent": "support",
            "intent": "support",
            "confidence": 0.9,
            "rag_used": True,
            "requires_human": False,
            "crm_actions": [],
            "latency_ms": 250.0,
            "trace_id": "trace_abc",
            "metadata": {"confidence": 0.9},
        }
        with patch("api.routes.chat.process_message", return_value=mock_result):
            response = client.post("/chat", json={
                "user_id": "user_123",
                "message": "I need help with my account",
            })

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "agent" in data
        assert "confidence" in data
        assert "trace_id" in data

    def test_chat_handles_server_error(self, client):
        with patch("api.routes.chat.process_message", side_effect=Exception("LLM error")):
            response = client.post("/chat", json={
                "user_id": "user_err",
                "message": "test message",
            })
        assert response.status_code == 500


class TestKnowledgeEndpoints:
    def test_add_document(self, client):
        response = client.post("/knowledge/add", json={
            "title": "Test Article",
            "content": "This is a test article about testing.",
            "category": "support",
            "tags": ["test"],
        })
        assert response.status_code == 200
        assert "total_docs" in response.json()

    def test_search_knowledge(self, client):
        response = client.get("/knowledge/search?query=test&k=3")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "count" in data

    def test_init_knowledge_base(self, client):
        response = client.get("/knowledge/init")
        assert response.status_code == 200
        assert "documents_loaded" in response.json()


class TestCRMEndpoints:
    def test_crm_stats(self, client):
        response = client.get("/crm/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_leads" in data

    def test_crm_leads(self, client):
        response = client.get("/crm/leads")
        assert response.status_code == 200
        assert "leads" in response.json()

"""
Shared test configuration.

The data directory is redirected to a temporary path *before* any project
module is imported, because several modules resolve their file paths at import
time. Without this the suite wrote into the developer's real `memory/data`,
polluting the CRM, knowledge base and metrics of a working install.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Must run before the first project import.
_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="maas-tests-"))
os.environ["DATA_DIR"] = str(_TEST_DATA_DIR)
os.environ["APP_ENV"] = "test"
os.environ.setdefault("DEMO_MODE", "true")
os.environ.pop("ADMIN_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from core import config  # noqa: E402
from core.security import reset_rate_limits  # noqa: E402
from memory.vector_store import reset_vector_store  # noqa: E402
from monitoring.metrics import reset_metrics  # noqa: E402
from tools.crm import reset_crm  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Give every test a clean data directory and empty shared singletons."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "ADMIN_API_KEY", None)

    _redirect_module_paths(monkeypatch, tmp_path)

    reset_vector_store()
    reset_crm()
    reset_metrics()
    reset_rate_limits()

    yield

    reset_vector_store()
    reset_crm()
    reset_metrics()
    reset_rate_limits()


def _redirect_module_paths(monkeypatch, tmp_path: Path) -> None:
    """Point every module-level data path at the per-test directory."""
    import core.review_queue as review_queue
    import evaluation.feedback as feedback
    import memory.conversation as conversation
    import memory.knowledge_base as knowledge_base
    import memory.vector_store as vector_store
    import monitoring.metrics as metrics
    import tools.crm as crm
    import tools.email as email
    import tools.followup as followup

    monkeypatch.setattr(vector_store, "VECTOR_STORE_PATH", tmp_path / "faiss_index.bin")
    monkeypatch.setattr(vector_store, "DOCUMENTS_PATH", tmp_path / "documents.json")
    monkeypatch.setattr(knowledge_base, "KB_METADATA_PATH", tmp_path / "kb_metadata.json")
    monkeypatch.setattr(conversation, "CONV_DIR", tmp_path / "conversations")
    monkeypatch.setattr(crm, "CRM_DATA_PATH", tmp_path / "crm_data.json")
    monkeypatch.setattr(email, "EMAIL_LOG_PATH", tmp_path / "email_log.json")
    monkeypatch.setattr(followup, "FOLLOWUP_PATH", tmp_path / "followups.json")
    monkeypatch.setattr(review_queue, "PENDING_PATH", tmp_path / "pending_reviews.json")
    monkeypatch.setattr(metrics, "METRICS_PATH", tmp_path / "metrics.jsonl")
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")


@pytest.fixture
def client():
    """TestClient with the application lifespan active."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def raw_client():
    """TestClient that surfaces server errors as responses, not exceptions."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def make_state(message: str, history: list = None, **overrides) -> dict:
    """Build a complete AgentState for direct agent-node tests."""
    state = {
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
    state.update(overrides)
    return state

"""
RAG Tests — Tests for FAISS vector store and knowledge base.
Run: pytest tests/test_rag.py -v
"""

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Create a FAISSVectorStore with a temp directory."""
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path))
    # Patch paths in vector_store module
    import memory.vector_store as vs
    monkeypatch.setattr(vs, "VECTOR_STORE_PATH", tmp_path / "faiss_index.pkl")
    monkeypatch.setattr(vs, "DOCUMENTS_PATH", tmp_path / "documents.json")

    from memory.vector_store import FAISSVectorStore
    return FAISSVectorStore()


class TestFAISSVectorStore:
    def test_empty_store_returns_no_results(self, fresh_store):
        results = fresh_store.search("test query")
        assert results == []

    def test_add_and_search_document(self, fresh_store):
        fresh_store.add_documents([{
            "title": "API Guide",
            "content": "How to use the API with authentication and rate limits",
            "metadata": {"category": "support"},
        }])
        results = fresh_store.search("API authentication", k=1)
        assert len(results) >= 1
        assert results[0]["title"] == "API Guide"

    def test_search_returns_relevance_score(self, fresh_store):
        fresh_store.add_documents([{
            "title": "Pricing",
            "content": "monthly pricing plans starter professional enterprise",
            "metadata": {},
        }])
        results = fresh_store.search("pricing plans")
        assert all("relevance_score" in r for r in results)
        assert all(0 <= r["relevance_score"] <= 1.0 for r in results)

    def test_k_limits_results(self, fresh_store):
        docs = [
            {"title": f"Doc {i}", "content": f"content about topic {i} keywords test", "metadata": {}}
            for i in range(10)
        ]
        fresh_store.add_documents(docs)
        results = fresh_store.search("keywords test", k=3)
        assert len(results) <= 3

    def test_get_stats(self, fresh_store):
        fresh_store.add_documents([
            {"title": "A", "content": "alpha beta gamma", "metadata": {"category": "sales"}},
            {"title": "B", "content": "delta epsilon zeta", "metadata": {"category": "support"}},
        ])
        stats = fresh_store.get_stats()
        assert stats["total_documents"] == 2
        assert stats["index_size"] == 2
        assert "sales" in stats["categories"]

    def test_embedding_is_normalized(self, fresh_store):
        vec = fresh_store._get_embedding("hello world test")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5 or norm == 0.0

    def test_load_sample_knowledge_base(self, fresh_store):
        count = fresh_store.load_sample_knowledge_base()
        assert count > 0
        assert fresh_store.index.ntotal == count
        assert len(fresh_store.documents) == count


class TestKnowledgeBase:
    def test_ingest_and_retrieve(self, tmp_path, monkeypatch):
        import memory.vector_store as vs
        monkeypatch.setattr(vs, "VECTOR_STORE_PATH", tmp_path / "faiss.pkl")
        monkeypatch.setattr(vs, "DOCUMENTS_PATH", tmp_path / "docs.json")

        import memory.knowledge_base as kb_module
        monkeypatch.setattr(kb_module, "KB_METADATA_PATH", tmp_path / "kb_meta.json")

        from memory.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()

        added = kb.ingest([
            {"title": "Security Policy", "content": "AES-256 encryption TLS GDPR compliance data privacy"},
        ], source="test")

        assert added == 1
        results = kb.retrieve("encryption security")
        assert len(results) >= 1

    def test_ingest_skips_invalid_docs(self, tmp_path, monkeypatch):
        import memory.vector_store as vs
        monkeypatch.setattr(vs, "VECTOR_STORE_PATH", tmp_path / "faiss.pkl")
        monkeypatch.setattr(vs, "DOCUMENTS_PATH", tmp_path / "docs.json")

        import memory.knowledge_base as kb_module
        monkeypatch.setattr(kb_module, "KB_METADATA_PATH", tmp_path / "kb_meta.json")

        from memory.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()

        added = kb.ingest([
            {"title": "", "content": "no title"},
            {"title": "valid", "content": ""},
            {},
        ])
        assert added == 0


class TestConversationManager:
    def test_add_and_retrieve_turns(self, tmp_path, monkeypatch):
        import memory.conversation as cm_module
        monkeypatch.setattr(cm_module, "CONV_DIR", tmp_path / "convs")

        from memory.conversation import ConversationManager
        cm = ConversationManager("test_sess")
        cm.add_turn("user", "Hello")
        cm.add_turn("assistant", "Hi there!")

        context = cm.get_context_window(n=4)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"

    def test_context_window_truncates(self, tmp_path, monkeypatch):
        import memory.conversation as cm_module
        monkeypatch.setattr(cm_module, "CONV_DIR", tmp_path / "convs")

        from memory.conversation import ConversationManager
        cm = ConversationManager("test_trunc")
        for i in range(20):
            cm.add_turn("user", f"message {i}")

        context = cm.get_context_window(n=6)
        assert len(context) <= 6

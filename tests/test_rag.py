"""
RAG Tests — Tests for FAISS vector store and knowledge base.
Run: pytest tests/test_rag.py -v
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from memory.vector_store import FAISSVectorStore, get_vector_store


@pytest.fixture
def store():
    """A fresh store rooted in the per-test data directory."""
    return FAISSVectorStore()


class TestFAISSVectorStore:
    def test_empty_store_returns_no_results(self, store):
        assert store.search("test query") == []

    def test_add_and_search_document(self, store):
        store.add_documents([{
            "title": "API Guide",
            "content": "How to use the API with authentication and rate limits",
            "metadata": {"category": "support"},
        }])
        results = store.search("API authentication", k=1)
        assert results and results[0]["title"] == "API Guide"

    def test_search_returns_relevance_score(self, store):
        store.add_documents([{
            "title": "Pricing",
            "content": "monthly pricing plans starter professional enterprise",
            "metadata": {},
        }])
        results = store.search("pricing plans")
        assert all(0 <= r["relevance_score"] <= 1.0 for r in results)

    def test_k_limits_results(self, store):
        store.add_documents([
            {"title": f"Doc {i}", "content": f"content about topic {i} keywords test", "metadata": {}}
            for i in range(10)
        ])
        assert len(store.search("keywords test", k=3)) <= 3

    def test_get_stats(self, store):
        store.add_documents([
            {"title": "A", "content": "alpha beta gamma", "metadata": {"category": "sales"}},
            {"title": "B", "content": "delta epsilon zeta", "metadata": {"category": "support"}},
        ])
        stats = store.get_stats()
        assert stats["total_documents"] == 2
        assert stats["index_size"] == 2
        assert "sales" in stats["categories"]

    def test_embedding_is_normalized(self, store):
        norm = np.linalg.norm(store._get_embedding("hello world test"))
        assert abs(norm - 1.0) < 1e-5 or norm == 0.0

    def test_load_sample_knowledge_base(self, store):
        count = store.load_sample_knowledge_base()
        assert count > 0
        assert store.index.ntotal == count == len(store.documents)

    def test_add_documents_returns_count_and_ignores_empties(self, store):
        assert store.add_documents([]) == 0
        assert store.add_documents([{}, {"title": "", "content": ""}]) == 0
        assert store.add_documents([{"title": "T", "content": "body text"}]) == 1

    def test_index_survives_a_reload(self, store):
        """Documents and index must stay in sync across instances."""
        store.load_sample_knowledge_base()
        reloaded = FAISSVectorStore()

        assert reloaded.index.ntotal == store.index.ntotal
        assert [d["title"] for d in reloaded.documents] == [d["title"] for d in store.documents]
        assert reloaded.search("pricing plans", k=1)[0]["title"] == "Product Pricing Plans"

    def test_index_rebuilds_when_out_of_sync(self, store, monkeypatch):
        import memory.vector_store as vs

        store.load_sample_knowledge_base()
        # Simulate a half-written pair: documents present, index file gone.
        vs.VECTOR_STORE_PATH.unlink()
        reloaded = FAISSVectorStore()

        assert reloaded.index.ntotal == len(reloaded.documents) > 0

    def test_corrupt_documents_file_does_not_crash(self, store):
        import memory.vector_store as vs

        vs.DOCUMENTS_PATH.write_text("{not json")
        rebuilt = FAISSVectorStore()
        assert rebuilt.documents == []


class TestEmbeddingDeterminism:
    def test_bucket_is_stable_within_the_process(self):
        from memory.vector_store import _stable_bucket

        assert _stable_bucket("pricing", 512) == _stable_bucket("pricing", 512)

    def test_bucket_is_stable_across_processes(self):
        """
        Regression: the embedding used Python's built-in `hash()`, which is
        salted per interpreter. A persisted index scored queries from a later
        process against entirely different buckets, so RAG silently returned
        nothing useful after every restart.
        """
        script = textwrap.dedent(
            """
            from memory.vector_store import _stable_bucket
            print(_stable_bucket("pricing", 512))
            """
        )
        seen = set()
        for seed in ("0", "1", "12345"):
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, check=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONPATH": "."},
            )
            seen.add(out.stdout.strip())

        assert len(seen) == 1, f"bucket varied across hash seeds: {seen}"


class TestSharedStore:
    def test_get_vector_store_returns_one_instance(self):
        """
        Regression: five modules each built their own store, so writes through
        one were invisible to the others.
        """
        assert get_vector_store() is get_vector_store()

    def test_writes_are_visible_to_every_holder(self):
        first = get_vector_store()
        first.add_documents([{"title": "Shared Doc", "content": "shared marker content"}])

        from tools.search import search_knowledge_base
        assert any(r["title"] == "Shared Doc" for r in search_knowledge_base("shared marker"))


class TestKnowledgeBase:
    def test_ingest_and_retrieve(self):
        from memory.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        added = kb.ingest([
            {"title": "Security Policy", "content": "AES-256 encryption TLS GDPR compliance data privacy"},
        ], source="test")

        assert added == 1
        assert kb.retrieve("encryption security")

    def test_ingest_skips_invalid_docs(self):
        from memory.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        assert kb.ingest([
            {"title": "", "content": "no title"},
            {"title": "valid", "content": ""},
            {},
        ]) == 0

    def test_category_filter_still_returns_results(self):
        """Filtering after a k-sized fetch used to empty the result set."""
        from memory.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        kb.ingest([
            {"title": f"Sales {i}", "content": "pricing plan enterprise discount",
             "metadata": {"category": "sales"}}
            for i in range(3)
        ] + [
            {"title": f"Support {i}", "content": "pricing plan enterprise discount",
             "metadata": {"category": "support"}}
            for i in range(6)
        ])

        results = kb.retrieve("pricing plan enterprise", k=2, category="sales")
        assert results
        assert all(r["metadata"]["category"] == "sales" for r in results)

    def test_rejects_non_list_input(self):
        from memory.knowledge_base import KnowledgeBase

        with pytest.raises(ValueError):
            KnowledgeBase().ingest({"title": "x", "content": "y"})


class TestConversationManager:
    def test_add_and_retrieve_turns(self):
        from memory.conversation import ConversationManager

        cm = ConversationManager("test_sess")
        cm.add_turn("user", "Hello")
        cm.add_turn("assistant", "Hi there!")

        context = cm.get_context_window(n=4)
        assert [t["role"] for t in context] == ["user", "assistant"]

    def test_context_window_truncates(self):
        from memory.conversation import ConversationManager

        cm = ConversationManager("test_trunc")
        for i in range(20):
            cm.add_turn("user", f"message {i}")

        assert len(cm.get_context_window(n=6)) <= 6

    def test_history_persists_across_instances(self):
        from memory.conversation import ConversationManager

        ConversationManager("persist_sess").add_turn("user", "remember me")
        assert ConversationManager("persist_sess").history[0]["content"] == "remember me"

    def test_session_id_cannot_escape_the_directory(self):
        """A path-traversal session id must not write outside conversations/."""
        from memory.conversation import CONV_DIR, ConversationManager

        cm = ConversationManager("../../etc/passwd")
        cm.add_turn("user", "hi")

        assert cm._path.parent.resolve() == CONV_DIR.resolve()
        assert cm._path.exists()

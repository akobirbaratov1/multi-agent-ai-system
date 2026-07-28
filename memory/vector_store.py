"""
FAISS Vector Store — Document Indexing & Retrieval
Production-ready RAG implementation using FAISS IndexFlatIP
"""

import hashlib
import threading
from typing import Dict, List, Optional

import faiss
import numpy as np

from core import config
from core.logging_config import get_logger
from core.storage import read_json, write_bytes, write_json

logger = get_logger(__name__)

VECTOR_STORE_PATH = config.data_path("faiss_index.bin")
DOCUMENTS_PATH = config.data_path("documents.json")
EMBEDDING_DIM = 512


def _stable_bucket(word: str, dim: int) -> int:
    """
    Map a word to a bucket deterministically across processes.

    Python's built-in `hash()` is salted per interpreter (PYTHONHASHSEED), so an
    index built in one process scores queries from another process against
    completely different buckets — a persisted index silently returned garbage
    after every restart. blake2b is stable everywhere.
    """
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


class FAISSVectorStore:
    """FAISS-based vector store for document retrieval."""

    def __init__(self):
        self.documents: List[Dict] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        # Guards index/documents against concurrent add + search from request handlers.
        self._lock = threading.RLock()
        self._load_or_init()

    # ── persistence ──────────────────────────────────────

    def _load_or_init(self) -> None:
        VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

        self.documents = read_json(DOCUMENTS_PATH, default=[])
        if not isinstance(self.documents, list):
            logger.warning("Documents file was malformed; starting with an empty set")
            self.documents = []

        self.index = self._read_index()

        # A mismatch means one of the two files was written by an older build or
        # lost mid-write. Rebuilding from documents.json is cheap and always
        # yields a consistent pair.
        if self.index.ntotal != len(self.documents):
            if self.documents:
                logger.warning(
                    "Vector index out of sync with documents; rebuilding",
                    extra={"index_size": self.index.ntotal, "documents": len(self.documents)},
                )
                self._rebuild_index()
            else:
                self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

    def _read_index(self) -> faiss.IndexFlatIP:
        if not VECTOR_STORE_PATH.exists():
            return faiss.IndexFlatIP(EMBEDDING_DIM)
        try:
            # faiss' own serialization — the previous build pickled the index,
            # which executes arbitrary code from whatever is on disk.
            data = np.fromfile(str(VECTOR_STORE_PATH), dtype=np.uint8)
            index = faiss.deserialize_index(data)
            if index.d != EMBEDDING_DIM:
                logger.warning("Persisted index has a different dimension; rebuilding")
                return faiss.IndexFlatIP(EMBEDDING_DIM)
            return index
        except Exception:
            logger.warning("Could not read persisted index; starting fresh", exc_info=True)
            return faiss.IndexFlatIP(EMBEDDING_DIM)

    def _rebuild_index(self) -> None:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if self.documents:
            matrix = np.array(
                [self._get_embedding(self._doc_text(doc)) for doc in self.documents],
                dtype=np.float32,
            )
            index.add(matrix)
        self.index = index
        self._save()

    def _save(self) -> None:
        write_json(DOCUMENTS_PATH, self.documents)
        write_bytes(VECTOR_STORE_PATH, faiss.serialize_index(self.index).tobytes())

    # ── embedding ────────────────────────────────────────

    @staticmethod
    def _doc_text(doc: Dict) -> str:
        return f"{doc.get('title', '')} {doc.get('content', '')}"

    def _get_embedding(self, text: str) -> np.ndarray:
        """Hash-based bag-of-words embedding, normalized for cosine similarity."""
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for word in text.lower().split():
            vector[_stable_bucket(word, EMBEDDING_DIM)] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    # ── public API ───────────────────────────────────────

    def add_documents(self, documents: List[Dict]) -> int:
        """Add documents to the FAISS index. Returns the number added."""
        valid = [d for d in documents if d.get("title") or d.get("content")]
        if not valid:
            return 0

        with self._lock:
            matrix = np.array(
                [self._get_embedding(self._doc_text(doc)) for doc in valid],
                dtype=np.float32,
            )
            self.index.add(matrix)
            self.documents.extend(valid)
            self._save()

        return len(valid)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents using FAISS."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            query_vec = self._get_embedding(query).reshape(1, -1)
            k = max(1, min(k, self.index.ntotal))
            scores, indices = self.index.search(query_vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx < 0 or idx >= len(self.documents) or score <= 0.1:
                    continue
                doc = dict(self.documents[idx])
                doc["relevance_score"] = float(score)
                results.append(doc)

        return results

    def load_sample_knowledge_base(self) -> int:
        """Load sample knowledge base for demo."""
        sample_docs = [
            {
                "title": "Product Pricing Plans",
                "content": "We offer three pricing plans: Starter ($49/mo) for small teams, "
                           "Professional ($149/mo) for growing businesses, and Enterprise (custom) "
                           "for large organizations. All plans include 14-day free trial.",
                "metadata": {"category": "sales", "tags": ["pricing", "plans"]},
            },
            {
                "title": "API Integration Guide",
                "content": "To integrate our API: 1) Get your API key from dashboard, "
                           "2) Install SDK: pip install our-sdk, 3) Initialize client with key, "
                           "4) Call endpoints. Rate limits: 100 req/min on Starter, 1000 on Pro.",
                "metadata": {"category": "support", "tags": ["api", "integration"]},
            },
            {
                "title": "Troubleshooting Connection Issues",
                "content": "Common connection issues: 1) Check API key validity, "
                           "2) Verify network connectivity, 3) Check service status at status.ourapp.com, "
                           "4) Clear cache and retry. If persists, contact support@ourapp.com",
                "metadata": {"category": "support", "tags": ["troubleshooting", "connection"]},
            },
            {
                "title": "Multi-Agent AI Overview",
                "content": "Our Multi-Agent AI system uses LangGraph for orchestration, "
                           "with specialized agents for sales, support, and research. "
                           "The system includes RAG for knowledge retrieval, confidence scoring "
                           "for quality control, and human-in-the-loop for edge cases.",
                "metadata": {"category": "research", "tags": ["multi-agent", "architecture"]},
            },
            {
                "title": "Data Security & Privacy",
                "content": "We are SOC2 Type II certified. Data is encrypted at rest (AES-256) "
                           "and in transit (TLS 1.3). GDPR compliant. Data retention: 90 days default, "
                           "configurable. PII is automatically detected and masked in logs.",
                "metadata": {"category": "research", "tags": ["security", "privacy", "compliance"]},
            },
            {
                "title": "Getting Started Guide",
                "content": "Welcome! To get started: 1) Sign up at app.ourplatform.com, "
                           "2) Complete onboarding (5 min), 3) Connect your data sources, "
                           "4) Deploy your first AI agent. Support is available 24/7 via chat.",
                "metadata": {"category": "support", "tags": ["onboarding", "getting-started"]},
            },
            {
                "title": "Enterprise Features",
                "content": "Enterprise plan includes: SSO/SAML, custom SLA (99.99% uptime), "
                           "dedicated support manager, on-premise deployment option, "
                           "unlimited API calls, custom model fine-tuning, and audit logs.",
                "metadata": {"category": "sales", "tags": ["enterprise", "features"]},
            },
            {
                "title": "Billing & Payments",
                "content": "We accept all major credit cards and ACH transfers. "
                           "Invoicing available for Enterprise. Cancel anytime — no lock-in. "
                           "Annual billing saves 20%. Refunds available within 30 days.",
                "metadata": {"category": "support", "tags": ["billing", "payment"]},
            },
        ]

        return self.add_documents(sample_docs)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_documents": len(self.documents),
                "index_size": self.index.ntotal if self.index else 0,
                "embedding_dim": EMBEDDING_DIM,
                "categories": sorted({
                    doc.get("metadata", {}).get("category", "unknown")
                    for doc in self.documents
                }),
            }


# ── shared instance ──────────────────────────────────────
#
# Every module used to construct its own FAISSVectorStore. Each one loaded a
# private copy of the index at import time, so a document added through the
# knowledge API was invisible to the support and research agents until the
# process restarted. One process-wide instance keeps every reader consistent.

_store: Optional[FAISSVectorStore] = None
_store_guard = threading.Lock()


def get_vector_store() -> FAISSVectorStore:
    """Return the process-wide vector store, constructing it on first use."""
    global _store
    if _store is None:
        with _store_guard:
            if _store is None:
                _store = FAISSVectorStore()
    return _store


def reset_vector_store() -> None:
    """Drop the shared instance. Used by tests that redirect the data paths."""
    global _store
    with _store_guard:
        _store = None

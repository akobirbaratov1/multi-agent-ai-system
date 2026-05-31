"""
FAISS Vector Store — Document Indexing & Retrieval
Production-ready RAG implementation using FAISS IndexFlatIP
"""

import os
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional

VECTOR_STORE_PATH = Path("memory/data/faiss_index.pkl")
DOCUMENTS_PATH = Path("memory/data/documents.json")
EMBEDDING_DIM = 512


class FAISSVectorStore:
    """FAISS-based vector store for document retrieval."""

    def __init__(self):
        self.documents: List[Dict] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._load_or_init()

    def _load_or_init(self):
        os.makedirs("memory/data", exist_ok=True)

        if DOCUMENTS_PATH.exists():
            with open(DOCUMENTS_PATH, "r") as f:
                self.documents = json.load(f)

        if VECTOR_STORE_PATH.exists():
            with open(VECTOR_STORE_PATH, "rb") as f:
                self.index = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Hash-based bag-of-words embedding, normalized for cosine similarity."""
        words = text.lower().split()
        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for word in words:
            idx = hash(word) % EMBEDDING_DIM
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the FAISS index."""
        embeddings = []
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            embeddings.append(self._get_embedding(text))
            self.documents.append(doc)

        matrix = np.array(embeddings, dtype=np.float32)
        self.index.add(matrix)
        self._save()

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents using FAISS."""
        if self.index.ntotal == 0:
            return []

        query_vec = self._get_embedding(query).reshape(1, -1)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.1:
                doc = self.documents[idx].copy()
                doc["relevance_score"] = float(score)
                results.append(doc)

        return results

    def _save(self):
        with open(DOCUMENTS_PATH, "w") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(self.index, f)

    def load_sample_knowledge_base(self):
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

        self.add_documents(sample_docs)
        return len(sample_docs)

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dim": EMBEDDING_DIM,
            "categories": list(set(
                doc.get("metadata", {}).get("category", "unknown")
                for doc in self.documents
            )),
        }

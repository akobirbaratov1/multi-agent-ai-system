"""
Knowledge Base — Document ingestion, management, and retrieval.
Wraps FAISSVectorStore with higher-level document management.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from core import config
from core.storage import read_json, write_json
from memory.vector_store import get_vector_store

KB_METADATA_PATH = config.data_path("kb_metadata.json")

MAX_INGEST_BATCH = 5_000


class KnowledgeBase:
    """High-level knowledge base manager on top of FAISSVectorStore."""

    def __init__(self):
        # Shares the process-wide store so ingestion is visible to the agents
        # immediately, rather than only after a restart.
        self.store = get_vector_store()
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        metadata = read_json(KB_METADATA_PATH, default=None)
        if not isinstance(metadata, dict):
            return {"total_ingested": 0, "sources": []}
        metadata.setdefault("total_ingested", 0)
        metadata.setdefault("sources", [])
        return metadata

    def _save_metadata(self):
        write_json(KB_METADATA_PATH, self._metadata)

    def ingest(self, documents: List[Dict], source: str = "manual") -> int:
        """
        Ingest documents into the knowledge base.

        Args:
            documents: List of dicts with 'title', 'content', optional 'metadata'
            source: Source label for tracking

        Returns:
            Number of documents added
        """
        if not isinstance(documents, list):
            raise ValueError("documents must be a list")

        valid_docs = [
            d for d in documents
            if isinstance(d, dict) and d.get("title") and d.get("content")
        ]
        if not valid_docs:
            return 0

        if len(valid_docs) > MAX_INGEST_BATCH:
            raise ValueError(
                f"Batch too large: {len(valid_docs)} documents "
                f"(limit {MAX_INGEST_BATCH})"
            )

        added = self.store.add_documents(valid_docs)

        self._metadata["total_ingested"] += added
        if source not in self._metadata["sources"]:
            self._metadata["sources"].append(source)
        self._save_metadata()

        return added

    def retrieve(self, query: str, k: int = 3, category: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        # Filtering after retrieval can empty a full result set, so when a
        # category is requested we over-fetch and then narrow to k.
        results = self.store.search(query=query, k=k * 4 if category else k)
        if category:
            results = [
                r for r in results
                if r.get("metadata", {}).get("category") == category
            ][:k]
        return results

    def ingest_from_file(self, filepath: str) -> int:
        """Load documents from a JSON file and ingest them."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path, encoding="utf-8") as f:
            documents = json.load(f)

        if not isinstance(documents, list):
            raise ValueError(f"{filepath} must contain a JSON array of documents")

        return self.ingest(documents, source=path.name)

    def get_stats(self) -> Dict:
        return {
            **self.store.get_stats(),
            "total_ingested": self._metadata["total_ingested"],
            "sources": self._metadata["sources"],
        }

    def initialize_sample_data(self) -> int:
        """Load sample knowledge base (idempotent)."""
        if self.store.index.ntotal > 0:
            return 0
        return self.store.load_sample_knowledge_base()

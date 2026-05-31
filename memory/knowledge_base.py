"""
Knowledge Base — Document ingestion, management, and retrieval.
Wraps FAISSVectorStore with higher-level document management.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from memory.vector_store import FAISSVectorStore

KB_METADATA_PATH = Path("memory/data/kb_metadata.json")


class KnowledgeBase:
    """High-level knowledge base manager on top of FAISSVectorStore."""

    def __init__(self):
        self.store = FAISSVectorStore()
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        if KB_METADATA_PATH.exists():
            with open(KB_METADATA_PATH) as f:
                return json.load(f)
        return {"total_ingested": 0, "sources": []}

    def _save_metadata(self):
        import os
        os.makedirs(KB_METADATA_PATH.parent, exist_ok=True)
        with open(KB_METADATA_PATH, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def ingest(self, documents: List[Dict], source: str = "manual") -> int:
        """
        Ingest documents into the knowledge base.

        Args:
            documents: List of dicts with 'title', 'content', optional 'metadata'
            source: Source label for tracking

        Returns:
            Number of documents added
        """
        valid_docs = [d for d in documents if d.get("title") and d.get("content")]
        if not valid_docs:
            return 0

        self.store.add_documents(valid_docs)

        self._metadata["total_ingested"] += len(valid_docs)
        if source not in self._metadata["sources"]:
            self._metadata["sources"].append(source)
        self._save_metadata()

        return len(valid_docs)

    def retrieve(self, query: str, k: int = 3, category: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        results = self.store.search(query=query, k=k)
        if category:
            results = [
                r for r in results
                if r.get("metadata", {}).get("category") == category
            ]
        return results

    def ingest_from_file(self, filepath: str) -> int:
        """Load documents from a JSON file and ingest them."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path) as f:
            documents = json.load(f)

        return self.ingest(documents, source=path.name)

    def get_stats(self) -> Dict:
        store_stats = self.store.get_stats()
        return {
            **store_stats,
            "total_ingested": self._metadata["total_ingested"],
            "sources": self._metadata["sources"],
        }

    def initialize_sample_data(self) -> int:
        """Load sample knowledge base (idempotent)."""
        if self.store.index.ntotal > 0:
            return 0
        return self.store.load_sample_knowledge_base()

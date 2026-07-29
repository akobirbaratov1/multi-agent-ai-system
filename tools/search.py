"""
Search Tools — Knowledge base & web search utilities
Provides unified search interface for agents.
"""

from typing import Dict, List, Optional

from memory.vector_store import FAISSVectorStore, get_vector_store


def _get_store() -> FAISSVectorStore:
    """Use the process-wide store so newly ingested documents are visible."""
    return get_vector_store()


def search_knowledge_base(query: str, k: int = 5, category: Optional[str] = None) -> List[Dict]:
    """
    Search the FAISS knowledge base.

    Args:
        query: Search query string
        k: Max number of results
        category: Optional filter by document category

    Returns:
        List of relevant documents with relevance_score
    """
    results = _get_store().search(query=query, k=k)

    if category:
        results = [
            r for r in results
            if r.get("metadata", {}).get("category") == category
        ]

    return results


def search_by_tags(tags: List[str], k: int = 5) -> List[Dict]:
    """Search documents that have matching tags."""
    store = _get_store()
    matches = []
    for doc in store.documents:
        doc_tags = doc.get("metadata", {}).get("tags", [])
        if any(tag in doc_tags for tag in tags):
            matches.append(doc)
        if len(matches) >= k:
            break
    return matches


def format_context(documents: List[Dict], max_chars: int = 3000) -> str:
    """Format retrieved documents into a context string for LLM prompts."""
    parts = []
    total = 0
    for doc in documents:
        snippet = f"[Source: {doc.get('title', 'Unknown')}]\n{doc.get('content', '')}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts) if parts else "No relevant sources found."

"""Knowledge base management routes."""

from fastapi import APIRouter, HTTPException
from api.schemas import DocumentRequest, DocumentResponse, SearchResponse
from memory.vector_store import FAISSVectorStore

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])

_store = FAISSVectorStore()


@router.post("/add", response_model=DocumentResponse)
async def add_document(request: DocumentRequest):
    """Add a document to the knowledge base."""
    _store.add_documents([{
        "title": request.title,
        "content": request.content,
        "metadata": {"category": request.category, "tags": request.tags},
    }])
    return {"status": "added", "total_docs": len(_store.documents)}


@router.get("/search", response_model=SearchResponse)
async def search_knowledge(query: str, k: int = 3):
    """Search the knowledge base for relevant documents."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = _store.search(query=query, k=min(k, 10))
    return {"results": results, "count": len(results)}


@router.get("/init")
async def init_knowledge_base():
    """Load the sample knowledge base (idempotent)."""
    if _store.index.ntotal > 0:
        return {"status": "already_initialized", "documents_loaded": 0}
    count = _store.load_sample_knowledge_base()
    return {"status": "initialized", "documents_loaded": count}


@router.get("/stats")
async def knowledge_stats():
    """Knowledge base statistics."""
    return _store.get_stats()

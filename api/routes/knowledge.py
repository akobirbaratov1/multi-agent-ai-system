"""Knowledge base management routes."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import require_admin
from api.schemas import DocumentRequest, DocumentResponse, SearchResponse
from memory.vector_store import get_vector_store

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


@router.post(
    "/add",
    response_model=DocumentResponse,
    dependencies=[Depends(require_admin)],
)
async def add_document(request: DocumentRequest):
    """
    Add a document to the knowledge base.

    Admin-only: knowledge base content is fed to the agents as grounding, so
    write access here is effectively write access to what the agents say.
    """
    store = get_vector_store()
    added = store.add_documents([{
        "title": request.title,
        "content": request.content,
        "metadata": {"category": request.category, "tags": request.tags},
    }])

    if not added:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document could not be indexed",
        )

    return {"status": "added", "total_docs": len(store.documents)}


@router.get("/search", response_model=SearchResponse)
async def search_knowledge(
    query: str = Query(..., min_length=1, max_length=1000),
    k: int = Query(3, ge=1, le=10),
):
    """Search the knowledge base for relevant documents."""
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty"
        )
    results = get_vector_store().search(query=query, k=k)
    return {"results": results, "count": len(results)}


@router.post("/init", dependencies=[Depends(require_admin)])
async def init_knowledge_base():
    """Load the sample knowledge base (idempotent)."""
    store = get_vector_store()
    if store.index.ntotal > 0:
        return {"status": "already_initialized", "documents_loaded": 0}
    count = store.load_sample_knowledge_base()
    return {"status": "initialized", "documents_loaded": count}


@router.get("/stats")
async def knowledge_stats():
    """Knowledge base statistics."""
    return get_vector_store().get_stats()

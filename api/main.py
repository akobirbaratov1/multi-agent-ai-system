"""
FastAPI Application — REST API for Multi-Agent AI System
Full OpenAPI documentation at /docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from api.routes import chat_router, knowledge_router, crm_router, monitoring_router, operator_router, webhooks_router
from api.schemas import HealthResponse
from memory.vector_store import FAISSVectorStore
from tools.crm import MockCRM

app = FastAPI(
    title="Multi-Agent AI System",
    description="Production-ready Multi-Agent AI with LangGraph, Claude & FAISS RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="web"), name="static")

app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(crm_router)
app.include_router(monitoring_router)
app.include_router(operator_router)
app.include_router(webhooks_router)

_vector_store = FAISSVectorStore()
_crm = MockCRM()


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("web/index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check with stats."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "vector_store_stats": _vector_store.get_stats(),
        "crm_stats": _crm.get_stats(),
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

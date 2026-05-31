"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    interface: str = Field("web", description="Interface type: web, api, telegram")
    conversation_history: Optional[List[Dict]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent: str
    intent: str
    confidence: float
    rag_used: bool
    requires_human: bool
    crm_actions: List[str]
    latency_ms: float
    trace_id: str
    metadata: Dict[str, Any]


class DocumentRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    category: str = Field("general", description="Document category: sales, support, research, general")
    tags: List[str] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    status: str
    total_docs: int


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_stats: Dict[str, Any]
    crm_stats: Dict[str, Any]


class FeedbackRequest(BaseModel):
    trace_id: str
    session_id: str
    rating: int = Field(..., ge=-1, le=1, description="1 = positive, -1 = negative")
    comment: Optional[str] = None
    agent: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    status: str
    sentiment: str


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    rag_usage_rate: float
    requests_by_agent: Dict[str, int]
    requests_by_intent: Dict[str, int]

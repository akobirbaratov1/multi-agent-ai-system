"""
Pydantic schemas for API request/response validation.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from core import config


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=config.MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, max_length=128, description="Unique user identifier"
    )
    message: str = Field(
        ...,
        min_length=1,
        # Bounded at the edge so an oversized body is rejected before it reaches
        # the graph — and, more importantly, before it is billed to the model.
        max_length=config.MAX_MESSAGE_LENGTH,
        description="User message",
    )
    session_id: Optional[str] = Field(
        None, max_length=128, description="Session ID for conversation continuity"
    )
    interface: Literal["web", "api", "telegram", "2chat", "test"] = Field(
        "web", description="Interface type"
    )
    conversation_history: List[ChatMessage] = Field(
        default_factory=list, max_length=50
    )

    @field_validator("user_id", "session_id")
    @classmethod
    def _no_control_chars(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned or any(ord(ch) < 32 for ch in cleaned):
            raise ValueError("must not be blank or contain control characters")
        return cleaned


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
    error: Optional[str] = None


class DocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1, max_length=50_000)
    category: Literal["sales", "support", "research", "general"] = Field(
        "general", description="Document category"
    )
    tags: List[str] = Field(default_factory=list, max_length=25)


class DocumentResponse(BaseModel):
    status: str
    total_docs: int


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    demo_mode: bool
    vector_store_stats: Dict[str, Any]
    crm_stats: Dict[str, Any]


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    rating: int = Field(..., ge=-1, le=1, description="1 = positive, -1 = negative")
    comment: Optional[str] = Field(None, max_length=2000)
    agent: Optional[str] = Field(None, max_length=64)

    @field_validator("rating")
    @classmethod
    def _non_zero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("rating must be 1 (positive) or -1 (negative)")
        return value


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

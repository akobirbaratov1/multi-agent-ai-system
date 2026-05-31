"""
Core State Schema for Multi-Agent AI System
LangGraph state management
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from enum import Enum
import operator


class AgentType(str, Enum):
    SALES = "sales"
    SUPPORT = "support"
    RESEARCH = "research"
    ORCHESTRATOR = "orchestrator"


class ConfidenceLevel(str, Enum):
    HIGH = "high"       # > 0.85 — auto proceed
    MEDIUM = "medium"   # 0.60 - 0.85 — proceed with caution
    LOW = "low"         # < 0.60 — human-in-the-loop


class IntentType(str, Enum):
    SALES = "sales"
    SUPPORT = "support"
    RESEARCH = "research"
    UNKNOWN = "unknown"


class AgentState(TypedDict):
    # Input
    user_id: str
    session_id: str
    user_message: str
    interface: str  # "web", "api", "telegram"

    # Validation
    is_valid: bool
    is_spam: bool
    validation_reason: Optional[str]

    # Context
    user_context: Dict[str, Any]
    conversation_history: Annotated[List[Dict], operator.add]

    # Routing
    intent: IntentType
    intent_confidence: float
    assigned_agent: AgentType

    # Agent outputs
    agent_response: Optional[str]
    agent_metadata: Dict[str, Any]

    # RAG
    retrieved_documents: List[Dict]
    rag_used: bool

    # Decision & Control
    confidence_level: ConfidenceLevel
    requires_human: bool
    human_approved: Optional[bool]
    human_feedback: Optional[str]

    # CRM Actions
    lead_created: bool
    lead_id: Optional[str]
    email_sent: bool
    crm_actions: List[str]

    # Final output
    final_response: str
    response_metadata: Dict[str, Any]

    # Monitoring
    trace_id: str
    latency_ms: float
    tokens_used: int
    error: Optional[str]

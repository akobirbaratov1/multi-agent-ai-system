"""
Research Agent — Information Retrieval, Analysis, Report Generation
RAG-powered with multi-source synthesis
"""

from core import config
from core.llm import LLMUnavailable, complete, parse_json_response
from core.logging_config import get_logger
from core.state import AgentState
from memory.vector_store import get_vector_store

logger = get_logger(__name__)


RESEARCH_SYSTEM_PROMPT = """You are an expert AI Research Agent. Your goal is to provide accurate, well-sourced information.

Your research methodology:
1. DECOMPOSE — Break the query into sub-questions
2. RETRIEVE — Search knowledge base for relevant information
3. SYNTHESIZE — Combine information from multiple sources
4. ANALYZE — Provide insights and conclusions
5. CITE — Reference your sources

Always:
- Be accurate and objective
- Clearly distinguish facts from analysis
- Cite knowledge base sources
- Provide structured, easy-to-read responses
- Flag uncertainty when present

Respond in JSON format:
{
    "response": "your research response with clear structure",
    "confidence": 0.0-1.0,
    "sources_used": ["source1", "source2"],
    "key_findings": ["finding1", "finding2"],
    "follow_up_questions": ["question1", "question2"],
    "requires_more_info": false
}"""

_DEFAULTS = {
    "response": "I couldn't find enough grounded information to answer that confidently.",
    "confidence": 0.5,
    "sources_used": [],
    "key_findings": [],
    "follow_up_questions": [],
    "requires_more_info": False,
}

_DEMO_RESPONSE = {
    "response": "## Multi-Agent AI Systems — Research Summary\n\n**Overview:**\nA multi-agent AI system is an architecture where multiple specialized AI agents collaborate to solve complex tasks. Each agent has a defined role and expertise domain.\n\n**Key Components (from KB):**\n- **Orchestrator**: LangGraph state machine that routes tasks to the right agent\n- **Intent Classification**: Claude-powered routing with confidence scoring\n- **RAG Pipeline**: FAISS vector search + document retrieval for grounded responses\n- **Human-in-the-Loop**: Low-confidence cases escalated to human review\n\n**Key Findings:**\n1. Specialized agents outperform single-agent systems on domain-specific tasks\n2. Confidence scoring reduces hallucination rates by routing uncertain cases to humans\n3. RAG significantly improves factual accuracy vs. pure parametric knowledge\n\n**Follow-up questions you might ask:**\n- How does confidence scoring work in practice?\n- What are the latency trade-offs of multi-agent systems?",
    "confidence": 0.94,
    "sources_used": ["Multi-Agent AI Overview", "Data Security & Privacy"],
    "key_findings": [
        "LangGraph enables stateful multi-agent orchestration",
        "FAISS provides sub-millisecond vector similarity search",
        "Confidence scoring gates human-in-the-loop escalation",
    ],
    "follow_up_questions": [
        "How does the confidence threshold affect routing decisions?",
        "What embedding model is used for RAG?",
    ],
    "requires_more_info": False,
}


def research_agent(state: AgentState) -> AgentState:
    """Research Agent — handles information retrieval and synthesis"""

    retrieved_docs = get_vector_store().search(query=state["user_message"], k=5)

    if config.DEMO_MODE:
        result = dict(_DEMO_RESPONSE)
    else:
        context = "\n\n".join(
            f"[Source: {doc.get('title', 'Untitled')}]\n{doc.get('content', '')}"
            for doc in retrieved_docs
        )

        try:
            text = complete(
                system=RESEARCH_SYSTEM_PROMPT,
                model=config.AGENT_MODEL,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Research query: {state['user_message']}\n\n"
                        "Available sources:\n"
                        f"{context if context else 'No specific sources found. Use general knowledge.'}\n\n"
                        "Provide a comprehensive, well-structured research response."
                    ),
                }],
            )
            result = parse_json_response(text, _DEFAULTS)
        except LLMUnavailable:
            logger.warning("Research agent model call failed; escalating to a human")
            return {
                **state,
                "agent_response": None,
                "agent_metadata": {"error": "model_unavailable"},
                "retrieved_documents": retrieved_docs,
                "rag_used": bool(retrieved_docs),
                "requires_human": True,
                "final_response": (
                    "I'm having trouble reaching our systems right now. "
                    "I've flagged this for a human operator who will follow up shortly."
                ),
                "error": "model_unavailable",
            }

    return {
        **state,
        "agent_response": result["response"],
        "agent_metadata": {
            "confidence": result["confidence"],
            "sources_used": result["sources_used"],
            "key_findings": result["key_findings"],
            "follow_up_questions": result["follow_up_questions"],
        },
        "retrieved_documents": retrieved_docs,
        "rag_used": len(retrieved_docs) > 0,
        "final_response": result["response"],
    }

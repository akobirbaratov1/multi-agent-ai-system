# Multi-Agent AI System — Production-Ready Orchestration

> Built with LangGraph • Claude API • FAISS • FastAPI • Web UI

![Tests](https://github.com/yourusername/multi-agent-ai-system/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Architecture Overview

```
User (Web / API)
        ↓
   Input Layer
   - Validation
   - Spam Filter
        ↓
  Orchestrator Agent (LangGraph)
   - Intent Classification (Claude)
   - Confidence Scoring
        ↓
  ┌─────────────────────────┐
  │      Agent Layer        │
  │  ┌──────────────────┐   │
  │  │   Sales Agent    │   │  → Lead Qualification → CRM
  │  ├──────────────────┤   │
  │  │  Support Agent   │   │  → RAG → Problem Solving
  │  ├──────────────────┤   │
  │  │ Research Agent   │   │  → RAG → Information Synthesis
  │  └──────────────────┘   │
  └─────────────────────────┘
        ↓
  Memory & Knowledge
   - FAISS Vector DB (IndexFlatIP)
   - Conversation History
   - Knowledge Base
        ↓
  Decision & Control
   - Confidence Scoring
   - Human-in-the-Loop
        ↓
  Action Layer
   - Mock CRM (HubSpot-compatible)
   - Email Dispatch
        ↓
  Monitoring
   - LangSmith Traces (optional)
   - Custom Metrics
   - Human Feedback
```

---

## Features

- **Multi-Agent Orchestration** — LangGraph state machine with conditional routing
- **3 Specialized Agents** — Sales, Support, Research
- **RAG System** — FAISS IndexFlatIP + cosine similarity document retrieval
- **Human-in-the-Loop** — confidence scoring with auto/manual routing
- **Mock CRM** — HubSpot-compatible lead management and activity logging
- **Email Dispatch** — template-based email mock (SendGrid-compatible structure)
- **Web UI** — real-time dark-themed chat interface
- **REST API** — FastAPI with full OpenAPI docs at `/docs`
- **Monitoring** — LangSmith traces + custom metrics + human feedback
- **Automated Evaluation** — test datasets + accuracy scoring
- **Docker Ready** — single-command deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude Sonnet 4.6 (Anthropic) |
| Orchestration | LangGraph |
| RAG | FAISS IndexFlatIP + numpy |
| Backend | FastAPI + Python 3.11 |
| Frontend | HTML/CSS/JS (single file, no build step) |
| Monitoring | LangSmith + local JSONL metrics |
| Container | Docker + Docker Compose |

---

## Project Structure

```
multi-agent-ai-system/
├── agents/
│   ├── orchestrator.py       # LangGraph state machine
│   ├── sales_agent.py        # Lead qualification & CRM
│   ├── support_agent.py      # FAQ & RAG problem solving
│   └── research_agent.py     # RAG information synthesis
├── core/
│   ├── state.py              # AgentState TypedDict schema
│   ├── router.py             # Intent classification & routing
│   └── confidence.py         # Confidence scoring utilities
├── memory/
│   ├── vector_store.py       # FAISS vector store
│   ├── knowledge_base.py     # Document ingestion & management
│   └── conversation.py       # Per-session history management
├── tools/
│   ├── crm.py                # Mock CRM (HubSpot-compatible)
│   ├── email.py              # Email dispatch (mock SMTP)
│   └── search.py             # Knowledge base search utilities
├── api/
│   ├── main.py               # FastAPI app entry point
│   ├── schemas.py            # Pydantic request/response models
│   └── routes/
│       ├── chat.py           # POST /chat
│       ├── knowledge.py      # /knowledge/*
│       ├── crm.py            # /crm/*
│       └── monitoring.py     # /monitoring/*
├── monitoring/
│   ├── langsmith.py          # LangSmith integration (graceful fallback)
│   └── metrics.py            # Custom latency & usage metrics
├── evaluation/
│   ├── auto_eval.py          # Automated accuracy evaluation
│   ├── feedback.py           # Human feedback collection
│   └── datasets/
│       └── sample.json       # Labeled evaluation dataset
├── tests/
│   ├── test_agents.py        # Agent unit tests (pytest)
│   ├── test_api.py           # API integration tests
│   └── test_rag.py           # Vector store & RAG tests
├── web/
│   └── index.html            # Web UI (dark theme, real-time chat)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Demo

> **Try it without an API key** using demo mode — pre-built responses showcase the full routing flow.

```bash
# Demo mode (no API key needed)
DEMO_MODE=true uvicorn api.main:app --reload
# Open http://localhost:8000 and try:
# → "What are your pricing plans?"    → Sales Agent
# → "I can't connect to the API"      → Support Agent + RAG
# → "Tell me about multi-agent AI"    → Research Agent
```

**Suggested Loom walkthrough:**
1. Open `/docs` — show the full OpenAPI spec
2. `GET /knowledge/init` — populate the knowledge base live
3. Chat UI — send one message per agent type, show routing badges
4. `GET /crm/leads` — show a lead was auto-created by Sales Agent
5. `GET /monitoring/metrics` — show live latency & RAG usage stats

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/multi-agent-ai-system.git
cd multi-agent-ai-system
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Initialize Knowledge Base & Run

```bash
uvicorn api.main:app --reload
# Then open: http://localhost:8000
# Initialize KB:  http://localhost:8000/knowledge/init
# API Docs:       http://localhost:8000/docs
```

### 4. Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Main agent chat endpoint |
| GET | `/health` | System health + stats |
| POST | `/knowledge/add` | Add document to KB |
| GET | `/knowledge/search?query=...` | Search KB |
| GET | `/knowledge/init` | Load sample KB |
| GET | `/crm/stats` | CRM dashboard |
| GET | `/crm/leads` | List all leads |
| GET | `/monitoring/metrics` | Latency & usage metrics |
| POST | `/monitoring/feedback` | Submit response rating |
| GET | `/monitoring/tracing` | LangSmith trace status |

Full OpenAPI spec available at `/docs`.

---

## Agents

### Sales Agent
- Lead qualification & scoring (0-100)
- Multi-turn sales conversation
- Objection handling
- CRM lead creation on score >= 60
- Demo request logging

### Support Agent
- RAG-powered FAQ retrieval (top-3 KB articles)
- Problem diagnosis & resolution
- Confidence-based escalation to human

### Research Agent
- RAG document synthesis (top-5 KB articles)
- Multi-source information analysis
- Structured report generation with citations

---

## Confidence Routing

| Level | Score | Behavior |
|-------|-------|----------|
| HIGH | >= 0.85 | Auto-route to agent |
| MEDIUM | 0.60–0.84 | Route with caution |
| LOW | < 0.60 | Escalate to human review |

---

## Monitoring

LangSmith tracing is **optional** — the system works without it and falls back to local `memory/data/traces.jsonl`.

```bash
# Enable full tracing:
LANGSMITH_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=multi-agent-ai-system
```

---

## Author

**Botirjon Akramov** — Senior Agentic AI Engineer  
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

> *"I don't just write code. I architect systems that run."*

---

## License

MIT License — feel free to use and adapt.

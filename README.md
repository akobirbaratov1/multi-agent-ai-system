# Multi-Agent AI System — Production-Ready Orchestration

> Built with LangGraph • Claude API • FAISS • FastAPI • Web UI

![Tests](https://github.com/akobirbarotovdev/multi-agent-ai-system/actions/workflows/tests.yml/badge.svg)
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
- **Security Check** — rate limiting, prompt injection detection, input sanitization
- **3 Specialized Agents** — Sales, Support, Research
- **RAG System** — FAISS IndexFlatIP + cosine similarity document retrieval
- **Human-in-the-Loop** — Operator Dashboard with approve/reject/override UI
- **Telegram Bot** — full bot interface via the same orchestrator pipeline
- **Mock CRM** — HubSpot-compatible lead management, activity logging, follow-ups
- **Email Dispatch** — template-based email mock (SendGrid-compatible structure)
- **Follow-up System** — scheduled 24h follow-ups, drained by a background worker
- **Web UI** — real-time dark-themed chat + Operator Dashboard
- **REST API** — FastAPI with full OpenAPI docs at `/docs`
- **Monitoring** — LangSmith traces + custom metrics + human feedback
- **Feedback Loop** — performance analysis, improvement suggestions, weekly reports
- **Automated Evaluation** — test datasets + accuracy scoring
- **API-key auth** — admin endpoints (KB writes, operator dashboard, CRM) gated by `ADMIN_API_KEY`
- **Signed webhooks** — HMAC-SHA256 verification on the 2Chat inbound path
- **Structured logging** — JSON logs with a correlation ID on every request and response
- **Health probes** — `/health/live` and `/health/ready` for orchestrators
- **Fail-fast config** — the app refuses to start in production when misconfigured
- **Docker Ready** — non-root image, healthcheck, single-command deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude (Anthropic) — model configurable via `AGENT_MODEL` / `ROUTER_MODEL` |
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
│   ├── config.py             # Central env-driven configuration + validation
│   ├── state.py              # AgentState TypedDict schema
│   ├── router.py             # Intent classification & routing
│   ├── confidence.py         # Confidence scoring utilities
│   ├── security.py           # Rate limiting, injection detection, auth helpers
│   ├── llm.py                # Anthropic client + tolerant JSON response parsing
│   ├── storage.py            # Atomic, lock-guarded JSON/JSONL persistence
│   ├── review_queue.py       # Human-in-the-loop queue (storage layer)
│   └── logging_config.py     # Structured logging + request correlation IDs
├── memory/
│   ├── vector_store.py       # FAISS vector store
│   ├── knowledge_base.py     # Document ingestion & management
│   └── conversation.py       # Per-session history management
├── tools/
│   ├── crm.py                # Mock CRM (HubSpot-compatible)
│   ├── email.py              # Email dispatch (mock SMTP)
│   ├── search.py             # Knowledge base search utilities
│   ├── followup.py           # Scheduled follow-up queue
│   └── scheduler.py          # Background worker that drains the queue
├── integrations/
│   └── twochat.py            # 2Chat WhatsApp/SMS integration
├── telegram_bot/
│   └── bot.py                # Telegram bot (polling mode)
├── api/
│   ├── main.py               # FastAPI app entry point
│   ├── deps.py               # Shared dependencies (admin auth)
│   ├── schemas.py            # Pydantic request/response models
│   └── routes/
│       ├── chat.py           # POST /chat
│       ├── knowledge.py      # /knowledge/*
│       ├── crm.py            # /crm/*
│       ├── operator.py       # /operator/* (Human-in-the-Loop)
│       ├── webhooks.py       # /webhook/2chat, /webhook/...
│       └── monitoring.py     # /monitoring/*
├── monitoring/
│   ├── langsmith.py          # LangSmith integration (graceful fallback)
│   └── metrics.py            # Custom latency & usage metrics
├── feedback_loop/
│   ├── analyzer.py           # Performance & log analysis
│   ├── improver.py           # Prompt improvement suggestions
│   └── reporter.py           # Weekly performance reports
├── evaluation/
│   ├── auto_eval.py          # Automated accuracy evaluation
│   ├── feedback.py           # Human feedback collection
│   └── datasets/
│       └── sample.json       # Labeled evaluation dataset
├── tests/
│   ├── conftest.py           # Per-test data isolation & fixtures
│   ├── test_agents.py        # Agent + orchestrator tests
│   ├── test_api.py           # API integration tests
│   ├── test_rag.py           # Vector store & RAG tests
│   └── test_infrastructure.py # Storage, security, config, metrics tests
├── web/
│   ├── index.html            # Web UI (dark theme, real-time chat)
│   └── operator.html         # Operator Dashboard (Human-in-the-Loop)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── pyproject.toml            # pytest & ruff configuration
├── requirements.txt
└── requirements-dev.txt
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
git clone https://github.com/akobirbarotovdev/multi-agent-ai-system.git
cd multi-agent-ai-system
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # + tests and linting
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
# Initialize KB:  curl -X POST http://localhost:8000/knowledge/init
# API Docs:       http://localhost:8000/docs
```

### 4. Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### 5. Run Tests

```bash
pytest tests/ -v      # 147 tests, no API key required
ruff check .          # lint
```

---

## Configuration

Every setting is read from the environment (or `.env`) through `core/config.py`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | Required unless `DEMO_MODE=true` |
| `DEMO_MODE` | `false` | Deterministic mock responses; no API key needed |
| `APP_ENV` | `development` | `production` enables strict startup validation |
| `LOG_FORMAT` | `text` (`json` in prod) | Structured JSON logs for aggregators |
| `DATA_DIR` | `memory/data` | Where runtime state is persisted |
| `ROUTER_MODEL` / `AGENT_MODEL` | `claude-sonnet-4-6` | Swap models without a code change |
| `ANTHROPIC_TIMEOUT` / `ANTHROPIC_MAX_RETRIES` | `30` / `3` | Model call resilience |
| `ADMIN_API_KEY` | — | Guards admin endpoints. **Required in production** |
| `CORS_ORIGINS` | `*` | Explicit origin list. `*` is **rejected in production** |
| `TWOCHAT_WEBHOOK_SECRET` | — | HMAC secret for inbound webhook verification |
| `RATE_LIMIT_MAX` / `RATE_LIMIT_WINDOW` | `30` / `60` | Per-user sliding window |
| `MAX_MESSAGE_LENGTH` | `5000` | Rejected at the API edge, before model spend |
| `FOLLOWUP_INTERVAL_SECONDS` | `0` (off) | Background follow-up sweep interval |

### Production checklist

`APP_ENV=production` makes the app **fail to start** rather than serve traffic
insecurely. It requires all of:

- `ADMIN_API_KEY` set — otherwise the operator dashboard, CRM reads and
  knowledge-base writes would be unauthenticated
- `CORS_ORIGINS` set to an explicit list — `*` is rejected
- `DEMO_MODE=false` with a valid `ANTHROPIC_API_KEY`

Also recommended: set `TWOCHAT_WEBHOOK_SECRET` if the 2Chat webhook is exposed,
and run the follow-up scheduler in exactly one replica.

### Authentication

Admin endpoints take the key in an `X-API-Key` header:

```bash
curl -H "X-API-Key: $ADMIN_API_KEY" http://localhost:8000/operator/cases
```

The operator dashboard prompts for the key and stores it in `localStorage`.
When `ADMIN_API_KEY` is unset (local development) the guard is a no-op.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
🔒 = requires `X-API-Key: $ADMIN_API_KEY` when one is configured.

| Method | Endpoint | Auth | Description |
|--------|----------|:----:|-------------|
| POST | `/chat` | | Main agent chat endpoint |
| GET | `/health` | | System health + stats |
| GET | `/health/live` | | Liveness probe (no dependencies touched) |
| GET | `/health/ready` | | Readiness probe — 503 when misconfigured |
| POST | `/knowledge/add` | 🔒 | Add document to KB |
| GET | `/knowledge/search?query=...` | | Search KB |
| POST | `/knowledge/init` | 🔒 | Load sample KB |
| GET | `/knowledge/stats` | | KB statistics |
| GET | `/crm/stats` | | CRM dashboard |
| GET | `/crm/leads` | 🔒 | List leads (paginated) |
| GET | `/crm/activities` | 🔒 | CRM activity log (paginated) |
| GET | `/operator` | | Operator Dashboard UI |
| GET | `/operator/cases` | 🔒 | List human-review cases |
| POST | `/operator/cases/{id}/review` | 🔒 | Approve / reject / override |
| GET | `/operator/stats` | 🔒 | Human-in-the-Loop stats |
| GET | `/monitoring/metrics` | | Latency & usage metrics |
| POST | `/monitoring/feedback` | | Submit response rating |
| GET | `/monitoring/tracing` | | LangSmith trace status |
| GET | `/monitoring/analysis` | 🔒 | Performance analysis (7-day) |
| GET | `/monitoring/improvements` | 🔒 | Improvement suggestions |
| GET | `/monitoring/report` | 🔒 | Full weekly performance report |
| GET | `/monitoring/followups` | 🔒 | Follow-up queue stats |
| POST | `/webhook/2chat` | signature | 2Chat inbound webhook (HMAC-verified) |
| GET | `/webhook/2chat/status` | | 2Chat configuration status |
| POST | `/webhook/2chat/test` | 🔒 | Test 2Chat pipeline (no real API call) |

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

## Telegram Bot

The same orchestrator pipeline is available via Telegram.

```bash
# Add to .env:
TELEGRAM_BOT_TOKEN=your_token_from_BotFather

# Run bot:
python -m telegram_bot.bot
```

Commands: `/start`, `/help`, `/clear`

---

## Operator Dashboard

Human-in-the-Loop review interface at `/operator`.

Operators can:
- View all low-confidence cases escalated by the system
- **Approve** — accept the agent's response as-is
- **Reject** — discard and flag for retraining
- **Override** — replace the agent's response with a custom one

Dashboard auto-refreshes every 15 seconds.

---

## Feedback Loop

Automated weekly analysis pipeline:

```bash
# Via API:
GET /monitoring/analysis      # 7-day performance breakdown
GET /monitoring/improvements  # Claude-generated prompt suggestions
GET /monitoring/report        # Full executive summary

# Via Python:
from feedback_loop.reporter import generate_weekly_report, print_report
print_report(generate_weekly_report())
```

---

## Confidence Routing

| Level | Score | Behavior |
|-------|-------|----------|
| HIGH | >= 0.85 | Auto-route to agent |
| MEDIUM | 0.60–0.84 | Route with caution |
| LOW | < 0.60 | Escalate to human review |

Thresholds are configurable via `CONFIDENCE_THRESHOLD_HIGH` and
`CONFIDENCE_THRESHOLD_MEDIUM`.

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
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/akobirbaratov1)

> *"I don't just write code. I architect systems that run."*

---

## License

MIT License — feel free to use and adapt.

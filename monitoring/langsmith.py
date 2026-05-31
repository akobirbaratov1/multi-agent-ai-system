"""
LangSmith Integration — Trace & observe agent runs.
Falls back gracefully when LANGSMITH_API_KEY is not set.
"""

import os
import time
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

_LANGSMITH_AVAILABLE = False
_client = None

try:
    from langsmith import Client
    if os.getenv("LANGSMITH_API_KEY"):
        _client = Client()
        _LANGSMITH_AVAILABLE = True
except ImportError:
    pass

LOCAL_TRACE_PATH = Path("memory/data/traces.jsonl")


def _save_local_trace(trace: dict):
    """Persist trace locally when LangSmith is not configured."""
    LOCAL_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_TRACE_PATH, "a") as f:
        f.write(json.dumps(trace, default=str) + "\n")


class AgentTracer:
    """
    Wraps an agent run with LangSmith tracing.
    Falls back to local JSONL file when LangSmith is unavailable.
    """

    def __init__(self, run_name: str, metadata: Optional[Dict] = None):
        self.run_name = run_name
        self.trace_id = str(uuid.uuid4())
        self.metadata = metadata or {}
        self.start_time = time.time()
        self.events: list = []

    def log_event(self, name: str, data: Any = None):
        """Log a named event within this trace."""
        self.events.append({
            "name": name,
            "data": data,
            "ts": datetime.now().isoformat(),
        })

    def finish(self, output: Any = None, error: Optional[str] = None) -> dict:
        """Finalize the trace and send to LangSmith or local storage."""
        latency_ms = round((time.time() - self.start_time) * 1000, 2)

        trace = {
            "trace_id": self.trace_id,
            "run_name": self.run_name,
            "metadata": self.metadata,
            "events": self.events,
            "output": output,
            "error": error,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat(),
        }

        if _LANGSMITH_AVAILABLE and _client:
            try:
                _client.create_run(
                    name=self.run_name,
                    run_type="chain",
                    inputs=self.metadata,
                    outputs={"result": output} if output else {},
                    error=error,
                    extra={"metadata": self.metadata},
                )
            except Exception:
                _save_local_trace(trace)
        else:
            _save_local_trace(trace)

        return trace


def trace_agent_run(
    agent_name: str,
    user_message: str,
    session_id: str,
    result: Dict,
) -> str:
    """Convenience function — trace a completed agent run."""
    tracer = AgentTracer(
        run_name=f"{agent_name}_run",
        metadata={
            "agent": agent_name,
            "session_id": session_id,
            "user_message": user_message[:200],
        },
    )
    tracer.log_event("agent_response", result.get("response", "")[:500])
    finished = tracer.finish(output=result)
    return finished["trace_id"]


def is_tracing_enabled() -> bool:
    return _LANGSMITH_AVAILABLE


def get_trace_status() -> dict:
    return {
        "langsmith_enabled": _LANGSMITH_AVAILABLE,
        "local_traces": LOCAL_TRACE_PATH.exists(),
        "project": os.getenv("LANGSMITH_PROJECT", "multi-agent-ai-system"),
    }

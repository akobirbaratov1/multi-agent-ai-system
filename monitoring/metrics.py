"""
Custom Metrics — Latency, token usage, agent performance tracking.
Stores metrics locally; can be forwarded to Prometheus/Datadog.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

METRICS_PATH = Path("memory/data/metrics.jsonl")


def _append(record: dict):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _load_all() -> List[dict]:
    if not METRICS_PATH.exists():
        return []
    records = []
    with open(METRICS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def record_request(
    agent: str,
    intent: str,
    latency_ms: float,
    tokens_used: int = 0,
    rag_used: bool = False,
    confidence: float = 0.0,
    error: Optional[str] = None,
):
    """Record a single request metric event."""
    _append({
        "event": "request",
        "agent": agent,
        "intent": intent,
        "latency_ms": round(latency_ms, 2),
        "tokens_used": tokens_used,
        "rag_used": rag_used,
        "confidence": round(confidence, 3),
        "error": error,
        "timestamp": datetime.now().isoformat(),
    })


def get_summary() -> Dict:
    """Aggregate metrics summary."""
    records = [r for r in _load_all() if r.get("event") == "request"]

    if not records:
        return {"total_requests": 0}

    latencies = [r["latency_ms"] for r in records if r.get("latency_ms")]
    by_agent: Dict[str, int] = defaultdict(int)
    by_intent: Dict[str, int] = defaultdict(int)
    errors = 0
    rag_count = 0

    for r in records:
        by_agent[r.get("agent", "unknown")] += 1
        by_intent[r.get("intent", "unknown")] += 1
        if r.get("error"):
            errors += 1
        if r.get("rag_used"):
            rag_count += 1

    return {
        "total_requests": len(records),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
        "error_rate": round(errors / len(records), 3),
        "rag_usage_rate": round(rag_count / len(records), 3),
        "requests_by_agent": dict(by_agent),
        "requests_by_intent": dict(by_intent),
    }

"""
Custom Metrics — Latency, token usage, agent performance tracking.

Records are appended to a JSONL log for durability, and an in-process rolling
aggregate answers `/monitoring/metrics` without re-reading the file. The
previous implementation parsed the entire log on every request, so dashboard
latency grew linearly with traffic and eventually dominated the endpoint.
"""

import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

from core import config
from core.storage import append_jsonl, iter_jsonl, tail_jsonl

METRICS_PATH = config.data_path("metrics.jsonl")

# Number of recent requests kept in memory for percentile calculations.
WINDOW_SIZE = 5_000


class _Aggregate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: Deque[float] = deque(maxlen=WINDOW_SIZE)
        self._by_agent: Dict[str, int] = defaultdict(int)
        self._by_intent: Dict[str, int] = defaultdict(int)
        self._total = 0
        self._errors = 0
        self._rag = 0
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Seed the aggregate from the tail of the log on first use."""
        if self._loaded:
            return
        self._loaded = True
        for record in tail_jsonl(METRICS_PATH, WINDOW_SIZE):
            if record.get("event") == "request":
                self._apply(record)

    def _apply(self, record: dict) -> None:
        self._total += 1
        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)):
            self._latencies.append(float(latency))
        self._by_agent[record.get("agent") or "unknown"] += 1
        self._by_intent[record.get("intent") or "unknown"] += 1
        if record.get("error"):
            self._errors += 1
        if record.get("rag_used"):
            self._rag += 1

    def add(self, record: dict) -> None:
        with self._lock:
            self._ensure_loaded()
            self._apply(record)

    def summary(self) -> Dict:
        with self._lock:
            self._ensure_loaded()
            total = self._total
            latencies = sorted(self._latencies)
            by_agent = dict(self._by_agent)
            by_intent = dict(self._by_intent)
            errors, rag = self._errors, self._rag

        if not total:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "error_rate": 0,
                "rag_usage_rate": 0,
                "requests_by_agent": {},
                "requests_by_intent": {},
            }

        return {
            "total_requests": total,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 1) if latencies else 0,
            "p99_latency_ms": round(_percentile(latencies, 0.99), 1) if latencies else 0,
            "error_rate": round(errors / total, 3),
            "rag_usage_rate": round(rag / total, 3),
            "requests_by_agent": by_agent,
            "requests_by_intent": by_intent,
            "window_size": len(latencies),
        }

    def reset(self) -> None:
        with self._lock:
            self.__init__()


def _percentile(sorted_values: List[float], fraction: float) -> float:
    """Nearest-rank percentile; index is clamped so it can never overrun."""
    if not sorted_values:
        return 0.0
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]


_aggregate = _Aggregate()


def record_request(
    agent: str,
    intent: str,
    latency_ms: float,
    tokens_used: int = 0,
    rag_used: bool = False,
    confidence: float = 0.0,
    error: Optional[str] = None,
) -> None:
    """Record a single request metric event."""
    record = {
        "event": "request",
        "agent": agent,
        "intent": intent,
        "latency_ms": round(float(latency_ms or 0), 2),
        "tokens_used": tokens_used,
        "rag_used": bool(rag_used),
        "confidence": round(float(confidence or 0), 3),
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Update the aggregate first: its lazy seed reads the tail of the log, so
    # appending beforehand makes the very first record count twice.
    _aggregate.add(record)
    append_jsonl(METRICS_PATH, record)


def get_summary() -> Dict:
    """Aggregate metrics summary over the rolling window."""
    return _aggregate.summary()


def reset_metrics() -> None:
    """Clear the in-memory aggregate. Used by tests."""
    _aggregate.reset()


def _load_all() -> List[dict]:
    """Full history from disk. Used by the feedback-loop analyzer."""
    return list(iter_jsonl(METRICS_PATH))

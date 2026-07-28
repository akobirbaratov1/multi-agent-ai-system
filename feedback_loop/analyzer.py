"""
Logs Analyzer — Analyzes system metrics, feedback, and traces to identify improvement areas.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from core import config
from core.storage import read_jsonl


def _load_jsonl(path: Path) -> List[dict]:
    return read_jsonl(path)


def _parse_ts(value: str) -> Optional[datetime]:
    """
    Parse a stored timestamp as an aware datetime.

    Records written before 1.1 carry naive local timestamps; comparing those
    against an aware cutoff raises TypeError and would break every report.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def analyze_performance(days: int = 7) -> Dict:
    """
    Analyze system performance over the past N days.

    Returns:
        Dict with performance insights and problem areas
    """
    from evaluation.feedback import _load_all as load_feedback
    from monitoring.metrics import _load_all as load_metrics

    since = datetime.now(timezone.utc) - timedelta(days=days)

    metrics = [
        r for r in load_metrics()
        if (_parse_ts(r.get("timestamp", "")) or since) > since
    ]
    feedback = [
        r for r in load_feedback()
        if (_parse_ts(r.get("submitted_at", "")) or since) > since
    ]

    # Latency analysis
    latencies = [r["latency_ms"] for r in metrics if r.get("latency_ms")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    slow_requests = [r for r in metrics if r.get("latency_ms", 0) > 3000]

    # Error analysis
    errors = [r for r in metrics if r.get("error")]
    error_rate = len(errors) / len(metrics) if metrics else 0

    # Agent performance
    agent_feedback: Dict[str, Dict] = defaultdict(lambda: {"positive": 0, "negative": 0})
    for f in feedback:
        agent = f.get("agent", "unknown")
        if f.get("rating") == 1:
            agent_feedback[agent]["positive"] += 1
        else:
            agent_feedback[agent]["negative"] += 1

    # Satisfaction per agent
    agent_satisfaction = {}
    for agent, counts in agent_feedback.items():
        total = counts["positive"] + counts["negative"]
        agent_satisfaction[agent] = round(counts["positive"] / total, 2) if total else None

    # Problem areas
    problems = []
    if error_rate > 0.05:
        problems.append(f"High error rate: {error_rate:.1%}")
    if avg_latency > 2000:
        problems.append(f"High avg latency: {avg_latency:.0f}ms")
    if len(slow_requests) > 0:
        problems.append(f"{len(slow_requests)} slow requests (>3s)")
    for agent, sat in agent_satisfaction.items():
        if sat is not None and sat < 0.7:
            problems.append(f"{agent} agent satisfaction below 70%: {sat:.0%}")

    return {
        "period_days": days,
        "total_requests": len(metrics),
        "avg_latency_ms": round(avg_latency, 1),
        "error_rate": round(error_rate, 3),
        "slow_requests": len(slow_requests),
        "total_feedback": len(feedback),
        "agent_satisfaction": agent_satisfaction,
        "problem_areas": problems,
        "health_score": _compute_health_score(error_rate, avg_latency, agent_satisfaction),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def _compute_health_score(error_rate: float, avg_latency: float, satisfaction: dict) -> int:
    """Compute an overall system health score 0-100."""
    score = 100
    score -= min(30, error_rate * 300)
    if avg_latency > 1000:
        score -= min(20, (avg_latency - 1000) / 100)
    for sat in satisfaction.values():
        if sat is not None and sat < 0.8:
            score -= 10
    return max(0, round(score))


def get_top_issues(limit: int = 5) -> List[Dict]:
    """Return top recurring user issues from traces."""
    traces = _load_jsonl(config.data_path("traces.jsonl"))
    intent_counts: Dict[str, int] = defaultdict(int)
    for trace in traces:
        meta = trace.get("metadata", {})
        intent = meta.get("intent", "unknown")
        intent_counts[intent] += 1

    sorted_issues = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"intent": k, "count": v} for k, v in sorted_issues[:limit]]

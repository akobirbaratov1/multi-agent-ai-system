"""
Reporter — Weekly performance report generator.
Combines metrics, feedback, and improvement suggestions into a readable summary.
"""

from datetime import datetime, timezone
from typing import Dict


def generate_weekly_report() -> Dict:
    """Generate a full weekly performance report."""
    from evaluation.feedback import get_feedback_stats
    from feedback_loop.analyzer import analyze_performance, get_top_issues
    from feedback_loop.improver import generate_improvement_suggestions
    from monitoring.metrics import get_summary

    analysis = analyze_performance(days=7)
    metrics = get_summary()
    feedback = get_feedback_stats()
    top_issues = get_top_issues(limit=5)
    suggestions = generate_improvement_suggestions(analysis)

    report = {
        "report_type": "weekly",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": "last 7 days",
        "executive_summary": {
            "health_score": analysis["health_score"],
            "total_requests": metrics.get("total_requests", 0),
            "avg_latency_ms": metrics.get("avg_latency_ms", 0),
            "error_rate": f"{metrics.get('error_rate', 0):.1%}",
            "satisfaction_rate": f"{feedback.get('satisfaction_rate', 0):.1%}",
            "rag_usage_rate": f"{metrics.get('rag_usage_rate', 0):.1%}",
        },
        "agent_performance": {
            "by_volume": metrics.get("requests_by_agent", {}),
            "satisfaction": analysis.get("agent_satisfaction", {}),
        },
        "top_user_intents": top_issues,
        "problem_areas": analysis.get("problem_areas", []),
        "improvement_suggestions": [
            {"priority": s["priority"], "suggestion": s["suggestion"]}
            for s in suggestions
        ],
    }

    return report


def print_report(report: Dict):
    """Pretty-print a weekly report to stdout."""
    print("\n" + "═" * 60)
    print(f"  WEEKLY PERFORMANCE REPORT — {report['generated_at'][:10]}")
    print("═" * 60)

    s = report["executive_summary"]
    print(f"\n  Health Score: {s['health_score']}/100")
    print(f"  Requests:     {s['total_requests']}")
    print(f"  Avg Latency:  {s['avg_latency_ms']}ms")
    print(f"  Error Rate:   {s['error_rate']}")
    print(f"  Satisfaction: {s['satisfaction_rate']}")

    if report["problem_areas"]:
        print("\n  ⚠️  Problem Areas:")
        for p in report["problem_areas"]:
            print(f"     • {p}")

    if report["improvement_suggestions"]:
        print("\n  💡 Improvement Suggestions:")
        for s in report["improvement_suggestions"]:
            icon = "🔴" if s["priority"] == "high" else "🟡" if s["priority"] == "medium" else "🟢"
            print(f"     {icon} {s['suggestion']}")

    print("\n" + "═" * 60 + "\n")

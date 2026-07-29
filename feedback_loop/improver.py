"""
Agent Improver — Generates prompt improvement suggestions based on feedback analysis.
Uses Claude to analyze negative feedback patterns and suggest prompt changes.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


def generate_improvement_suggestions(analysis: Dict) -> List[Dict]:
    """
    Generate prompt improvement suggestions based on performance analysis.

    Args:
        analysis: Output from feedback_loop.analyzer.analyze_performance()

    Returns:
        List of improvement suggestions with priority and rationale
    """
    suggestions = []

    # Rule-based suggestions from analysis data
    if analysis.get("error_rate", 0) > 0.05:
        suggestions.append({
            "priority": "high",
            "area": "error_handling",
            "suggestion": "Add more explicit JSON format enforcement in system prompts",
            "rationale": f"Error rate {analysis['error_rate']:.1%} — likely JSON parse failures",
            "affected_agents": ["sales", "support", "research"],
        })

    if analysis.get("avg_latency_ms", 0) > 2000:
        suggestions.append({
            "priority": "medium",
            "area": "performance",
            "suggestion": "Reduce max_tokens in router from 500 to 300 — classification needs less output",
            "rationale": f"Avg latency {analysis['avg_latency_ms']:.0f}ms — router may be bottleneck",
            "affected_agents": ["router"],
        })

    satisfaction = analysis.get("agent_satisfaction", {})
    for agent, sat in satisfaction.items():
        if sat is not None and sat < 0.7:
            suggestions.append({
                "priority": "high",
                "area": "response_quality",
                "suggestion": f"Review and strengthen {agent} agent system prompt with more specific examples",
                "rationale": f"{agent} satisfaction {sat:.0%} — responses not meeting user expectations",
                "affected_agents": [agent],
            })

    for problem in analysis.get("problem_areas", []):
        if "slow" in problem.lower():
            suggestions.append({
                "priority": "medium",
                "area": "latency",
                "suggestion": "Enable prompt caching for static system prompts (saves ~30% latency)",
                "rationale": problem,
                "affected_agents": ["all"],
            })

    if not suggestions:
        suggestions.append({
            "priority": "low",
            "area": "maintenance",
            "suggestion": "System performing well. Consider A/B testing response formats for higher engagement.",
            "rationale": "No critical issues detected",
            "affected_agents": [],
        })

    return sorted(suggestions, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s["priority"]])


def save_improvement_report(suggestions: List[Dict], analysis: Dict) -> str:
    """Save improvement report to disk and return the file path."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health_score": analysis.get("health_score", 0),
        "problem_areas": analysis.get("problem_areas", []),
        "suggestions": suggestions,
        "summary": f"{len(suggestions)} improvement(s) identified, "
                   f"{sum(1 for s in suggestions if s['priority'] == 'high')} high priority",
    }

    from core import config
    from core.storage import write_json

    path = config.data_path(
        f"improvement_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    write_json(path, report)

    return str(path)


def run_improvement_cycle() -> Dict:
    """Full improvement cycle: analyze → suggest → save report."""
    from feedback_loop.analyzer import analyze_performance

    analysis = analyze_performance(days=7)
    suggestions = generate_improvement_suggestions(analysis)
    report_path = save_improvement_report(suggestions, analysis)

    return {
        "health_score": analysis["health_score"],
        "suggestions_count": len(suggestions),
        "high_priority": sum(1 for s in suggestions if s["priority"] == "high"),
        "report_path": report_path,
        "suggestions": suggestions,
    }

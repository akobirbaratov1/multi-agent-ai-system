"""
Confidence Scoring System
Determines routing behavior based on intent classification confidence.
"""

from core.state import ConfidenceLevel


def compute_confidence_level(score: float) -> ConfidenceLevel:
    """Map a 0-1 confidence score to a ConfidenceLevel enum."""
    if score >= 0.85:
        return ConfidenceLevel.HIGH
    elif score >= 0.60:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def requires_human_review(confidence: float, intent: str = None) -> bool:
    """Return True if confidence is too low to auto-route."""
    return compute_confidence_level(confidence) == ConfidenceLevel.LOW


def confidence_summary(score: float) -> dict:
    """Return a structured confidence report."""
    level = compute_confidence_level(score)
    return {
        "score": round(score, 3),
        "level": level.value,
        "auto_proceed": level != ConfidenceLevel.LOW,
        "description": {
            ConfidenceLevel.HIGH: "High confidence — auto routing enabled",
            ConfidenceLevel.MEDIUM: "Medium confidence — proceeding with caution",
            ConfidenceLevel.LOW: "Low confidence — escalating to human review",
        }[level],
    }

"""
Security Check — Rate limiting, request sanitization, API key validation.
Applied at the Input Layer before validation and routing.
"""

import os
import time
import re
from collections import defaultdict
from typing import Tuple

# Rate limit: max requests per window
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# Simple in-memory rate limit store {user_id: [(timestamp, count)]}
_rate_store: dict = defaultdict(list)

# Patterns to detect prompt injection and malicious input
_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"system prompt",
    r"<\|.*?\|>",           # special tokens
    r"\\x[0-9a-fA-F]{2}",  # hex escape sequences
    r"javascript:",
    r"<script",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def check_rate_limit(user_id: str) -> Tuple[bool, str]:
    """
    Check if user has exceeded rate limit.

    Returns:
        (allowed: bool, reason: str)
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_store[user_id] = [t for t in _rate_store[user_id] if t > window_start]

    count = len(_rate_store[user_id])
    if count >= RATE_LIMIT_MAX:
        return False, f"Rate limit exceeded: {count}/{RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s"

    _rate_store[user_id].append(now)
    return True, ""


def check_injection(text: str) -> Tuple[bool, str]:
    """
    Detect prompt injection and malicious patterns.

    Returns:
        (safe: bool, reason: str)
    """
    if _INJECTION_RE.search(text):
        return False, "Potential prompt injection detected"
    return True, ""


def sanitize_input(text: str) -> str:
    """Strip control characters and normalize whitespace."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def security_check(user_id: str, message: str) -> Tuple[bool, str]:
    """
    Full security check pipeline:
    1. Rate limiting
    2. Prompt injection detection
    3. Input sanitization

    Returns:
        (passed: bool, reason: str)
    """
    # Rate limit
    allowed, reason = check_rate_limit(user_id)
    if not allowed:
        return False, reason

    # Injection check
    safe, reason = check_injection(message)
    if not safe:
        return False, reason

    return True, ""


def get_rate_limit_stats(user_id: str) -> dict:
    """Return current rate limit usage for a user."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    recent = [t for t in _rate_store.get(user_id, []) if t > window_start]
    return {
        "user_id": user_id,
        "requests_in_window": len(recent),
        "limit": RATE_LIMIT_MAX,
        "window_seconds": RATE_LIMIT_WINDOW,
        "remaining": max(0, RATE_LIMIT_MAX - len(recent)),
    }

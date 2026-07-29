"""
Security Check — Rate limiting, request sanitization, API key validation.
Applied at the Input Layer before validation and routing.
"""

import hmac
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

from core import config

# Patterns to detect prompt injection and malicious input
_INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?previous instructions",
    r"ignore (?:all|any) (?:prior |preceding )?instructions",
    r"disregard (?:all |any |the )?(?:previous|prior) instructions",
    r"reveal (?:your |the )?system prompt",
    r"repeat (?:your |the )?system prompt",
    r"<\|.*?\|>",           # special tokens
    r"\\x[0-9a-fA-F]{2}",  # hex escape sequences
    r"javascript:",
    r"<\s*script",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


class _RateLimiter:
    """
    Sliding-window rate limiter with bounded memory.

    The previous implementation kept a `defaultdict` entry for every user id it
    ever saw and never removed empty ones, so the map grew without bound —
    a slow leak in any long-running process, and a trivial memory-exhaustion
    vector given that `user_id` is caller-supplied.
    """

    # Prune idle users once the map exceeds this, so a burst of unique ids
    # can't pin memory between sweeps.
    _SWEEP_THRESHOLD = 10_000

    def __init__(self) -> None:
        self._hits: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - config.RATE_LIMIT_WINDOW
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]

    def check(self, user_id: str) -> Tuple[bool, str]:
        now = time.time()
        window_start = now - config.RATE_LIMIT_WINDOW

        with self._lock:
            if len(self._hits) > self._SWEEP_THRESHOLD:
                self._prune(now)

            recent = [t for t in self._hits.get(user_id, []) if t > window_start]

            if len(recent) >= config.RATE_LIMIT_MAX:
                self._hits[user_id] = recent
                return False, (
                    f"Rate limit exceeded: {len(recent)}/{config.RATE_LIMIT_MAX} "
                    f"requests per {config.RATE_LIMIT_WINDOW}s"
                )

            recent.append(now)
            self._hits[user_id] = recent
            return True, ""

    def stats(self, user_id: str) -> dict:
        now = time.time()
        window_start = now - config.RATE_LIMIT_WINDOW
        with self._lock:
            recent = [t for t in self._hits.get(user_id, []) if t > window_start]
        return {
            "user_id": user_id,
            "requests_in_window": len(recent),
            "limit": config.RATE_LIMIT_MAX,
            "window_seconds": config.RATE_LIMIT_WINDOW,
            "remaining": max(0, config.RATE_LIMIT_MAX - len(recent)),
        }

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    @property
    def tracked_users(self) -> int:
        with self._lock:
            return len(self._hits)


_limiter = _RateLimiter()


def check_rate_limit(user_id: str) -> Tuple[bool, str]:
    """
    Check if user has exceeded rate limit.

    Returns:
        (allowed: bool, reason: str)
    """
    return _limiter.check(user_id)


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
    text = _CONTROL_CHARS.sub("", text or "")
    return _WHITESPACE.sub(" ", text).strip()


def security_check(user_id: str, message: str) -> Tuple[bool, str]:
    """
    Full security check pipeline:
    1. Rate limiting
    2. Prompt injection detection

    Returns:
        (passed: bool, reason: str)
    """
    allowed, reason = check_rate_limit(user_id)
    if not allowed:
        return False, reason

    safe, reason = check_injection(message)
    if not safe:
        return False, reason

    return True, ""


def get_rate_limit_stats(user_id: str) -> dict:
    """Return current rate limit usage for a user."""
    return _limiter.stats(user_id)


def reset_rate_limits() -> None:
    """Clear all rate-limit state. Used by tests."""
    _limiter.reset()


def verify_admin_key(provided: Optional[str]) -> bool:
    """
    Constant-time comparison against the configured admin key.

    Returns True when no key is configured — development convenience. The
    startup check refuses to run in production without one.
    """
    expected = config.ADMIN_API_KEY
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)


def verify_webhook_signature(
    secret: Optional[str],
    payload: bytes,
    signature: Optional[str],
) -> bool:
    """Verify an HMAC-SHA256 webhook signature in constant time."""
    import hashlib

    if not secret:
        return True  # Verification disabled — no secret configured.
    if not signature:
        return False

    # Accept both bare hex and the common "sha256=<hex>" form.
    if signature.startswith("sha256="):
        signature = signature[len("sha256="):]

    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip())

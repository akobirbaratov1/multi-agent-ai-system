"""
Anthropic client access and response parsing.

Two problems this module exists to solve:

1. Constructing `Anthropic()` at import time couples module import to the
   presence of an API key, so a missing key breaks `import api.main` rather
   than the one request that needs the model. The client is built lazily and
   cached instead.
2. Every agent parsed the model's JSON with `json.loads` + bare `result["key"]`
   access, so a well-formed-but-incomplete response raised `KeyError` and
   surfaced as a 500. `parse_json_response` merges into caller-supplied
   defaults, so a partial response degrades instead of failing.
"""

import json
import re
from typing import Any, Dict, Optional

from anthropic import Anthropic, APIError, APIStatusError

from core import config
from core.logging_config import get_logger

logger = get_logger(__name__)

_client: Optional[Anthropic] = None


class LLMUnavailable(RuntimeError):
    """Raised when the model cannot be reached or is not configured."""


def get_client() -> Anthropic:
    """Return the shared Anthropic client, constructing it on first use."""
    global _client

    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not configured. Set it, or run with "
                "DEMO_MODE=true to use built-in mock responses."
            )
        # The SDK retries 429/5xx/connection errors with exponential backoff.
        _client = Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.ANTHROPIC_TIMEOUT,
            max_retries=config.ANTHROPIC_MAX_RETRIES,
        )

    return _client


def reset_client() -> None:
    """Drop the cached client. Used by tests after changing configuration."""
    global _client
    _client = None


def complete(
    system: str,
    messages: list,
    model: Optional[str] = None,
    max_tokens: int = 1024,
) -> str:
    """
    Send a Messages API request and return the concatenated text content.

    Raises `LLMUnavailable` for any API-level failure so callers have one
    exception type to handle rather than the full SDK hierarchy.
    """
    client = get_client()
    try:
        response = client.messages.create(
            model=model or config.AGENT_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
    except APIStatusError as exc:
        logger.error(
            "Anthropic API error",
            extra={"status_code": exc.status_code, "error_type": getattr(exc, "type", None)},
        )
        raise LLMUnavailable(f"Model request failed with status {exc.status_code}") from exc
    except APIError as exc:
        logger.error("Anthropic API error", extra={"error": str(exc)})
        raise LLMUnavailable("Model request failed") from exc

    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(text: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a JSON object out of a model response, falling back safely.

    The model is asked for bare JSON but sometimes wraps it in prose or a
    ```json fence. We try a strict parse first, then the first balanced-looking
    object in the text. Whatever we recover is layered over `defaults`, so
    every key the caller expects is guaranteed to be present with a
    correctly-typed value.
    """
    result = dict(defaults)

    payload: Any = None
    candidate = text.strip()

    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        match = _JSON_BLOCK.search(candidate)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        # No JSON at all — treat the whole reply as the user-facing text so the
        # caller still has something useful to return.
        if candidate and "response" in result:
            result["response"] = candidate
        return result

    for key, default_value in defaults.items():
        if key not in payload:
            continue
        value = payload[key]
        # Guard against the model returning the right key with the wrong shape.
        if default_value is not None and not isinstance(value, type(default_value)):
            if isinstance(default_value, float) and isinstance(value, int):
                value = float(value)
            elif isinstance(default_value, int) and isinstance(value, float):
                value = int(value)
            elif isinstance(default_value, str):
                value = str(value)
            else:
                continue
        result[key] = value

    return result

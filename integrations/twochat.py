"""
2Chat Integration — WhatsApp/SMS messaging via 2chat.co
Routes incoming messages through the same orchestrator pipeline.

Docs: https://docs.2chat.co
Webhook: POST /webhook/2chat
"""

import re
import threading
from collections import OrderedDict
from typing import List, Optional

import httpx

from core import config
from core.logging_config import get_logger

logger = get_logger(__name__)

MAX_SESSIONS = 5_000
MAX_TURNS = 10
MAX_REPLY_CHARS = 4_000

# E.164, which is what 2Chat delivers. Anything else is not a number we can
# route a reply to, and it becomes a `user_id` in rate limiting and CRM records.
_PHONE_RE = re.compile(r"^\+[1-9]\d{6,14}$")


class _SessionStore:
    """
    Bounded per-phone conversation history.

    A plain dict grew one entry per phone number forever — unbounded memory in
    a process that is expected to stay up for weeks. This evicts the
    least-recently-used conversation once the cap is reached.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS, max_turns: int = MAX_TURNS):
        self._data: "OrderedDict[str, List[dict]]" = OrderedDict()
        self._max_sessions = max_sessions
        self._max_turns = max_turns
        self._lock = threading.Lock()

    def get(self, phone: str) -> List[dict]:
        with self._lock:
            history = self._data.get(phone)
            if history is None:
                return []
            self._data.move_to_end(phone)
            return list(history)

    def append(self, phone: str, role: str, content: str) -> None:
        with self._lock:
            history = self._data.get(phone, [])
            history.append({"role": role, "content": content})
            self._data[phone] = history[-self._max_turns:]
            self._data.move_to_end(phone)
            while len(self._data) > self._max_sessions:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_sessions = _SessionStore()


def parse_incoming(payload: dict) -> Optional[dict]:
    """
    Parse an incoming 2Chat webhook payload.

    Returns normalized message dict or None if not a valid user message.

    2Chat webhook payload structure:
    {
        "event": "message.received",
        "data": {
            "id": "msg_id",
            "from": "+1234567890",
            "text": "Hello",
            "channel_id": "channel_id",
            "timestamp": 1234567890
        }
    }
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("event") != "message.received":
        return None

    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    phone = str(data.get("from") or "").strip()
    text = str(data.get("text") or "").strip()

    if not _PHONE_RE.match(phone) or not text:
        return None

    return {
        "phone": phone,
        "text": text[:config.MAX_MESSAGE_LENGTH],
        "channel_id": str(data.get("channel_id") or config.TWOCHAT_CHANNEL_ID or ""),
        "message_id": str(data.get("id") or ""),
    }


async def send_message(to: str, text: str, channel_id: Optional[str] = None) -> dict:
    """
    Send a WhatsApp/SMS message via 2Chat API.

    Args:
        to: Recipient phone number (E.164 format: +1234567890)
        text: Message text
        channel_id: 2Chat channel ID (uses env var if not provided)

    Returns:
        API response dict. Delivery failures are reported, never raised — the
        caller is a webhook handler that must still return a response.
    """
    if not config.TWOCHAT_API_KEY:
        logger.warning("TWOCHAT_API_KEY not set — message not sent (mock mode)")
        return {"status": "mock", "to": to, "text": text}

    channel = channel_id or config.TWOCHAT_CHANNEL_ID

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{config.TWOCHAT_API_URL}/messages/send-text",
                headers={
                    "Authorization": f"Bearer {config.TWOCHAT_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"to": to, "text": text[:MAX_REPLY_CHARS], "channel_id": channel},
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.error("2Chat delivery failed", extra={"error": str(exc)})
        return {"status": "delivery_failed", "to": to}


async def handle_incoming(payload: dict) -> dict:
    """
    Full pipeline: parse incoming → orchestrator → send reply.

    Args:
        payload: Raw 2Chat webhook payload

    Returns:
        Processing result dict
    """
    import asyncio

    from agents.orchestrator import process_message

    message = parse_incoming(payload)
    if not message:
        return {"status": "ignored", "reason": "not a user message"}

    phone = message["phone"]
    text = message["text"]
    history = _sessions.get(phone)
    _sessions.append(phone, "user", text)

    try:
        # `process_message` is synchronous and calls the model; running it
        # inline would block the event loop for the whole request.
        result = await asyncio.to_thread(
            process_message,
            user_id=phone,
            message=text,
            interface="2chat",
            conversation_history=history,
        )
    except Exception:
        logger.exception("2Chat processing error", extra={"phone": phone})
        await send_message(
            to=phone,
            text="Something went wrong. Please try again shortly.",
            channel_id=message["channel_id"],
        )
        return {"status": "error"}

    reply = result.get("response") or "I could not process your request."
    agent = str(result.get("agent") or "unknown").upper()

    _sessions.append(phone, "assistant", reply)

    await send_message(
        to=phone,
        text=f"{reply}\n\n_{agent} Agent_",
        channel_id=message["channel_id"],
    )

    return {
        "status": "processed",
        "phone": phone,
        "agent": result.get("agent"),
        "intent": result.get("intent"),
        "confidence": result.get("confidence"),
        "rag_used": result.get("rag_used"),
    }


def get_status() -> dict:
    """Return 2Chat integration status."""
    return {
        "configured": bool(config.TWOCHAT_API_KEY and config.TWOCHAT_CHANNEL_ID),
        "channel_id": config.TWOCHAT_CHANNEL_ID or "not set",
        "webhook_signature_verification": bool(config.TWOCHAT_WEBHOOK_SECRET),
        "active_sessions": len(_sessions),
    }

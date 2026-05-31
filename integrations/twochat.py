"""
2Chat Integration — WhatsApp/SMS messaging via 2chat.co
Routes incoming messages through the same orchestrator pipeline.

Docs: https://docs.2chat.co
Webhook: POST /webhook/2chat
"""

import os
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TWOCHAT_API_KEY = os.getenv("TWOCHAT_API_KEY", "")
TWOCHAT_CHANNEL_ID = os.getenv("TWOCHAT_CHANNEL_ID", "")
TWOCHAT_API_URL = "https://api.2chat.co/v1"

# Per-user session history (in-memory)
_sessions: dict = {}


def _get_history(phone: str) -> list:
    return _sessions.get(phone, [])


def _update_history(phone: str, role: str, content: str):
    if phone not in _sessions:
        _sessions[phone] = []
    _sessions[phone].append({"role": role, "content": content})
    _sessions[phone] = _sessions[phone][-10:]


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
    event = payload.get("event", "")
    if event != "message.received":
        return None

    data = payload.get("data", {})
    phone = data.get("from", "")
    text = data.get("text", "").strip()

    if not phone or not text:
        return None

    return {
        "phone": phone,
        "text": text,
        "channel_id": data.get("channel_id", TWOCHAT_CHANNEL_ID),
        "message_id": data.get("id", ""),
    }


async def send_message(to: str, text: str, channel_id: str = None) -> dict:
    """
    Send a WhatsApp/SMS message via 2Chat API.

    Args:
        to: Recipient phone number (E.164 format: +1234567890)
        text: Message text
        channel_id: 2Chat channel ID (uses env var if not provided)

    Returns:
        API response dict
    """
    if not TWOCHAT_API_KEY:
        logger.warning("TWOCHAT_API_KEY not set — message not sent (mock mode)")
        return {"status": "mock", "to": to, "text": text}

    channel = channel_id or TWOCHAT_CHANNEL_ID

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TWOCHAT_API_URL}/messages/send-text",
            headers={
                "Authorization": f"Bearer {TWOCHAT_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "to": to,
                "text": text,
                "channel_id": channel,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()


async def handle_incoming(payload: dict) -> dict:
    """
    Full pipeline: parse incoming → orchestrator → send reply.

    Args:
        payload: Raw 2Chat webhook payload

    Returns:
        Processing result dict
    """
    from agents.orchestrator import process_message

    message = parse_incoming(payload)
    if not message:
        return {"status": "ignored", "reason": "not a user message"}

    phone = message["phone"]
    text = message["text"]
    history = _get_history(phone)
    _update_history(phone, "user", text)

    try:
        result = process_message(
            user_id=phone,
            message=text,
            interface="2chat",
            conversation_history=history,
        )

        reply = result.get("response", "I could not process your request.")
        agent = result.get("agent", "unknown").upper()
        reply_with_meta = f"{reply}\n\n_{agent} Agent_"

        _update_history(phone, "assistant", reply)

        await send_message(
            to=phone,
            text=reply_with_meta,
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

    except Exception as e:
        logger.error(f"2Chat processing error for {phone}: {e}")

        await send_message(
            to=phone,
            text="Something went wrong. Please try again shortly.",
            channel_id=message["channel_id"],
        )

        return {"status": "error", "error": str(e)}


def get_status() -> dict:
    """Return 2Chat integration status."""
    return {
        "configured": bool(TWOCHAT_API_KEY and TWOCHAT_CHANNEL_ID),
        "channel_id": TWOCHAT_CHANNEL_ID or "not set",
        "active_sessions": len(_sessions),
    }

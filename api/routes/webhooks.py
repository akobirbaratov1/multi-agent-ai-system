"""
Webhook routes — 2Chat, Telegram webhook mode, and future integrations.
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Any

router = APIRouter(prefix="/webhook", tags=["Webhooks"])


@router.post("/2chat")
async def twochat_webhook(request: Request):
    """
    2Chat incoming message webhook.

    Configure in 2Chat dashboard:
    Webhook URL: https://your-domain.com/webhook/2chat
    Events: message.received
    """
    try:
        payload: Any = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    from integrations.twochat import handle_incoming
    result = await handle_incoming(payload)
    return result


@router.get("/2chat/status")
async def twochat_status():
    """2Chat integration status and configuration check."""
    from integrations.twochat import get_status
    return get_status()


@router.get("/2chat/test")
async def twochat_test():
    """
    Test the 2Chat pipeline with a sample message (no real API call).
    Useful for verifying the orchestrator routing works end-to-end.
    """
    sample_payload = {
        "event": "message.received",
        "data": {
            "id": "test_msg_001",
            "from": "+10000000000",
            "text": "What are your pricing plans?",
            "channel_id": "test_channel",
            "timestamp": 1700000000,
        },
    }
    from integrations.twochat import handle_incoming
    return await handle_incoming(sample_payload)

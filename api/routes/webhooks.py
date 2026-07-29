"""
Webhook routes — 2Chat, Telegram webhook mode, and future integrations.
"""

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from api.deps import require_admin
from core import config
from core.logging_config import get_logger
from core.security import verify_webhook_signature

router = APIRouter(prefix="/webhook", tags=["Webhooks"])
logger = get_logger(__name__)

MAX_WEBHOOK_BYTES = 256 * 1024


@router.post("/2chat")
async def twochat_webhook(
    request: Request,
    x_signature: Optional[str] = Header(default=None, alias="X-Signature"),
):
    """
    2Chat incoming message webhook.

    Configure in 2Chat dashboard:
    Webhook URL: https://your-domain.com/webhook/2chat
    Events: message.received

    The body is authenticated with an HMAC-SHA256 signature over the raw
    payload when TWOCHAT_WEBHOOK_SECRET is set. Without it the endpoint is a
    public, unauthenticated path into the orchestrator — anyone could drive
    model spend and send messages as the bot.
    """
    raw = await request.body()

    if len(raw) > MAX_WEBHOOK_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Payload too large",
        )

    if not verify_webhook_signature(config.TWOCHAT_WEBHOOK_SECRET, raw, x_signature):
        logger.warning("Rejected 2Chat webhook with an invalid signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature"
        )

    if not config.TWOCHAT_WEBHOOK_SECRET:
        logger.warning(
            "TWOCHAT_WEBHOOK_SECRET is not set — the 2Chat webhook is unauthenticated"
        )

    try:
        payload: Any = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload"
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must be a JSON object",
        )

    from integrations.twochat import handle_incoming

    return await handle_incoming(payload)


@router.get("/2chat/status")
async def twochat_status():
    """2Chat integration status and configuration check."""
    from integrations.twochat import get_status

    return get_status()


@router.post("/2chat/test", dependencies=[Depends(require_admin)])
async def twochat_test():
    """
    Test the 2Chat pipeline with a sample message (no real API call).

    Admin-only and POST rather than GET: it drives a full orchestrator run,
    which costs model tokens and mutates CRM state, so it must not be
    reachable by an unauthenticated crawler following links.
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

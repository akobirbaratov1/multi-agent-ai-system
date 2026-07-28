"""
Shared FastAPI dependencies.
"""

from typing import Optional

from fastapi import Header, HTTPException, status

from core import config
from core.security import verify_admin_key

API_KEY_HEADER = "X-API-Key"


async def require_admin(
    x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """
    Guard state-changing and operator endpoints.

    Writing to the knowledge base steers what every agent says, and the
    operator dashboard exposes full conversation transcripts — both were
    reachable unauthenticated. When ADMIN_API_KEY is unset the guard is a
    no-op so local development stays frictionless; `config.validate()`
    refuses to start in production without it.
    """
    if verify_admin_key(x_api_key):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )


def admin_required_enabled() -> bool:
    """Whether admin authentication is actually enforced in this deployment."""
    return bool(config.ADMIN_API_KEY)

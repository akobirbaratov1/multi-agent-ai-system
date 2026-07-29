"""
Central configuration — single source of truth for env-driven settings.

Importing this module loads `.env` exactly once, so every entry point
(API, Telegram bot, CLI scripts, tests) sees the same configuration.
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env once, at first import. Values already present in the real
# environment win — that is what deployments (Docker, k8s) rely on.
load_dotenv(override=False)

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _csv(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ─── App ─────────────────────────────────────────────────
APP_ENV: str = os.getenv("APP_ENV", "development")
APP_PORT: int = _int("APP_PORT", 8000)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json" if APP_ENV == "production" else "text")
VERSION: str = "1.1.0"

IS_PRODUCTION: bool = APP_ENV.lower() in ("production", "prod")

# ─── Claude ──────────────────────────────────────────────
DEMO_MODE: bool = _bool("DEMO_MODE", False)
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY") or None

# Model IDs are configurable so a deployment can move models without a code change.
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "claude-sonnet-4-6")
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")

ANTHROPIC_TIMEOUT: float = _float("ANTHROPIC_TIMEOUT", 30.0)
ANTHROPIC_MAX_RETRIES: int = _int("ANTHROPIC_MAX_RETRIES", 3)

# ─── Data ────────────────────────────────────────────────
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "memory" / "data")))

# ─── Agent behaviour ─────────────────────────────────────
CONFIDENCE_THRESHOLD_HIGH: float = _float("CONFIDENCE_THRESHOLD_HIGH", 0.85)
CONFIDENCE_THRESHOLD_MEDIUM: float = _float("CONFIDENCE_THRESHOLD_MEDIUM", 0.60)
MAX_CONVERSATION_HISTORY: int = _int("MAX_CONVERSATION_HISTORY", 20)
MAX_MESSAGE_LENGTH: int = _int("MAX_MESSAGE_LENGTH", 5000)

# ─── Security ────────────────────────────────────────────
RATE_LIMIT_MAX: int = _int("RATE_LIMIT_MAX", 30)
RATE_LIMIT_WINDOW: int = _int("RATE_LIMIT_WINDOW", 60)

# Comma-separated list of allowed CORS origins. "*" is rejected in production.
CORS_ORIGINS: List[str] = _csv("CORS_ORIGINS", "*")

# Shared secret protecting write/admin endpoints (operator dashboard, KB writes).
# When unset in development the endpoints stay open; in production it is required.
ADMIN_API_KEY: Optional[str] = os.getenv("ADMIN_API_KEY") or None

# ─── Integrations ────────────────────────────────────────
TWOCHAT_API_KEY: Optional[str] = os.getenv("TWOCHAT_API_KEY") or None
TWOCHAT_CHANNEL_ID: Optional[str] = os.getenv("TWOCHAT_CHANNEL_ID") or None
TWOCHAT_API_URL: str = os.getenv("TWOCHAT_API_URL", "https://api.2chat.co/v1")
TWOCHAT_WEBHOOK_SECRET: Optional[str] = os.getenv("TWOCHAT_WEBHOOK_SECRET") or None

TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN") or None

LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY") or None
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "multi-agent-ai-system")


def data_path(*parts: str) -> Path:
    """Resolve a path inside the data directory, creating parents as needed."""
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate() -> List[str]:
    """
    Check configuration coherence. Returns a list of human-readable problems.

    Called at API startup: fatal in production, logged as warnings otherwise.
    """
    problems: List[str] = []

    if not DEMO_MODE and not ANTHROPIC_API_KEY:
        problems.append(
            "ANTHROPIC_API_KEY is not set and DEMO_MODE is false — "
            "agent calls will fail. Set the key or run with DEMO_MODE=true."
        )

    if IS_PRODUCTION:
        if "*" in CORS_ORIGINS:
            problems.append(
                "CORS_ORIGINS is '*' in production — set an explicit origin list."
            )
        if not ADMIN_API_KEY:
            problems.append(
                "ADMIN_API_KEY is not set in production — admin and knowledge-base "
                "write endpoints would be unauthenticated."
            )
        if DEMO_MODE:
            problems.append("DEMO_MODE is enabled in production.")

    return problems

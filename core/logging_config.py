"""
Structured logging with per-request correlation IDs.

In production the formatter emits one JSON object per line so log aggregators
(CloudWatch, Loki, Datadog) can index fields directly. In development it stays
human-readable.
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Optional

from core import config

# Correlation ID for the in-flight request, set by the API middleware and
# picked up automatically by every log record emitted while handling it.
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


def set_request_id(value: Optional[str]) -> None:
    _request_id.set(value)


def get_request_id() -> Optional[str]:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Promote any structured extras the caller attached.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging() -> None:
    """Install handlers on the root logger. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if config.LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s [req=%(request_id)s] %(message)s"
            )
        )
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)

    # Uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

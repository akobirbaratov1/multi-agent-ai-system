"""
Conversation Manager — Context & history management per session.
Persists conversation turns and provides context window management.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from core import config
from core.storage import read_json, write_json

CONV_DIR = config.DATA_DIR / "conversations"
MAX_HISTORY = config.MAX_CONVERSATION_HISTORY

# Session IDs reach this from request bodies and webhook payloads, and they are
# interpolated straight into a filename — anything but these characters could
# escape the conversations directory.
_SAFE_SESSION_ID = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(session_id: str) -> str:
    cleaned = _SAFE_SESSION_ID.sub("_", (session_id or "").strip())[:128]
    return cleaned or "unnamed"


class ConversationManager:
    """Manages per-session conversation history with persistence."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._path = self._dir() / f"{_safe_name(session_id)}.json"
        self.history: List[Dict] = self._load()

    @staticmethod
    def _dir() -> Path:
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        return CONV_DIR

    def _load(self) -> List[Dict]:
        history = read_json(self._path, default=[])
        return history if isinstance(history, list) else []

    def _save(self) -> None:
        write_json(self._path, self.history)

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Append a conversation turn."""
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            turn["metadata"] = metadata

        self.history.append(turn)

        # Keep within MAX_HISTORY
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-(MAX_HISTORY * 2):]

        self._save()

    def get_context_window(self, n: int = 6) -> List[Dict]:
        """Return last n turns as role/content pairs for LLM messages."""
        return [
            {"role": t["role"], "content": t["content"]}
            for t in self.history[-n:]
            if t.get("role") and t.get("content")
        ]

    def get_summary_context(self) -> str:
        """Return a plain-text summary of recent conversation for context injection."""
        recent = self.history[-10:]
        if not recent:
            return ""
        return "\n".join(
            f"{'User' if turn.get('role') == 'user' else 'Assistant'}: "
            f"{str(turn.get('content', ''))[:200]}"
            for turn in recent
        )

    def clear(self):
        self.history = []
        if self._path.exists():
            self._path.unlink(missing_ok=True)

    @staticmethod
    def list_sessions() -> List[str]:
        return [p.stem for p in ConversationManager._dir().glob("*.json")]

    @staticmethod
    def get_session_count() -> int:
        return len(list(ConversationManager._dir().glob("*.json")))

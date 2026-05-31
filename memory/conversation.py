"""
Conversation Manager — Context & history management per session.
Persists conversation turns and provides context window management.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

CONV_DIR = Path("memory/data/conversations")
MAX_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))


class ConversationManager:
    """Manages per-session conversation history with persistence."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._path = CONV_DIR / f"{session_id}.json"
        self.history: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Append a conversation turn."""
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
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
        recent = self.history[-n:]
        return [{"role": t["role"], "content": t["content"]} for t in recent]

    def get_summary_context(self) -> str:
        """Return a plain-text summary of recent conversation for context injection."""
        recent = self.history[-10:]
        if not recent:
            return ""
        lines = []
        for turn in recent:
            label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{label}: {turn['content'][:200]}")
        return "\n".join(lines)

    def clear(self):
        self.history = []
        if self._path.exists():
            self._path.unlink()

    @staticmethod
    def list_sessions() -> List[str]:
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        return [p.stem for p in CONV_DIR.glob("*.json")]

    @staticmethod
    def get_session_count() -> int:
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        return len(list(CONV_DIR.glob("*.json")))

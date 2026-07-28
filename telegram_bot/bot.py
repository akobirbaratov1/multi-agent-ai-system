"""
Telegram Bot Interface — Multi-Agent AI System
Routes Telegram messages through the same orchestrator as Web/API.

Run: python -m telegram_bot.bot
Requires: TELEGRAM_BOT_TOKEN in .env
"""

import asyncio
import threading
from collections import OrderedDict
from typing import List

from core import config
from core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

MAX_SESSIONS = 10_000
MAX_TURNS = 10
# Telegram rejects messages over 4096 characters outright.
TELEGRAM_MAX_CHARS = 4000


class _SessionStore:
    """Bounded per-user history — a plain dict never released idle chats."""

    def __init__(self, max_sessions: int = MAX_SESSIONS, max_turns: int = MAX_TURNS):
        self._data: "OrderedDict[str, List[dict]]" = OrderedDict()
        self._max_sessions = max_sessions
        self._max_turns = max_turns
        self._lock = threading.Lock()

    def get(self, user_id: str) -> List[dict]:
        with self._lock:
            history = self._data.get(user_id)
            if history is None:
                return []
            self._data.move_to_end(user_id)
            return list(history)

    def append(self, user_id: str, role: str, content: str) -> None:
        with self._lock:
            history = self._data.get(user_id, [])
            history.append({"role": role, "content": content})
            self._data[user_id] = history[-self._max_turns:]
            self._data.move_to_end(user_id)
            while len(self._data) > self._max_sessions:
                self._data.popitem(last=False)

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._data.pop(user_id, None)


_user_sessions = _SessionStore()


def _format_response(result: dict) -> str:
    """Format agent response for Telegram."""
    response = result.get("response") or "I could not process your request."
    agent = str(result.get("agent") or "unknown").upper()
    confidence = result.get("confidence") or 0
    rag = " • RAG" if result.get("rag_used") else ""
    latency = result.get("latency_ms") or 0

    footer = f"\n\n[{agent}{rag} • {confidence:.0%} conf • {latency:.0f}ms]"
    body = response[:TELEGRAM_MAX_CHARS - len(footer)]
    return body + footer


async def start_command(update, context):
    await update.message.reply_text(
        "👋 *Welcome to Multi-Agent AI System!*\n\n"
        "I have 3 specialized agents:\n"
        "🎯 *Sales* — pricing, demos, plans\n"
        "🛟 *Support* — help, troubleshooting\n"
        "🔍 *Research* — information, analysis\n\n"
        "Just send me a message and I'll route it to the right agent.",
        parse_mode="Markdown",
    )


async def help_command(update, context):
    await update.message.reply_text(
        "💡 *Try asking:*\n"
        "• `What are your pricing plans?`\n"
        "• `I can't connect to the API`\n"
        "• `Tell me about multi-agent AI`\n"
        "• `I want to schedule a demo`\n\n"
        "/start — Welcome message\n"
        "/clear — Clear conversation history",
        parse_mode="Markdown",
    )


async def clear_command(update, context):
    _user_sessions.clear(str(update.effective_user.id))
    await update.message.reply_text("🗑 Conversation history cleared.")


async def _reply(update, text: str) -> None:
    """
    Send a reply, falling back to plain text when Markdown fails.

    Agent output contains model-authored Markdown — an unbalanced `*` or a
    stray `_` makes Telegram reject the whole message with a parse error, so
    the user saw nothing at all. Retry unformatted rather than dropping it.
    """
    from telegram.error import BadRequest

    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except BadRequest:
        logger.warning("Markdown parse failed; resending as plain text")
        await update.message.reply_text(text)


async def handle_message(update, context):
    from agents.orchestrator import process_message

    user_id = str(update.effective_user.id)
    message = (update.message.text or "").strip()
    if not message:
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    history = _user_sessions.get(user_id)
    _user_sessions.append(user_id, "user", message)

    try:
        # The orchestrator is synchronous; keep it off the event loop so one
        # slow model call doesn't stall every other chat.
        result = await asyncio.to_thread(
            process_message,
            user_id=user_id,
            message=message,
            interface="telegram",
            conversation_history=history,
        )
    except Exception:
        logger.exception("Error processing Telegram message")
        await update.message.reply_text("❌ An error occurred. Please try again.")
        return

    _user_sessions.append(user_id, "assistant", result.get("response", ""))

    response_text = _format_response(result)
    if result.get("requires_human"):
        response_text += "\n\n⚠️ This case has been escalated to a human agent."

    await _reply(update, response_text)


async def error_handler(update, context):
    logger.error("Telegram update caused an error", exc_info=context.error)


def run():
    """Start the Telegram bot (polling mode)."""
    if not config.TELEGRAM_BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN not set. Add it to your .env file.\n"
            "Get a token from @BotFather on Telegram."
        )

    try:
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError as exc:
        raise ImportError(
            "python-telegram-bot not installed. Run: pip install python-telegram-bot"
        ) from exc

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Telegram bot started. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run()

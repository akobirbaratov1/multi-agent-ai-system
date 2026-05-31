"""
Telegram Bot Interface — Multi-Agent AI System
Routes Telegram messages through the same orchestrator as Web/API.

Run: python -m telegram_bot.bot
Requires: TELEGRAM_BOT_TOKEN in .env
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Per-user conversation history (in-memory for demo)
_user_sessions: dict = {}


def _get_history(user_id: str) -> list:
    return _user_sessions.get(user_id, [])


def _update_history(user_id: str, role: str, content: str):
    if user_id not in _user_sessions:
        _user_sessions[user_id] = []
    _user_sessions[user_id].append({"role": role, "content": content})
    # Keep last 10 turns
    _user_sessions[user_id] = _user_sessions[user_id][-10:]


def _format_response(result: dict) -> str:
    """Format agent response for Telegram Markdown."""
    response = result.get("response", "I could not process your request.")
    agent = result.get("agent", "unknown").upper()
    confidence = result.get("confidence", 0)
    rag = " • RAG" if result.get("rag_used") else ""
    latency = result.get("latency_ms", 0)

    footer = f"\n\n`[{agent}{rag} • {confidence:.0%} conf • {latency:.0f}ms]`"
    return response + footer


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
    user_id = str(update.effective_user.id)
    _user_sessions.pop(user_id, None)
    await update.message.reply_text("🗑 Conversation history cleared.")


async def handle_message(update, context):
    from agents.orchestrator import process_message

    user_id = str(update.effective_user.id)
    message = update.message.text

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    history = _get_history(user_id)
    _update_history(user_id, "user", message)

    try:
        result = process_message(
            user_id=user_id,
            message=message,
            interface="telegram",
            conversation_history=history,
        )

        _update_history(user_id, "assistant", result.get("response", ""))

        response_text = _format_response(result)

        # Alert if human review needed
        if result.get("requires_human"):
            response_text += "\n\n⚠️ _This case has been escalated to a human agent._"

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        response_text = "❌ An error occurred. Please try again."

    await update.message.reply_text(response_text, parse_mode="Markdown")


async def error_handler(update, context):
    logger.error(f"Update {update} caused error: {context.error}")


def run():
    """Start the Telegram bot (polling mode)."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN not set. Add it to your .env file.\n"
            "Get a token from @BotFather on Telegram."
        )

    try:
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError:
        raise ImportError(
            "python-telegram-bot not installed. Run: pip install python-telegram-bot"
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Telegram bot started. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run()

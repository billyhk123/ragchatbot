"""Telegram bot integration using python-telegram-bot (webhook mode)."""

import os
import asyncio
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters,
)

from app.crypto import DEFAULT_COINS, format_price, get_price

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

_application: Application | None = None
_rag_answer_fn = None


def set_rag_answer(fn):
    """Inject the rag_answer callable so this module has no circular imports."""
    global _rag_answer_fn
    _rag_answer_fn = fn


async def _start_command(update: Update, context) -> None:
    keyboard = [[InlineKeyboardButton("Check Crypto Price", callback_data="menu:price")]]
    await update.message.reply_text(
        "Hi! I'm your RAG chatbot. Send me a message and I'll answer "
        "using my knowledge base.\n\n"
        "You can also check live crypto prices:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def _price_command(update: Update, context) -> None:
    """Show coin selection keyboard."""
    keyboard = [
        [InlineKeyboardButton(c.title(), callback_data=f"coin:{c}")]
        for c in DEFAULT_COINS
    ]
    await update.message.reply_text(
        "Choose a cryptocurrency:", reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def _button_callback(update: Update, context) -> None:
    """Handle inline-button presses."""
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data == "menu:price":
        keyboard = [
            [InlineKeyboardButton(c.title(), callback_data=f"coin:{c}")]
            for c in DEFAULT_COINS
        ]
        await query.edit_message_text(
            "Choose a cryptocurrency:", reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    if data.startswith("coin:"):
        slug = data.removeprefix("coin:")
        await query.edit_message_text(f"Fetching {slug.title()} price…")
        info = await asyncio.to_thread(get_price, slug)
        if info:
            await query.edit_message_text(format_price(info))
        else:
            await query.edit_message_text(f"Could not fetch price for {slug}.")


async def _handle_message(update: Update, context) -> None:
    if not update.message or not update.message.text:
        return

    user_id = f"tg_{update.message.from_user.id}"
    user_text = update.message.text.strip()
    if not user_text:
        return

    logger.info("[TG] %s: %s", user_id, user_text[:80])

    try:
        answer = await asyncio.to_thread(_rag_answer_fn, user_text, user_id)
        await update.message.reply_text(answer[:4096])
    except Exception:
        logger.exception("[TG] Error generating answer")
        await update.message.reply_text(
            "Sorry, something went wrong, please try again later."
        )


def get_application() -> Application:
    global _application
    if _application is None:
        if not TELEGRAM_BOT_TOKEN:
            raise RuntimeError("TELEGRAM_BOT_TOKEN env var is not set")
        _application = (
            Application.builder()
            .token(TELEGRAM_BOT_TOKEN)
            .updater(None)
            .build()
        )
        _application.add_handler(CommandHandler("start", _start_command))
        _application.add_handler(CommandHandler("price", _price_command))
        _application.add_handler(CallbackQueryHandler(_button_callback))
        _application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message)
        )
    return _application


async def init() -> None:
    """Call once on server startup to initialise the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("[TG] TELEGRAM_BOT_TOKEN not set – Telegram bot disabled")
        return
    tg_app = get_application()
    await tg_app.initialize()
    await tg_app.start()
    logger.info("[TG] Bot initialised")


async def register_webhook(base_url: str) -> dict:
    """Register the webhook URL with Telegram."""
    tg_app = get_application()
    webhook_url = f"{base_url.rstrip('/')}/telegram/webhook"
    await tg_app.bot.set_webhook(url=webhook_url)
    logger.info("[TG] Webhook registered: %s", webhook_url)
    return {"status": "ok", "webhook_url": webhook_url}


async def process_update(payload: dict) -> None:
    """Process an incoming Telegram webhook update."""
    tg_app = get_application()
    update = Update.de_json(payload, tg_app.bot)
    await tg_app.process_update(update)


async def shutdown() -> None:
    if _application is not None:
        await _application.stop()
        await _application.shutdown()
        logger.info("[TG] Bot shut down")

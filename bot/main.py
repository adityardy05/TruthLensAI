from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from backend.core.config import load_settings, validate_runtime_config
from telegram.ext import Application, CommandHandler, MessageHandler, filters

settings = load_settings()
validate_runtime_config(settings, mode="bot")

from bot.handlers import (
    about,
    check,
    forwarded_message,
    help_command,
    history,
    start,
    status,
    unknown_command,
    verify_photo,
    verify_text,
)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be configured.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("check", check))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(MessageHandler(filters.PHOTO, verify_photo))
    app.add_handler(MessageHandler(filters.FORWARDED & filters.TEXT, forwarded_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, verify_text))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    app.run_polling()


if __name__ == "__main__":
    main()

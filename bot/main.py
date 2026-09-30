from __future__ import annotations

import os

from dotenv import load_dotenv
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

load_dotenv()

from backend.core.config import load_settings, validate_runtime_config

settings = load_settings()
validate_runtime_config(settings, mode="bot")

from bot.handlers import (
    about,
    check,
    error_handler,
    help_command,
    start,
    status,
    unknown_command,
    verify_forwarded,
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
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("about", about))

    # Handle forwarded text before the generic text handler.
    app.add_handler(MessageHandler(filters.FORWARDED & filters.TEXT, verify_forwarded))
    app.add_handler(MessageHandler(filters.PHOTO, verify_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, verify_text))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    app.add_error_handler(error_handler)

    print("TruthLens Telegram bot started.")
    print("Backend:", os.getenv("TRUTHLENS_BACKEND_URL", "http://127.0.0.1:5000"))

    app.run_polling()


if __name__ == "__main__":
    main()

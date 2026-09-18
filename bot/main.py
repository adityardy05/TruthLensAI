from __future__ import annotations

import os

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from bot.handlers import start, verify_photo, verify_text


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be configured.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, verify_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, verify_text))
    app.run_polling()


if __name__ == "__main__":
    main()

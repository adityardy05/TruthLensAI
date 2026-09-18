from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from bot.backend_client import BackendClient


def _summary(result: dict) -> str:
    sources = result.get("evidence", [])[:3]
    links = "\n".join(f"• {item.get('source_domain', 'source')}: {item.get('url', '')}" for item in sources)
    return (f"Verdict: {result['verdict']} ({round(result['confidence'])}%)\n\n"
            f"{result['justification']}\n\n"
            f"Recommendation: {result['recommendation']}"
            + (f"\n\nTop evidence:\n{links}" if links else ""))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("Send a claim as text or a photo containing a claim to verify it.")


async def verify_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        result = BackendClient().verify_text(update.effective_message.text)
        await update.effective_message.reply_text(_summary(result), disable_web_page_preview=True)
    except RuntimeError as error:
        await update.effective_message.reply_text(str(error))


async def verify_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        photo = update.effective_message.photo[-1]
        image = await context.bot.get_file(photo.file_id)
        payload = bytes(await image.download_as_bytearray())
        result = BackendClient().verify_image(payload)
        await update.effective_message.reply_text(_summary(result), disable_web_page_preview=True)
    except RuntimeError as error:
        await update.effective_message.reply_text(str(error))

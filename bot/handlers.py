from __future__ import annotations

import os
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from bot.backend_client import BackendClient


def _client() -> BackendClient:
    return BackendClient()


def _summary(result: dict[str, Any]) -> str:
    verdict = result.get("verdict", "UNVERIFIABLE")
    confidence = result.get("confidence", 0)
    justification = result.get(
        "justification",
        "No justification was returned by the verification service.",
    )
    recommendation = result.get("recommendation", "")
    evidence = result.get("evidence") or []

    try:
        confidence_text = f"{float(confidence):.0f}%"
    except (TypeError, ValueError):
        confidence_text = str(confidence)

    lines = [
        f"Verdict: {verdict} ({confidence_text})",
        "",
        str(justification),
    ]

    if recommendation:
        lines.extend(["", f"Recommendation: {recommendation}"])

    links = []
    for item in evidence[:3]:
        if not isinstance(item, dict):
            continue
        domain = item.get("source_domain") or item.get("domain") or "source"
        url = item.get("url")
        if url:
            links.append(f"• {domain}: {url}")

    if links:
        lines.extend(["", "Top evidence:", *links])

    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Welcome to TruthLens.\n\n"
        "Send a claim as text or a photo containing a claim to verify it.\n"
        "Use /help to see available commands."
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await update.effective_message.reply_text(
        "TruthLens Commands\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/check <claim> - Verify a specific claim\n"
        "/status - Check backend service status\n"
        "/about - About TruthLens\n\n"
        "You can also send a claim directly as text or send a photo containing a claim."
    )


async def about(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await update.effective_message.reply_text(
        "TruthLens\n\n"
        "A multilingual multi-agent fake-news detection system with Telegram integration.\n\n"
        "Pipeline: Telegram → Verification API → claim processing → "
        "FAISS/BM25 evidence retrieval → source credibility and recency scoring → "
        "LangGraph multi-agent reasoning → final verification result.\n\n"
        "The Telegram bot supports text and image-based verification."
    )


async def status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    backend_url = os.getenv(
        "TRUTHLENS_BACKEND_URL",
        "http://127.0.0.1:5000",
    )

    try:
        health = _client().health()
        state = health.get("status", "unknown")
        await update.effective_message.reply_text(
            "TruthLens Status\n\n"
            f"Backend: {state}\n"
            f"API: {backend_url}\n"
            "Verification endpoints: available"
        )
    except RuntimeError as error:
        await update.effective_message.reply_text(
            "TruthLens Status\n\n"
            f"Backend: unavailable\n"
            f"API: {backend_url}\n\n"
            f"Details: {error}"
        )


async def check(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    claim = " ".join(context.args).strip()

    if not claim:
        await update.effective_message.reply_text(
            "Usage: /check <claim>\n\n"
            "Example: /check Telangana was formed on 2 June 2014."
        )
        return

    await _verify_claim(update, claim)


async def verify_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    claim = (message.text or "").strip()

    if not claim:
        await message.reply_text("Please send a claim to verify.")
        return

    await _verify_claim(update, claim)


async def verify_forwarded(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    claim = (message.text or message.caption or "").strip()

    if not claim:
        await message.reply_text(
            "I received the forwarded message, but it does not contain text to verify."
        )
        return

    await _verify_claim(update, claim)


async def verify_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message

    try:
        photo = message.photo[-1]
        telegram_file = await context.bot.get_file(photo.file_id)
        payload = bytes(await telegram_file.download_as_bytearray())

        await message.reply_text("Analyzing the image...")

        result = _client().verify_image(payload)
        await message.reply_text(
            _summary(result),
            disable_web_page_preview=True,
        )
    except RuntimeError as error:
        await message.reply_text(str(error))
    except Exception:
        await message.reply_text(
            "I couldn't process that image. Please try again with a clearer image."
        )


async def _verify_claim(update: Update, claim: str) -> None:
    message = update.effective_message

    try:
        await message.reply_text("Verifying the claim...")
        result = _client().verify_text(claim)
        await message.reply_text(
            _summary(result),
            disable_web_page_preview=True,
        )
    except RuntimeError as error:
        await message.reply_text(str(error))
    except Exception:
        await message.reply_text(
            "Something went wrong while verifying the claim. Please try again."
        )


async def unknown_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await update.effective_message.reply_text(
        "Unknown command. Use /help to see the available commands."
    )


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    # Keep Telegram's polling loop alive if a handler raises unexpectedly.
    print(f"TruthLens bot error: {context.error!r}")

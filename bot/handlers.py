from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from bot.backend_client import BackendClient

MAX_CLAIM_LENGTH = 2000
MAX_HISTORY_ITEMS = 10
_history: dict[int, list[dict[str, str]]] = {}


def _summary(result: dict) -> str:
    sources = result.get("evidence", [])[:3]
    links = "\n".join(
        f"- {item.get('source_domain', 'source')}: {item.get('url', '')}"
        for item in sources
    )
    claim = result.get("original_claim") or result.get("normalized_claim", "")
    confidence = round(float(result.get("confidence", 0)))
    lines = [
        f"Claim: {claim}",
        f"Verdict: {result.get('verdict', 'UNVERIFIABLE')} ({confidence}%)",
        f"\nAnalysis: {result.get('justification', '')}",
    ]

    patterns = result.get("patterns_detected", [])
    if patterns:
        lines.append("\nPatterns detected:\n- " + "\n- ".join(str(pattern) for pattern in patterns))

    stance_breakdown = result.get("stance_breakdown", {})
    if stance_breakdown:
        stance = ", ".join(f"{name}: {value}" for name, value in stance_breakdown.items())
        lines.append(f"\nStance breakdown: {stance}")

    if links:
        lines.append(f"\nTop evidence:\n{links}")

    rounds = result.get("rounds_executed")
    if rounds is not None:
        lines.append(f"\nAnalysed in {rounds} round(s)")

    lines.append(f"\nRecommendation: {result.get('recommendation', '')}")
    return "\n".join(lines)


def _print_verification_result(update: Update, claim: str, result: dict) -> None:
    user = update.effective_user
    user_id = user.id if user else "unknown"
    confidence = round(float(result.get("confidence", 0)))

    print("\n" + "=" * 60)
    print("[TruthLens Bot] VERIFICATION REQUEST")
    print(f"User ID: {user_id}")
    print(f"Claim: {claim}")
    print(f"Final verdict: {result.get('verdict', 'UNVERIFIABLE')} ({confidence}%)")
    print(f"Final output: {result.get('justification', '')}")
    print(f"Recommendation: {result.get('recommendation', '')}")
    print("=" * 60)


def _record_history(update: Update, result: dict) -> None:
    user = update.effective_user
    if not user:
        return
    user_history = _history.setdefault(user.id, [])
    user_history.insert(0, {
        "claim": result.get("original_claim") or result.get("normalized_claim", ""),
        "verdict": str(result.get("verdict", "UNVERIFIABLE")),
        "confidence": str(round(float(result.get("confidence", 0)))),
    })
    del user_history[MAX_HISTORY_ITEMS:]


async def _verify_claim(update: Update, claim: str) -> None:
    claim = claim.strip()
    if not claim:
        await update.effective_message.reply_text("Usage: /check <claim>")
        return
    if len(claim) > MAX_CLAIM_LENGTH:
        await update.effective_message.reply_text(
            f"Claim is too long. Please keep it under {MAX_CLAIM_LENGTH} characters."
        )
        return

    try:
        result = BackendClient().verify_text(claim)
        _record_history(update, result)
        _print_verification_result(update, claim, result)
        await update.effective_message.reply_text(_summary(result), disable_web_page_preview=True)
    except RuntimeError as error:
        await update.effective_message.reply_text(str(error))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "🤖 Welcome to TruthLens AI!\n\n"
        "AI-powered claim verification using OCR, evidence retrieval, and "
        "multi-agent analysis.\n\n"
        "You can send a claim, forwarded message, or screenshot.\n\n"
        "Use /help to see available commands."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "📖 TruthLens AI - Help\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/check <claim> - Verify a specific claim\n"
        "/history - Show your recent verifications\n"
        "/status - Check backend and retrieval status\n"
        "/about - About TruthLens AI\n\n"
        "You can also send normal text, forwarded messages, or screenshots."
    )


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "🔎 About TruthLens AI\n\n"
        "TruthLens is an evidence-grounded fake-news detection system.\n\n"
        "It combines Telegram, OCR, language detection, translation, FAISS, "
        "Tavily, LangGraph, and multi-agent analysis to produce a verdict "
        "with confidence, explanation, and sources."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        health = BackendClient().health()
        backend_line = "🟢 Backend: ONLINE"
        faiss_line = "🟢 FAISS: LOADED" if health.get("faiss_loaded") else "🟡 FAISS: NOT LOADED"
        tavily_line = "🟢 Tavily: CONFIGURED" if health.get("tavily_configured") else "🟡 Tavily: NOT CONFIGURED"
    except RuntimeError:
        backend_line = "🔴 Backend: OFFLINE"
        faiss_line = "⚪ FAISS: UNKNOWN"
        tavily_line = "⚪ Tavily: UNKNOWN"

    await update.effective_message.reply_text(
        "🖥 TruthLens AI System Status\n\n"
        "🟢 Telegram Bot: ONLINE\n"
        f"{backend_line}\n{faiss_line}\n{tavily_line}\n"
        "🟢 OCR pipeline: AVAILABLE (Tesseract/OCR.space fallback)"
    )


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    entries = _history.get(user.id, []) if user else []
    if not entries:
        await update.effective_message.reply_text("No verifications yet.")
        return
    lines = ["Your recent verifications:"]
    for index, entry in enumerate(entries, start=1):
        claim = entry["claim"].replace("\n", " ")[:120]
        lines.append(f"{index}. {entry['verdict']} ({entry['confidence']}%) - {claim}")
    await update.effective_message.reply_text("\n".join(lines))


async def check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _verify_claim(update, " ".join(context.args))


async def forwarded_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    text = message.text or message.caption
    if not text:
        await message.reply_text(
            "📨 Forwarded message received, but no readable text was found. "
            "Send an image directly for OCR processing."
        )
        return
    await _verify_claim(update, text)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "❌ Unknown command.\n\nUse /help to see the available commands."
    )


async def verify_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _verify_claim(update, update.effective_message.text)


async def verify_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        photo = update.effective_message.photo[-1]
        image = await context.bot.get_file(photo.file_id)
        payload = bytes(await image.download_as_bytearray())
        result = BackendClient().verify_image(payload)
        _record_history(update, result)
        _print_verification_result(update, result.get("original_claim", "[image claim]"), result)
        await update.effective_message.reply_text(_summary(result), disable_web_page_preview=True)
    except RuntimeError as error:
        await update.effective_message.reply_text(str(error))

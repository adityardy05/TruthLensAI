from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from backend.core.env_loader import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    tavily_api_key: str | None
    ollama_base_url: str
    ollama_model: str
    faiss_index_path: Path
    faiss_metadata_path: Path
    max_upload_bytes: int
    request_timeout_seconds: int
    frontend_origin: str | None


def load_settings() -> Settings:
    """Read configuration exclusively from process environment variables."""
    index_dir = Path(os.getenv("TRUTHLENS_INDEX_DIR", PROJECT_ROOT / "indexes"))
    return Settings(
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "deepseek-r1:7b"),
        faiss_index_path=Path(os.getenv("FAISS_INDEX_PATH", index_dir / "truthlens.faiss")),
        faiss_metadata_path=Path(os.getenv("FAISS_METADATA_PATH", index_dir / "metadata.pkl")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        frontend_origin=os.getenv("FRONTEND_ORIGIN") or None,
    )


def validate_runtime_config(settings: Settings, *, mode: str = "web", strict: bool = False) -> bool:
    """Validate runtime config.

    Web retrieval and FAISS are optional for demo-mode startup. In non-strict mode we
    log warnings and continue so the app still serves the frontend; in strict mode we
    fail fast if required integration values are missing.
    """
    placeholder_values = {"your_tavily_key_here", "your_telegram_bot_token_here"}
    warnings: list[str] = []

    if mode == "web":
        if not settings.tavily_api_key or settings.tavily_api_key in placeholder_values:
            warnings.append("TAVILY_API_KEY missing or placeholder; live web retrieval will be disabled.")

        if not settings.faiss_index_path.exists() or not settings.faiss_metadata_path.exists():
            warnings.append(
                f"FAISS index missing: {settings.faiss_index_path} and {settings.faiss_metadata_path}; "
                "local retrieval will be disabled."
            )

    elif mode == "bot":
        token = os.getenv("TELEGRAM_BOT_TOKEN") or ""
        if not token or token in placeholder_values or ":" not in token:
            if strict:
                raise RuntimeError(
                    "Missing or invalid TELEGRAM_BOT_TOKEN. Create a bot with BotFather and set the real token in the .env file."
                )
            warnings.append("TELEGRAM_BOT_TOKEN missing or invalid; bot startup will be skipped.")
    else:
        raise ValueError(f"Unsupported runtime config mode: {mode}")

    if warnings:
        print("[config] Warning:")
        for warning in warnings:
            print(f"[config] - {warning}")
        return False

    return True

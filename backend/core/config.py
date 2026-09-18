from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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

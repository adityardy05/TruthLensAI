"""Create (or validate) the empty FAISS evidence cache.

TruthLens no longer ships a pre-built dataset index. The cache fills itself with extracted web
evidence as claims are verified. Run this once to create the files, or to see why an existing
index would be reset (an index without a matching manifest is moved aside as *.legacy-<timestamp>).

    python -m scripts.init_index
"""
from __future__ import annotations

from backend.core.config import load_settings
from backend.rag.embedding import EMBED_DIM, MODEL_NAME
from backend.rag.faiss_store import FaissEvidenceStore


def main() -> int:
    settings = load_settings()
    store = FaissEvidenceStore(settings.faiss_index_path, settings.faiss_metadata_path,
                               model_name=MODEL_NAME, dim=EMBED_DIM,
                               ttl_days=settings.retrieval.faiss_cache_ttl_days)
    if not store.available:
        print("FAISS cache unavailable (is faiss-cpu installed?)")
        return 1
    print(f"FAISS cache ready: {settings.faiss_index_path} ({store.count} entries)")
    if store.last_reset_reason:
        print(f"Previous index was moved aside: {store.last_reset_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

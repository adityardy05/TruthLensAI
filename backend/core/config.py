from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from backend.core.env_loader import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class RetrievalSettings:
    """Configuration for the SearXNG-based evidence retrieval pipeline.

    Every value can be overridden through an environment variable (see
    ``config/.env.example``). Defaults are the values agreed for the migration.
    """

    # --- SearXNG --------------------------------------------------------
    searxng_base_url: str = "http://127.0.0.1:8888"
    searxng_timeout_seconds: float = 10.0
    searxng_results_per_query: int = 20      # minimum target; every result on a page is kept
    searxng_max_pages: int = 1
    searxng_max_queries: int = 2
    searxng_categories: str = "general,news"
    max_query_chars: int = 200

    # --- FAISS cache routing -------------------------------------------
    faiss_similarity_threshold: float = 0.75  # routing threshold only (cosine); NOT R(d)
    faiss_cache_ttl_days: int = 7
    faiss_min_recency_score: float = 0.60  # below the existing one-year recency boundary is stale
    faiss_search_top_n: int = 30
    cache_min_relevance: float = 0.25         # chunks below this are not cached

    # --- Search-result ranking -----------------------------------------
    search_top_n: int = 10
    max_per_domain_search: int = 3
    search_weight_embedding: float = 0.70
    search_weight_bm25: float = 0.30

    # --- Fetching -------------------------------------------------------
    fetch_timeout_seconds: float = 8.0
    fetch_total_budget_seconds: float = 25.0
    fetch_max_bytes: int = 2_000_000
    fetch_workers: int = 5
    fetch_max_redirects: int = 3
    host_min_delay_seconds: float = 1.0
    crawler_user_agent: str = "TruthLensBot/1.0"
    robots_cache_ttl_seconds: int = 3600
    robots_max_delay_seconds: float = 5.0
    min_article_chars: int = 200

    # --- Chunking / cleaning -------------------------------------------
    chunk_chars: int = 600
    chunk_overlap_chars: int = 80
    chunk_min_chars: int = 80
    snippet_min_chars: int = 30
    max_article_chars: int = 100_000
    max_chunks_per_article: int = 25
    max_chunks_per_claim: int = 150
    near_duplicate_jaccard: float = 0.80
    syndication_containment: float = 0.60

    # --- R(d) -----------------------------------------------------------
    weight_relevance: float = 0.40
    weight_quality: float = 0.20
    weight_credibility: float = 0.25
    weight_recency: float = 0.15
    relevance_weight_embedding: float = 0.70
    relevance_weight_bm25: float = 0.30
    undated_recency_score: float = 0.40

    # --- Final selection -------------------------------------------------
    evidence_top_k: int = 5
    max_per_url_evidence: int = 2
    max_per_domain_evidence: int = 2
    max_per_group_evidence: int = 2


def load_retrieval_settings() -> RetrievalSettings:
    d = RetrievalSettings()
    return RetrievalSettings(
        searxng_base_url=_env_str("SEARXNG_BASE_URL", d.searxng_base_url).rstrip("/"),
        searxng_timeout_seconds=_env_float("SEARXNG_TIMEOUT_SECONDS", d.searxng_timeout_seconds),
        searxng_results_per_query=_env_int("SEARXNG_RESULTS_PER_QUERY", d.searxng_results_per_query),
        searxng_max_pages=max(1, _env_int("SEARXNG_MAX_PAGES", d.searxng_max_pages)),
        searxng_max_queries=max(1, _env_int("SEARXNG_MAX_QUERIES", d.searxng_max_queries)),
        searxng_categories=_env_str("SEARXNG_CATEGORIES", d.searxng_categories),
        max_query_chars=_env_int("MAX_QUERY_CHARS", d.max_query_chars),
        faiss_similarity_threshold=_env_float("FAISS_SIMILARITY_THRESHOLD", d.faiss_similarity_threshold),
        faiss_cache_ttl_days=_env_int("FAISS_CACHE_TTL_DAYS", d.faiss_cache_ttl_days),
        faiss_min_recency_score=_env_float("FAISS_MIN_RECENCY_SCORE", d.faiss_min_recency_score),
        faiss_search_top_n=_env_int("FAISS_SEARCH_TOP_N", d.faiss_search_top_n),
        cache_min_relevance=_env_float("CACHE_MIN_RELEVANCE", d.cache_min_relevance),
        search_top_n=_env_int("SEARCH_TOP_N", d.search_top_n),
        max_per_domain_search=_env_int("MAX_PER_DOMAIN_SEARCH", d.max_per_domain_search),
        search_weight_embedding=_env_float("SEARCH_WEIGHT_EMBEDDING", d.search_weight_embedding),
        search_weight_bm25=_env_float("SEARCH_WEIGHT_BM25", d.search_weight_bm25),
        fetch_timeout_seconds=_env_float("FETCH_TIMEOUT_SECONDS", d.fetch_timeout_seconds),
        fetch_total_budget_seconds=_env_float("FETCH_TOTAL_BUDGET_SECONDS", d.fetch_total_budget_seconds),
        fetch_max_bytes=_env_int("FETCH_MAX_BYTES", d.fetch_max_bytes),
        fetch_workers=max(1, _env_int("FETCH_WORKERS", d.fetch_workers)),
        fetch_max_redirects=_env_int("FETCH_MAX_REDIRECTS", d.fetch_max_redirects),
        host_min_delay_seconds=_env_float("HOST_MIN_DELAY_SECONDS", d.host_min_delay_seconds),
        crawler_user_agent=_env_str("CRAWLER_USER_AGENT", d.crawler_user_agent),
        robots_cache_ttl_seconds=_env_int("ROBOTS_CACHE_TTL_SECONDS", d.robots_cache_ttl_seconds),
        robots_max_delay_seconds=_env_float("ROBOTS_MAX_DELAY_SECONDS", d.robots_max_delay_seconds),
        min_article_chars=_env_int("MIN_ARTICLE_CHARS", d.min_article_chars),
        chunk_chars=_env_int("CHUNK_CHARS", d.chunk_chars),
        chunk_overlap_chars=_env_int("CHUNK_OVERLAP_CHARS", d.chunk_overlap_chars),
        chunk_min_chars=_env_int("CHUNK_MIN_CHARS", d.chunk_min_chars),
        snippet_min_chars=_env_int("SNIPPET_MIN_CHARS", d.snippet_min_chars),
        max_article_chars=_env_int("MAX_ARTICLE_CHARS", d.max_article_chars),
        max_chunks_per_article=_env_int("MAX_CHUNKS_PER_ARTICLE", d.max_chunks_per_article),
        max_chunks_per_claim=_env_int("MAX_CHUNKS_PER_CLAIM", d.max_chunks_per_claim),
        near_duplicate_jaccard=_env_float("NEAR_DUPLICATE_JACCARD", d.near_duplicate_jaccard),
        syndication_containment=_env_float("SYNDICATION_CONTAINMENT", d.syndication_containment),
        weight_relevance=_env_float("WEIGHT_RELEVANCE", d.weight_relevance),
        weight_quality=_env_float("WEIGHT_QUALITY", d.weight_quality),
        weight_credibility=_env_float("WEIGHT_CREDIBILITY", d.weight_credibility),
        weight_recency=_env_float("WEIGHT_RECENCY", d.weight_recency),
        relevance_weight_embedding=_env_float("RELEVANCE_WEIGHT_EMBEDDING", d.relevance_weight_embedding),
        relevance_weight_bm25=_env_float("RELEVANCE_WEIGHT_BM25", d.relevance_weight_bm25),
        undated_recency_score=_env_float("UNDATED_RECENCY_SCORE", d.undated_recency_score),
        evidence_top_k=max(1, _env_int("EVIDENCE_TOP_K", d.evidence_top_k)),
        max_per_url_evidence=max(1, _env_int("MAX_PER_URL_EVIDENCE", d.max_per_url_evidence)),
        max_per_domain_evidence=max(1, _env_int("MAX_PER_DOMAIN_EVIDENCE", d.max_per_domain_evidence)),
        max_per_group_evidence=max(1, _env_int("MAX_PER_GROUP_EVIDENCE", d.max_per_group_evidence)),
    )


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    faiss_index_path: Path
    faiss_metadata_path: Path
    max_upload_bytes: int
    request_timeout_seconds: int
    frontend_origin: str | None
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)


def load_settings() -> Settings:
    """Read configuration exclusively from process environment variables."""
    index_dir = Path(os.getenv("TRUTHLENS_INDEX_DIR", PROJECT_ROOT / "indexes"))
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "deepseek-r1:7b"),
        faiss_index_path=Path(os.getenv("FAISS_INDEX_PATH", index_dir / "truthlens.faiss")),
        faiss_metadata_path=Path(os.getenv("FAISS_METADATA_PATH", index_dir / "metadata.pkl")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        frontend_origin=os.getenv("FRONTEND_ORIGIN") or None,
        retrieval=load_retrieval_settings(),
    )


def validate_runtime_config(settings: Settings, *, mode: str = "web", strict: bool = False) -> bool:
    """Validate runtime config.

    SearXNG and the FAISS cache are optional for demo-mode startup. In non-strict
    mode we log warnings and continue so the app still serves the frontend; in
    strict mode we fail fast if required integration values are missing.
    """
    placeholder_values = {"your_telegram_bot_token_here"}
    warnings: list[str] = []

    if mode == "web":
        if not settings.retrieval.searxng_base_url.startswith(("http://", "https://")):
            warnings.append("SEARXNG_BASE_URL is not an http(s) URL; live web retrieval will be disabled.")

        index_dir = settings.faiss_index_path.parent
        if not index_dir.exists():
            warnings.append(
                f"FAISS index directory {index_dir} does not exist; it will be created on first use."
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


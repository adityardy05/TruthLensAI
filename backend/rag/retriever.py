"""Evidence retrieval orchestrator (Team A).

    claim -> queries -> FAISS cache lookup
          -> best cosine >= threshold : reuse cached chunks
          -> otherwise                : SearXNG -> hybrid rank -> Top 10 -> robots.txt -> fetch ->
                                        Trafilatura (snippet fallback) -> chunk -> cleanup
          -> de-duplicate -> R(d) -> Top 5 -> Team B

The public interface is unchanged: ``TeamARetrievalPipeline.process_claim(claim, original_language,
top_k)`` returns ``{claim, original_language, normalized_claim, sub_claims, evidence, metadata}``.
Tavily is no longer used anywhere.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from backend.core.config import RetrievalSettings, load_retrieval_settings
from backend.rag.article_fetcher import ArticleFetcher
from backend.rag.domain_credibility import HybridDomainCredibility
from backend.rag.embedding import MODEL_NAME, Embedder, MiniLMEmbedder
from backend.rag.evidence_processing import build_chunks, dedupe_pool, prefilter_chunks
from backend.rag.faiss_store import FaissEvidenceStore
from backend.rag.query_builder import build_queries
from backend.rag.scoring import compute_recency_score, score_chunks, select_top_k  # noqa: F401 (re-exported)
from backend.rag.search_ranking import rank_search_results
from backend.rag.searxng_client import SearXNGClient, SearXNGError, merge_duplicates

logger = logging.getLogger(__name__)


class TeamARetrievalPipeline:
    def __init__(
        self,
        index_path=None,
        metadata_path=None,
        model_name: str = MODEL_NAME,
        *,
        settings: Optional[RetrievalSettings] = None,
        embedder: Optional[Embedder] = None,
        store: Optional[FaissEvidenceStore] = None,
        searxng: Optional[SearXNGClient] = None,
        fetcher: Optional[ArticleFetcher] = None,
        domain_evaluator: Optional[HybridDomainCredibility] = None,
    ):
        self.settings = settings or load_retrieval_settings()
        s = self.settings
        self.embedder = embedder if embedder is not None else MiniLMEmbedder(model_name)
        self.store = store if store is not None else FaissEvidenceStore(
            index_path, metadata_path, self.embedder, model_name=model_name, ttl_days=s.faiss_cache_ttl_days)
        self.searxng = searxng or SearXNGClient(
            s.searxng_base_url, timeout=s.searxng_timeout_seconds, categories=s.searxng_categories,
            max_pages=s.searxng_max_pages, results_per_query=s.searxng_results_per_query,
            user_agent=s.crawler_user_agent)
        self.fetcher = fetcher or ArticleFetcher(s)
        self.domain_evaluator = domain_evaluator or HybridDomainCredibility()

    # ------------------------------------------------------------- utilities
    def generate_sub_claims(self, claim: str) -> List[str]:
        """Heuristic sub-claims (informational only; search uses the whole claim)."""
        claim_clean = claim.strip()
        sub_claims = [claim_clean]
        words = claim_clean.split()
        if len(words) > 6:
            sub_claims.append(" ".join(words[: len(words) // 2]))
            sub_claims.append(" ".join(words[len(words) // 2:]))
        return list(dict.fromkeys(sub_claims))

    def health(self) -> Dict[str, Any]:
        return {
            "faiss_loaded": bool(self.store.available),
            "faiss_entries": self.store.count if self.store.available else 0,
            "embedder_available": bool(self.embedder is not None and self.embedder.available),
            "searxng_configured": bool(self.settings.searxng_base_url),
            "searxng_reachable": self.searxng.is_reachable(),
        }

    def _cred(self, url: str) -> float:
        try:
            return float(self.domain_evaluator.evaluate(url)[0])
        except Exception:
            return 0.5

    def _embed_query(self, query_text: str, meta: dict):
        if self.embedder is None or not self.embedder.available:
            meta["degraded"].append("embedder_unavailable")
            return None
        try:
            return self.embedder.encode([query_text])[0]
        except Exception as exc:  # EmbeddingError or anything unexpected: skip the cache, still search the web
            logger.error("Query embedding failed, cache lookup skipped: %s", exc)
            meta["degraded"].append("embedder_unavailable")
            return None

    def _expand_hits(self, hits: List[dict], min_anchor_similarity: float) -> List[dict]:
        """Chunk hits plus the chunks linked from matching query anchors (de-duplicated)."""
        chunks = [h for h in hits if h.get("kind", "chunk") == "chunk"]
        seen = {h["faiss_id"] for h in chunks}
        keys: List[str] = []
        for h in hits:
            if h.get("kind") == "query" and h["embedding_similarity"] >= min_anchor_similarity:
                keys.extend(h.get("chunk_keys", []))
        for linked in self.store.get_chunks_by_keys(keys):
            if linked["faiss_id"] not in seen:
                seen.add(linked["faiss_id"])
                chunks.append(linked)
        return chunks

    @staticmethod
    def _hit_to_chunk(hit: dict, rank: int) -> dict:
        try:
            index = int(str(hit.get("chunk_id", "")).rsplit(":", 1)[-1])
        except ValueError:
            index = 0
        return {
            "article_id": hit.get("article_id") or hit.get("url"), "url": hit["url"], "final_url": hit["url"],
            "domain": hit.get("domain"), "title": hit.get("title") or "", "text": hit["text"],
            "published_date": hit.get("published_date"), "date_source": hit.get("date_source") or "none",
            "author": hit.get("author"), "robots_allowed": None, "access_status": "CACHED",
            "fetch_status": "CACHED", "extraction_status": "OK",
            "extraction_method": hit.get("extraction_method") or "trafilatura", "evidence_type": "extracted",
            "retrieval_rank": rank, "search_relevance": None, "search_query": hit.get("search_query"),
            "search_engines": [], "origin": "faiss_cache", "source_type": "faiss_cache",
            "embedding_similarity": hit.get("embedding_similarity"), "chunk_id": hit.get("chunk_id") or f"{hit['url']}:{index}",
            "chunk_index": index, "cleanup_flags": [], "removed_chars": 0, "cached_at": hit.get("cached_at"),
            "content_hash": hit.get("content_hash"), "source_records": hit.get("source_records", []),
            "last_seen_at": hit.get("last_seen_at"),
        }

    @staticmethod
    def _to_evidence(c: dict) -> dict:
        return {
            "source_id": c["chunk_id"], "source_domain": c["domain"], "source_title": c["title"], "url": c["url"],
            "content_hash": c.get("content_hash"), "source_records": c.get("source_records", []),
            "content": c["text"], "published_date": c.get("published_date"), "date_source": c.get("date_source"),
            "source_type": c["source_type"], "origin": c["origin"], "evidence_type": c["evidence_type"],
            "extraction_method": c["extraction_method"], "access_status": c["access_status"],
            "fetch_status": c["fetch_status"], "extraction_status": c["extraction_status"],
            "robots_allowed": c["robots_allowed"],
            "bm25_score": c["bm25_score"], "cosine_score": c["cosine_score"],
            "relevance_score": c["relevance_score"], "quality_score": c["quality_score"],
            "domain_credibility": c["domain_credibility"], "credibility_source": c["credibility_source"],
            "recency_score": c["recency_score"], "combined_reliability": c["combined_reliability"],
            "retrieval_rank": c["retrieval_rank"], "article_id": c["article_id"], "chunk_id": c["chunk_id"],
            "syndication_group_id": c["syndication_group_id"], "origin_url": c["origin_url"],
            "is_syndicated_copy": c["is_syndicated_copy"], "also_published_by": c.get("also_published_by", []),
            "cleanup_flags": c.get("cleanup_flags", []), "search_query": c.get("search_query"),
            "search_engines": c.get("search_engines", []), "author": c.get("author"), "cached_at": c.get("cached_at"),
        }

    # --------------------------------------------------------------- web path
    def _web_chunks(self, query_text: str, queries: List[str], language: str, meta: dict) -> List[dict]:
        s = self.settings
        t = time.perf_counter()
        results: List[dict] = []
        for q in queries:
            try:
                resp = self.searxng.search(q, language)
            except SearXNGError as exc:
                logger.warning("SearXNG search failed for %r: %s", q, exc)
                meta["searxng_errors"].append({"query": q, "kind": exc.kind})
                continue
            except Exception as exc:  # never let a search client bug take the claim down
                logger.exception("Unexpected SearXNG error for %r", q)
                meta["searxng_errors"].append({"query": q, "kind": f"unexpected:{type(exc).__name__}"})
                continue
            results.extend(resp.results)
            meta["unresponsive_engines"].extend([list(p) for p in resp.unresponsive_engines])
        merged = merge_duplicates(results)
        meta["raw_results"] = len(merged)
        meta["stage_timings_ms"]["search"] = round((time.perf_counter() - t) * 1000)
        if not merged:
            return []

        t = time.perf_counter()
        selected, _, degraded = rank_search_results(
            query_text, merged, self.embedder, w_emb=s.search_weight_embedding, w_bm25=s.search_weight_bm25,
            top_n=s.search_top_n, max_per_domain=s.max_per_domain_search)
        meta["degraded"].extend(degraded)
        meta["search_ranked"] = len(selected)
        meta["stage_timings_ms"]["search_ranking"] = round((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        articles = self.fetcher.fetch_all(selected)
        meta["fetched_extracted"] = sum(1 for a in articles if a["evidence_type"] == "extracted")
        meta["snippet_fallbacks"] = len(articles) - meta["fetched_extracted"]
        meta["stage_timings_ms"]["fetch"] = round((time.perf_counter() - t) * 1000)

        chunks: List[dict] = []
        for article in articles:
            chunks.extend(build_chunks(article, s))
        return prefilter_chunks(query_text, chunks, s)

    # ------------------------------------------------------------ main entry
    def process_claim(self, claim: str, original_language: str = "en", top_k: Optional[int] = None,
                      original_claim: Optional[str] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        s = self.settings
        k = max(1, int(top_k)) if top_k else s.evidence_top_k
        normalized_claim = (claim or "").strip()
        sub_claims = self.generate_sub_claims(normalized_claim)
        queries = build_queries(normalized_claim, original_claim, original_language,
                                s.searxng_max_queries, s.max_query_chars)
        query_text = queries[0] if queries else normalized_claim[: s.max_query_chars]
        meta: Dict[str, Any] = {
            "route": None, "queries": queries, "faiss_threshold": s.faiss_similarity_threshold,
            "best_faiss_similarity": 0.0, "searxng_errors": [], "unresponsive_engines": [], "degraded": [],
            "raw_results": 0, "search_ranked": 0, "fetched_extracted": 0, "snippet_fallbacks": 0,
            "cached_chunks_added": 0, "stage_timings_ms": {},
        }

        t = time.perf_counter()
        qvec = self._embed_query(query_text, meta) if (query_text and self.store.available) else None
        hits: List[dict] = self.store.search(qvec, s.faiss_search_top_n) if qvec is not None else []
        best_hit = max(hits, key=lambda h: h["embedding_similarity"], default=None)
        best = best_hit["embedding_similarity"] if best_hit else 0.0
        meta["best_faiss_similarity"] = round(best, 4)
        meta["best_match_kind"] = best_hit.get("kind", "chunk") if best_hit else None
        meta["faiss_entries"] = self.store.count if self.store.available else 0
        meta["stage_timings_ms"]["faiss"] = round((time.perf_counter() - t) * 1000)

        pool: List[dict] = []
        cached_chunks = [
            self._hit_to_chunk(hit, i + 1)
            for i, hit in enumerate(self._expand_hits(hits, s.faiss_similarity_threshold))
        ]
        fresh_cached_chunks = [
            chunk for chunk in cached_chunks
            if compute_recency_score(chunk.get("published_date"), s.undated_recency_score)
            >= s.faiss_min_recency_score
        ]
        meta["fresh_cached_chunks"] = len(fresh_cached_chunks)
        meta["faiss_min_recency_score"] = s.faiss_min_recency_score
        # The cache must be able to satisfy the requested evidence count; otherwise
        # search live rather than silently returning a partial cached set.
        cache_sufficient = len(fresh_cached_chunks) >= k
        meta["faiss_cache_sufficient"] = cache_sufficient

        if hits and best >= s.faiss_similarity_threshold and cache_sufficient:
            meta["route"] = "faiss"
            pool = fresh_cached_chunks
        else:
            # Permanent evidence is retained, but stale or undated evidence never suppresses
            # a fresh SearXNG lookup. It remains a fallback if live search yields nothing.
            meta["route"] = "searxng"
            if query_text:
                pool = self._web_chunks(query_text, queries, original_language, meta)
            if not pool and cached_chunks:
                meta["route"] = "faiss_fallback"
                pool = cached_chunks

        t = time.perf_counter()
        pool, dstats = dedupe_pool(
            pool, cred_fn=self._cred, near_jaccard=s.near_duplicate_jaccard,
            syndication_containment=s.syndication_containment)
        meta["dedupe"] = dstats
        meta["degraded"].extend(score_chunks(query_text, pool, self.embedder, self.domain_evaluator, s))
        selected, info = select_top_k(pool, s, k)
        meta["stage_timings_ms"]["process_and_score"] = round((time.perf_counter() - t) * 1000)

        if meta["route"] == "searxng" and self.store.available:
            to_cache = [c for c in pool if c.get("origin") == "searxng" and c["evidence_type"] == "extracted"
                        and c["relevance_score"] >= s.cache_min_relevance]
            try:
                meta["cached_chunks_added"] = self.store.add_chunks(to_cache, query=query_text, query_vector=qvec)
            except Exception:
                logger.exception("Caching evidence failed; continuing without cache")

        evidence = [self._to_evidence(c) for c in selected]
        meta.update(
            retrieval_time_ms=round((time.perf_counter() - started) * 1000),
            num_sources_total=len(pool), num_sources_kept=len(evidence),
            num_live_web_sources=sum(1 for e in evidence if e["source_type"] == "web"),
            num_cached_sources=sum(1 for e in evidence if e["source_type"] == "faiss_cache"),
            top_source_domain=evidence[0]["source_domain"] if evidence else None,
            independent_source_count=info["independent_source_count"],
            insufficient_evidence=info["insufficient_evidence"],
            insufficient_reason=info.get("insufficient_reason"),
            degraded=sorted(set(meta["degraded"])),
        )
        if info["insufficient_evidence"]:
            logger.info("Insufficient evidence for claim (%s): %s", meta["route"], info.get("insufficient_reason"))
        return {
            "claim": normalized_claim, "original_language": original_language,
            "normalized_claim": normalized_claim, "sub_claims": sub_claims,
            "evidence": evidence, "metadata": meta,
        }


# Backwards-compatible alias
EvidenceRetrievalPipeline = TeamARetrievalPipeline


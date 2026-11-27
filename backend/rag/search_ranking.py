"""Ranking of raw SearXNG results (before any page is fetched) and the shared BM25 scorer.

SearchRelevance = w_emb * EmbeddingSimilarity + w_bm25 * NormalizedBM25   (defaults 0.70 / 0.30)

This is a *search-candidate* score used only to choose which URLs are worth
fetching. It is not the final evidence score R(d).
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.rag.embedding import Embedder, EmbeddingError
from backend.rag.text_utils import content_tokens, tokenize

logger = logging.getLogger(__name__)


class BM25:
    """Okapi BM25 with the Lucene IDF (always positive, stable on tiny corpora).

    Normalisation: a term's score is strictly below ``idf * (k1 + 1)`` (the limit as
    term frequency grows), so dividing a document's score by the sum of that bound over
    the query terms gives a value in [0, 1) that does not depend on the pool maximum.
    (Dividing by the best document's score instead would always give the top hit 1.0.)
    """

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        self.idf: Dict[str, float] = {
            t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def _terms(self, query_terms: Sequence[str]) -> List[str]:
        return [t for t in dict.fromkeys(query_terms) if t in self.idf]

    def upper_bound(self, query_terms: Sequence[str]) -> float:
        return sum(self.idf[t] * (self.k1 + 1.0) for t in self._terms(query_terms))

    def raw_scores(self, query_terms: Sequence[str]) -> List[float]:
        terms = self._terms(query_terms)
        scores: List[float] = []
        for counts, dl in zip(self.tf, self.doc_len):
            norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
            s = 0.0
            for t in terms:
                f = counts.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1.0) / (f + norm)
            scores.append(s)
        return scores

    def normalized_scores(self, query_terms: Sequence[str]) -> List[float]:
        bound = self.upper_bound(query_terms)
        if bound <= 0:
            return [0.0] * self.n
        return [min(1.0, max(0.0, s / bound)) for s in self.raw_scores(query_terms)]


def bm25_normalized(query_text: str, doc_texts: Sequence[str]) -> List[float]:
    if not doc_texts:
        return []
    bm25 = BM25([tokenize(t) for t in doc_texts])
    return bm25.normalized_scores(content_tokens(query_text))


def cosine_scores(embedder: Optional[Embedder], claim: str, texts: Sequence[str]) -> Tuple[List[float], Optional[np.ndarray]]:
    """Cosine similarity of ``claim`` to each text, clipped to [0, 1]. Returns (scores, text_vectors).

    On embedding failure returns zeros and ``None`` so the caller can degrade to BM25 only."""
    if not texts:
        return [], None
    if embedder is None or not embedder.available:
        return [0.0] * len(texts), None
    try:
        vectors = embedder.encode([claim] + list(texts))
    except EmbeddingError as exc:
        logger.error("Embedding failed, degrading to BM25 only: %s", exc)
        return [0.0] * len(texts), None
    sims = vectors[1:] @ vectors[0]
    return [float(min(1.0, max(0.0, s))) for s in sims], vectors[1:]


def effective_weights(w_emb: float, w_bm25: float, embeddings_ok: bool) -> Tuple[float, float]:
    """When embeddings are unavailable, rank on BM25 alone instead of silently halving every score."""
    return (w_emb, w_bm25) if embeddings_ok else (0.0, 1.0)


def rank_search_results(
    claim: str,
    results: List[dict],
    embedder: Optional[Embedder],
    *,
    w_emb: float = 0.70,
    w_bm25: float = 0.30,
    top_n: int = 10,
    max_per_domain: int = 3,
) -> Tuple[List[dict], List[dict], List[str]]:
    """Score every result and select the Top-N URLs.

    Returns ``(selected, all_scored, degraded_flags)``. Each result keeps its original
    metadata plus ``embedding_similarity``, ``bm25_norm`` and ``search_relevance``.
    """
    degraded: List[str] = []
    if not results:
        return [], [], degraded
    texts = [f"{r.get('title', '')}. {r.get('snippet', '')}".strip(". ") for r in results]
    bm25 = bm25_normalized(claim, texts)
    cos, vectors = cosine_scores(embedder, claim, texts)
    if vectors is None:  # model missing or encode failed: rank on BM25 alone
        degraded.append("embedder_unavailable")
    we, wb = effective_weights(w_emb, w_bm25, "embedder_unavailable" not in degraded)

    scored: List[dict] = []
    for r, c, b in zip(results, cos, bm25):
        item = dict(r)
        item["embedding_similarity"] = round(c, 4)
        item["bm25_norm"] = round(b, 4)
        item["search_relevance"] = round(we * c + wb * b, 4)
        scored.append(item)
    scored.sort(key=lambda x: (-x["search_relevance"], x["retrieval_rank"]))

    selected: List[dict] = []
    skipped: List[dict] = []
    per_domain: Counter = Counter()
    for item in scored:
        if len(selected) >= top_n:
            break
        if per_domain[item["domain"]] >= max_per_domain:
            skipped.append(item)
            continue
        per_domain[item["domain"]] += 1
        selected.append(item)
    for item in skipped:  # relax the domain cap rather than return fewer than top_n
        if len(selected) >= top_n:
            break
        selected.append(item)
    return selected, scored, degraded

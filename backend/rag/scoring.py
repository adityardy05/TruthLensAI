"""Evidence scoring R(d) and final Top-K selection.

    R(d) = 0.40*Relevance + 0.20*Quality + 0.25*Credibility + 0.15*Recency        (weights configurable)
    Relevance = 0.70*EmbeddingSimilarity + 0.30*NormalizedBM25                     (both in [0, 1])

R(d) is a ranking/selection score, not a deletion filter: weak and conflicting evidence still
reaches Team B when it is among the best available.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

from typing import List, Optional, Tuple

from backend.core.config import RetrievalSettings
from backend.rag.embedding import Embedder, EmbeddingError
from backend.rag.search_ranking import BM25, effective_weights
from backend.rag.text_utils import content_tokens, tokenize

logger = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y")


# ---------------------------------------------------------------- recency
def parse_date(value) -> Optional[datetime]:
    """Parse a date to a naive UTC datetime, or None if missing/invalid."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        dt = None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in _FORMATS:
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            m = _ISO.search(text)
            if m:
                try:
                    dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    return None
        if dt is None:
            return None
    if dt.tzinfo is not None:
        dt = (dt - dt.utcoffset()).replace(tzinfo=None)
    return dt


def compute_recency_score(value, undated: float = 0.40, now: Optional[datetime] = None) -> float:
    """<=30 days: 1.0; then linear to 0.6 at 1 year and to 0.1 at 5 years (older: 0.1).
    Missing, unparseable or future-dated values get ``undated`` (an unknown date is never
    treated as a confirmed recent publication)."""
    dt = parse_date(value)
    now = now or _utcnow_naive()
    if dt is None or dt > now + timedelta(days=1):
        return round(undated, 4)
    age = max((now - dt).days, 0)
    if age <= 30:
        score = 1.0
    elif age <= 365:
        score = 1.0 - 0.4 * (age - 30) / 335
    elif age <= 1825:
        score = 0.6 - 0.5 * (age - 365) / 1460
    else:
        score = 0.1
    return round(max(0.1, min(1.0, score)), 4)


# ---------------------------------------------------------------- quality
def quality_score(chunk: dict) -> float:
    """Deterministic evidence-quality estimate in [0, 1].

    0.50 * source type (extracted 1.0 / search snippet 0.4)
    0.20 * length adequacy (saturates at 300 chars: longer is not automatically better)
    0.20 * content signal (numbers, named entities, full sentences)
    0.10 * cleanliness (reduced if cleanup had to remove text)
    """
    text = chunk.get("text", "") or ""
    type_score = 1.0 if chunk.get("evidence_type") == "extracted" else 0.4
    length_score = min(len(text) / 300.0, 1.0)
    words = text.split()
    signal = 0.0
    if any(ch.isdigit() for ch in text):
        signal += 0.3
    if sum(1 for w in words[1:] if w[:1].isupper()) >= 2:
        signal += 0.3
    if len(re.findall(r"[.!?\u0964](\s|$)", text)) >= 2:
        signal += 0.2
    if text.rstrip().endswith((".", "!", "?", "\u0964", '"', "\u201d")):
        signal += 0.2
    clean = 0.5 if chunk.get("cleanup_flags") else 1.0
    return round(0.5 * type_score + 0.2 * length_score + 0.2 * min(signal, 1.0) + 0.1 * clean, 4)


# ---------------------------------------------------------------- R(d)
def compute_rd(relevance: float, quality: float, credibility: float, recency: float, s: RetrievalSettings) -> float:
    rd = (s.weight_relevance * relevance + s.weight_quality * quality
          + s.weight_credibility * credibility + s.weight_recency * recency)
    return round(max(0.0, min(1.0, rd)), 4)


def score_chunks(
    claim: str,
    chunks: List[dict],
    embedder: Optional[Embedder],
    domain_eval,
    s: RetrievalSettings,
    now: Optional[datetime] = None,
) -> List[str]:
    """Add every R(d) component to each chunk in place. Returns degradation flags.

    Chunks that already carry ``embedding_similarity`` (FAISS hits) are not re-embedded. New vectors
    are kept in ``_embedding`` so the FAISS cache does not have to embed them again.
    """
    degraded: List[str] = []
    if not chunks:
        return degraded

    bm25 = BM25([tokenize(c["text"]) for c in chunks]).normalized_scores(content_tokens(claim))

    todo = [i for i, c in enumerate(chunks) if c.get("embedding_similarity") is None]
    embeddings_ok = True
    if todo:
        embeddings_ok = False
        if embedder is not None and embedder.available:
            try:
                vecs = embedder.encode([claim] + [chunks[i]["text"] for i in todo])
                for pos, i in enumerate(todo):
                    chunks[i]["embedding_similarity"] = float(min(1.0, max(0.0, vecs[pos + 1] @ vecs[0])))
                    chunks[i]["_embedding"] = vecs[pos + 1]
                embeddings_ok = True
            except EmbeddingError as exc:
                logger.error("Chunk embedding failed, scoring on BM25 only: %s", exc)
        if not embeddings_ok:
            degraded.append("embedder_unavailable")
            for i in todo:
                chunks[i]["embedding_similarity"] = 0.0

    w_emb, w_bm25 = effective_weights(s.relevance_weight_embedding, s.relevance_weight_bm25, embeddings_ok)
    for c, b in zip(chunks, bm25):
        cos = float(c["embedding_similarity"])
        cred, cred_source = domain_eval.evaluate(c.get("url") or c.get("domain") or "")
        rel = max(0.0, min(1.0, w_emb * cos + w_bm25 * b))
        qual = quality_score(c)
        rec = compute_recency_score(c.get("published_date"), s.undated_recency_score, now)
        c.update(
            cosine_score=round(cos, 4), bm25_score=round(b, 4), relevance_score=round(rel, 4),
            quality_score=qual, domain_credibility=round(float(cred), 4), credibility_source=cred_source,
            recency_score=rec, combined_reliability=compute_rd(rel, qual, float(cred), rec, s),
        )
    return degraded


# --------------------------------------------------------------- selection
def select_top_k(chunks: List[dict], s: RetrievalSettings, k: Optional[int] = None) -> Tuple[List[dict], dict]:
    """Pick the K best chunks by R(d), limiting copies of the same story.

    Greedy in R(d) order with caps per URL / domain / syndication group. If the caps would leave
    fewer than K, they are relaxed (group, then domain, then URL) rather than returning fewer
    items than exist. No item is ever invented."""
    k = k or s.evidence_top_k
    ranked = sorted(chunks, key=lambda c: (-c["combined_reliability"], c.get("retrieval_rank") or 0))
    selected: List[dict] = []
    chosen = set()

    def run(max_url: Optional[int], max_domain: Optional[int], max_group: Optional[int]) -> None:
        urls, domains, groups = Counter(), Counter(), Counter()
        for c in selected:
            urls[c["article_id"]] += 1
            domains[c["domain"]] += 1
            groups[c["syndication_group_id"]] += 1
        for c in ranked:
            if len(selected) >= k:
                return
            if id(c) in chosen:
                continue
            if max_url is not None and urls[c["article_id"]] >= max_url:
                continue
            if max_domain is not None and domains[c["domain"]] >= max_domain:
                continue
            if max_group is not None and groups[c["syndication_group_id"]] >= max_group:
                continue
            selected.append(c)
            chosen.add(id(c))
            urls[c["article_id"]] += 1
            domains[c["domain"]] += 1
            groups[c["syndication_group_id"]] += 1

    run(s.max_per_url_evidence, s.max_per_domain_evidence, s.max_per_group_evidence)
    if len(selected) < k:
        run(s.max_per_url_evidence, s.max_per_domain_evidence, None)
    if len(selected) < k:
        run(s.max_per_url_evidence, None, None)
    if len(selected) < k:
        run(None, None, None)

    selected.sort(key=lambda c: (-c["combined_reliability"], c.get("retrieval_rank") or 0))
    for rank, c in enumerate(selected, 1):
        c["retrieval_rank"] = rank
    info = {
        "requested": k,
        "returned": len(selected),
        "available_candidates": len(chunks),
        "independent_source_count": len({c["syndication_group_id"] for c in selected}),
        "insufficient_evidence": len(selected) < k,
    }
    if info["insufficient_evidence"]:
        info["insufficient_reason"] = (
            f"only {len(selected)} valid evidence item(s) were available after retrieval, cleaning and de-duplication")
    return selected, info

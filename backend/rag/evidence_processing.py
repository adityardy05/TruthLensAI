"""Chunking, deterministic prompt-injection / junk cleanup, and de-duplication.

Webpage text is untrusted input and is never treated as instructions. Cleanup is regex-based (no
LLM) and sentence-level: only sentences that try to instruct the model (or are obvious page
boilerplate) are dropped; the rest of a chunk and its source attribution are kept.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from hashlib import sha1
from typing import Callable, Dict, List, Optional, Set, Tuple

from backend.core.config import RetrievalSettings
from backend.rag.search_ranking import BM25
from backend.rag.text_utils import (
    containment,
    content_tokens,
    jaccard,
    normalized_hash,
    shingles,
    split_sentences,
    tokenize,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- chunking
_WS = re.compile(r"\s+")


def _collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _hard_split(sentence: str, size: int) -> List[str]:
    parts, cur = [], ""
    for word in sentence.split(" "):
        if cur and len(cur) + 1 + len(word) > size:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        parts.append(cur)
    return parts


def _overlap_tail(chunk: str, overlap: int) -> str:
    if overlap <= 0 or len(chunk) <= overlap:
        return ""
    tail = chunk[-overlap:]
    return tail.split(" ", 1)[1].strip() if " " in tail else ""


def chunk_text(text: str, size: int = 600, overlap: int = 80, min_chars: int = 80) -> List[str]:
    """Sentence-aware chunks of about ``size`` characters with a word-aligned overlap.
    A short trailing chunk is merged into its predecessor instead of being dropped."""
    sentences = [_collapse(s) for s in split_sentences(text)]
    chunks: List[str] = []
    cur, tail = "", ""
    for sentence in sentences:
        for piece in ([sentence] if len(sentence) <= size else _hard_split(sentence, size)):
            if cur and len(cur) + 1 + len(piece) > size:
                chunks.append(cur)
                tail = _overlap_tail(cur, overlap)
                cur = f"{tail} {piece}".strip() if tail else piece
            else:
                cur = f"{cur} {piece}".strip() if cur else piece
    if cur:
        if chunks and len(cur) < min_chars:
            chunks[-1] = f"{chunks[-1]} {cur[len(tail):].strip()}".strip()
        else:
            chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- cleanup
_INJECTION_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("ignore_instructions", re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,30}\b(previous|prior|above|earlier|preceding|system)\b"
        r"[^.\n]{0,20}\b(instructions?|prompts?|rules|messages?|context)\b", re.I)),
    ("ignore_your_instructions", re.compile(
        r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+)?your\s+(instructions?|prompt|rules|guidelines)\b", re.I)),
    ("reveal_prompt", re.compile(
        r"\b(reveal|print|repeat|output|leak|show)\b[^.\n]{0,25}\b(your\s+(instructions?|prompt|rules)|"
        r"(the\s+)?(system|hidden|secret|initial)\s+(instructions?|prompt))\b", re.I)),
    ("role_override", re.compile(
        r"\b(you\s+are\s+(now\s+)?(a|an)\s+[^.\n]{0,25}\b(assistant|language\s+model|chatbot|AI)\b|"
        r"act\s+as\s+(a|an|if)\b[^.\n]{0,40}\b(assistant|AI|chatbot|DAN|jailbroken)\b|"
        r"as\s+an\s+AI\s+language\s+model)", re.I)),
    ("verdict_steering", re.compile(
        r"\b((always|only)\s+(answer|respond|reply|output)\s+(with\s+)?[\"']?(true|false|verified)\b|"
        r"the\s+(final\s+)?verdict\s+(must|should|has\s+to)\s+be\b)", re.I)),
    ("role_marker", re.compile(
        r"(<\|?(im_start|im_end|system|assistant|endoftext)\|?>|\[/?INST\]|<</?SYS>>|^\s*(system|assistant)\s*:|"
        r"\bnew\s+instructions?\s*:|\bsystem\s+(prompt|message)\s*:|^\s*#{2,}\s*(system|instructions?)\b)",
        re.I | re.M)),
    ("conceal_from_user", re.compile(r"\bdo\s+not\s+(tell|inform)\s+the\s+user\b", re.I)),
]
_BOILERPLATE = re.compile(
    r"\b(accept\s+(all\s+)?cookies|cookie\s+(policy|settings|preferences)|subscribe\s+to\s+(our|the)\s+newsletter|"
    r"sign\s+up\s+for\s+(our|the)|all\s+rights\s+reserved|follow\s+us\s+on|click\s+here\s+to|advertisement|"
    r"read\s+more:|related\s+(articles?|stories)|share\s+(this|on)\s+(facebook|twitter|whatsapp)|"
    r"terms\s+of\s+(use|service)|privacy\s+policy)\b", re.I)
_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def clean_text(text: str) -> Tuple[str, List[str], int]:
    """Return ``(cleaned_text, flags, removed_chars)``.

    Flags name what was found (e.g. ``ignore_instructions``, ``boilerplate``, ``junk``); the removed
    text itself is not stored. Normal factual sentences, including ordinary imperatives such as
    "Wash your hands for 20 seconds.", are untouched."""
    text = "".join(ch for ch in _INVISIBLE.sub("", text or "")
                   if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    kept: List[str] = []
    flags: List[str] = []
    removed = 0
    for sentence in split_sentences(text):
        sentence = _collapse(sentence)
        hit = next((name for name, pat in _INJECTION_PATTERNS if pat.search(sentence)), None)
        if hit is None and len(sentence) < 140 and _BOILERPLATE.search(sentence):
            hit = "boilerplate"
        if hit is None and len(sentence) > 20 and sum(c.isalnum() for c in sentence) / len(sentence) < 0.5:
            hit = "junk"
        if hit:
            flags.append(hit)
            removed += len(sentence)
            continue
        kept.append(sentence)
    return " ".join(kept), sorted(set(flags)), removed


# ---------------------------------------------------------- chunk building
_ARTICLE_FIELDS = (
    "article_id", "url", "final_url", "domain", "title", "published_date", "date_source", "author",
    "robots_allowed", "access_status", "fetch_status", "extraction_status", "extraction_method",
    "evidence_type", "retrieval_rank", "search_relevance", "search_query", "search_engines", "origin",
    "source_type",
)


def build_chunks(article: dict, s: RetrievalSettings) -> List[dict]:
    """Chunk one article record, clean each chunk and keep the source metadata on every chunk."""
    text = (article.get("text") or "")[: s.max_article_chars]
    is_snippet = article.get("evidence_type") == "snippet"
    pieces = [text] if is_snippet else chunk_text(text, s.chunk_chars, s.chunk_overlap_chars, s.chunk_min_chars)
    floor = s.snippet_min_chars if is_snippet else s.chunk_min_chars
    out: List[dict] = []
    for idx, piece in enumerate(pieces):
        cleaned, flags, removed = clean_text(piece)
        if flags:
            logger.warning("Cleanup removed %s from %s (%d chars)", flags, article.get("url"), removed)
        if len(cleaned) < floor:
            continue
        chunk = {k: article.get(k) for k in _ARTICLE_FIELDS}
        chunk.update(text=cleaned, chunk_id=f"{article['article_id']}:{idx}", chunk_index=idx,
                     cleanup_flags=flags, removed_chars=removed)
        out.append(chunk)
    return out


def prefilter_chunks(claim: str, chunks: List[dict], s: RetrievalSettings) -> List[dict]:
    """Keep at most ``max_chunks_per_article`` / ``max_chunks_per_claim`` chunks, chosen by BM25
    against the claim (cheap, no embeddings). Snippets are always kept; original order is preserved."""
    if not chunks:
        return []
    scores = BM25([tokenize(c["text"]) for c in chunks]).normalized_scores(content_tokens(claim))
    indexed = list(enumerate(zip(chunks, scores)))
    by_article: Dict[str, List[Tuple[int, float]]] = {}
    keep: Set[int] = set()
    for i, (c, sc) in indexed:
        if c["evidence_type"] == "snippet":
            keep.add(i)
        else:
            by_article.setdefault(c["article_id"], []).append((i, sc))
    candidates: List[Tuple[int, float]] = []
    for items in by_article.values():
        items.sort(key=lambda x: -x[1])  # stable: ties keep document order
        candidates.extend(items[: s.max_chunks_per_article])
    candidates.sort(key=lambda x: -x[1])
    room = max(0, s.max_chunks_per_claim - len(keep))
    keep.update(i for i, _ in candidates[:room])
    return [chunks[i] for i in sorted(keep)]


# ------------------------------------------------------------ de-duplication
def _date_key(value) -> str:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    return m.group(0) if m else "9999-99-99"


def dedupe_pool(
    chunks: List[dict],
    *,
    cred_fn: Callable[[str], float],
    near_jaccard: float = 0.80,
    syndication_containment: float = 0.60,
    syndication_min_shingles: int = 15,
    near_containment: float = 0.85,
) -> Tuple[List[dict], dict]:
    """Remove exact and near-duplicate chunks and detect syndicated articles.

    * Duplicate passages inside one article: dropped (Jaccard >= ``near_jaccard``).
    * Distinct chunks of the same article: always kept.
    * Near-identical text on another URL/domain: the copy is dropped and its domain recorded in the
      kept chunk's ``also_published_by``.
    * Articles whose text mostly overlaps (containment >= ``syndication_containment``) form a
      syndication group; the origin is the earliest dated article (then most credible, then best
      search rank). Every kept chunk gets ``syndication_group_id``, ``origin_url`` and
      ``is_syndicated_copy`` so selection can avoid counting copies as independent confirmation.

    Cost: O(n^2) shingle comparisons over at most ``max_chunks_per_claim`` (150) chunks.
    """
    stats = {"input": len(chunks), "exact_removed": 0, "near_removed": 0, "groups": 0, "syndicated_articles": 0}
    if not chunks:
        return [], stats

    sh = [shingles(tokenize(c["text"])) for c in chunks]
    art: Dict[str, dict] = {}
    for c, s_ in zip(chunks, sh):
        a = art.setdefault(c["article_id"], {"sh": set(), "url": c["url"], "date": c.get("published_date"),
                                             "rank": c.get("retrieval_rank") or 0, "domain": c.get("domain")})
        a["sh"] |= s_

    ids = list(art)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            A, B = art[a_id]["sh"], art[b_id]["sh"]
            if min(len(A), len(B)) >= syndication_min_shingles and containment(A, B) >= syndication_containment:
                parent[find(a_id)] = find(b_id)

    members: Dict[str, List[str]] = {}
    for a_id in ids:
        members.setdefault(find(a_id), []).append(a_id)
    origin_of: Dict[str, str] = {}
    group_of: Dict[str, str] = {}
    for root, group in members.items():
        if len(group) > 1:
            stats["groups"] += 1
            stats["syndicated_articles"] += len(group) - 1
        origin = min(group, key=lambda g: (_date_key(art[g]["date"]), -cred_fn(art[g]["url"]), art[g]["rank"]))
        gid = ("syn-" + sha1("|".join(sorted(group)).encode()).hexdigest()[:8]) if len(group) > 1 else group[0]
        for g in group:
            origin_of[g], group_of[g] = origin, gid

    order = sorted(range(len(chunks)), key=lambda i: (
        0 if chunks[i]["evidence_type"] == "extracted" else 1,
        0 if chunks[i]["article_id"] == origin_of[chunks[i]["article_id"]] else 1,
        chunks[i].get("retrieval_rank") or 0, chunks[i].get("chunk_index", 0)))

    kept: List[int] = []
    exact: Dict[str, int] = {}
    for i in order:
        c = chunks[i]
        h = normalized_hash(c["text"])
        dup_of: Optional[int] = exact.get(h)
        kind = "exact"
        if dup_of is None:
            kind = "near"
            for j in kept:
                same_article = chunks[j]["article_id"] == c["article_id"]
                sim = jaccard(sh[i], sh[j])
                # A chunk wholly contained in an already-kept chunk from another article is a copy. The
                # reverse (a larger chunk that merely includes a kept passage plus extra text) is kept,
                # so additional information in a syndicated version is never lost.
                if sim >= near_jaccard or (not same_article and len(sh[i]) >= 8 and len(sh[i]) <= len(sh[j])
                                           and containment(sh[i], sh[j]) >= near_containment):
                    dup_of = j
                    break
        if dup_of is not None:
            stats["exact_removed" if kind == "exact" else "near_removed"] += 1
            if chunks[dup_of]["article_id"] != c["article_id"]:
                also = chunks[dup_of].setdefault("also_published_by", [])
                if c.get("domain") and c["domain"] != chunks[dup_of].get("domain") and c["domain"] not in also:
                    also.append(c["domain"])
            continue
        exact[h] = i
        kept.append(i)

    out: List[dict] = []
    for i in kept:
        c = chunks[i]
        a_id = c["article_id"]
        c.setdefault("also_published_by", [])
        c["syndication_group_id"] = group_of[a_id]
        c["origin_url"] = art[origin_of[a_id]]["url"]
        c["is_syndicated_copy"] = a_id != origin_of[a_id]
        out.append(c)
    return out, stats

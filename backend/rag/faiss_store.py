"""FAISS cache of *extracted, cleaned evidence chunks*.

The index starts empty (there is no pre-built dataset). It holds only chunks that came from fetched
and extracted web pages; search snippets are deliberately not cached, because a weak snippet that
happens to match a future claim would suppress a fresh web search while offering weak evidence.

Two kinds of entries share the index:
  * ``chunk``  - an extracted evidence chunk (vector of its text).
  * ``query``  - an *anchor*: the vector of the claim/query that retrieved a batch of chunks, with the
    keys of those chunks. A claim is one short sentence while chunks are long passages, so claim-to-chunk
    cosine is usually well below 0.75 even for relevant text; claim-to-earlier-claim cosine is a
    like-for-like comparison, which is what makes the 0.75 routing threshold effective for repeated or
    paraphrased claims.

Metric: ``IndexFlatIP`` over L2-normalised MiniLM vectors, so the inner product IS cosine similarity
(0.75 is a cosine threshold). Evidence chunks and embeddings are permanent; only query anchors expire
after ``ttl_days``. The retriever recalculates publication-date freshness before reusing cached chunks.

Safety: a ``<index>.manifest.json`` records schema version, embedding model, dimension and entry
count. On load the index size, metadata length and manifest must all agree and the model must match;
otherwise the old files are renamed ``*.legacy-<timestamp>`` (never deleted) and an empty cache is
started. The cache can always be rebuilt from the web, so resetting is safe.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.rag.embedding import EMBED_DIM, MODEL_NAME, Embedder, EmbeddingError
from backend.rag.text_utils import normalized_hash

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:  # pragma: no cover - exercised only when faiss is not installed
    faiss = None
    HAS_FAISS = False

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
_ENTRY_FIELDS = (
    "text", "url", "title", "domain", "published_date", "date_source", "article_id", "chunk_id",
    "extraction_method", "author", "search_query", "content_hash", "source_records", "last_seen_at",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FaissEvidenceStore:
    def __init__(
        self,
        index_path: Optional[os.PathLike],
        metadata_path: Optional[os.PathLike],
        embedder: Optional[Embedder] = None,
        *,
        model_name: str = MODEL_NAME,
        dim: int = EMBED_DIM,
        ttl_days: int = 7,
    ):
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.manifest_path = Path(f"{self.index_path}.manifest.json") if self.index_path else None
        self.embedder = embedder
        self.model_name, self.dim, self.ttl_days = model_name, dim, ttl_days
        self._lock = threading.RLock()
        self.index = None
        self.metadata: List[dict] = []
        self._hash_to_idx: Dict[str, int] = {}
        self.available = False
        self.last_reset_reason: Optional[str] = None
        if not (HAS_FAISS and self.index_path and self.metadata_path):
            logger.warning("FAISS cache disabled (faiss installed=%s, paths configured=%s)",
                           HAS_FAISS, bool(self.index_path and self.metadata_path))
            return
        try:
            self._load_or_init()
            self.available = True
        except Exception as exc:
            logger.error("FAISS cache could not be initialised: %s", exc)

    # ------------------------------------------------------------ lifecycle
    @property
    def count(self) -> int:
        return len(self.metadata)

    def _empty(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self._hash_to_idx = {}

    def _quarantine(self, reason: str) -> None:
        self.last_reset_reason = reason
        stamp = _utcnow().strftime("%Y%m%d%H%M%S")
        logger.warning("Resetting FAISS cache (%s); old files are kept as *.legacy-%s", reason, stamp)
        for path in (self.index_path, self.metadata_path, self.manifest_path):
            if path and path.exists():
                try:
                    os.replace(path, f"{path}.legacy-{stamp}")
                except OSError as exc:
                    logger.error("Could not move %s aside: %s", path, exc)

    def _load_or_init(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        exists = [p.exists() for p in (self.index_path, self.metadata_path)]
        if not any(exists) and not self.manifest_path.exists():
            self._empty()
            self._persist()
            return
        try:
            if not self.manifest_path.exists():
                raise ValueError("no manifest (legacy or unknown index)")
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"schema version {manifest.get('schema_version')} != {SCHEMA_VERSION}")
            if manifest.get("embedding_model") != self.model_name or manifest.get("dim") != self.dim:
                raise ValueError(f"embedding mismatch: {manifest.get('embedding_model')}/{manifest.get('dim')}")
            index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as fh:
                metadata = pickle.load(fh)
            if not isinstance(metadata, list):
                raise ValueError("metadata is not a list")
            if index.d != self.dim:
                raise ValueError(f"index dimension {index.d} != {self.dim}")
            if not (index.ntotal == len(metadata) == manifest.get("ntotal")):
                raise ValueError(f"misaligned: index={index.ntotal} metadata={len(metadata)} manifest={manifest.get('ntotal')}")
        except Exception as exc:
            self._quarantine(str(exc))
            self._empty()
            self._persist()
            return
        self.index, self.metadata = index, metadata
        self._reindex()
        self.prune_expired()

    @staticmethod
    def _content_hash(text: str) -> str:
        """Stable SHA-256 over normalized evidence text, independent of its source URL."""
        normalized = " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def _entry_hash(cls, url: str, text: str) -> str:
        # URL is deliberately excluded: identical evidence from different pages is one chunk.
        return f"chunk|{cls._content_hash(text)}"

    @staticmethod
    def _query_hash(query: str) -> str:
        return f"query|{normalized_hash(query)}"

    def _meta_hash(self, m: dict) -> str:
        if m.get("kind") == "query":
            return self._query_hash(m.get("query", ""))
        return self._entry_hash(m.get("url", ""), m.get("text", ""))

    def _reindex(self) -> None:
        self._hash_to_idx = {self._meta_hash(m): i for i, m in enumerate(self.metadata)}

    def _persist(self) -> None:
        """Write index, metadata and manifest via temp files. The three replaces are not one atomic
        operation; a crash between them is detected on the next load (counts disagree) and handled
        by ``_quarantine``."""
        tmp_i, tmp_m, tmp_f = (Path(f"{p}.tmp") for p in (self.index_path, self.metadata_path, self.manifest_path))
        try:
            faiss.write_index(self.index, str(tmp_i))
            with open(tmp_m, "wb") as fh:
                pickle.dump(self.metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_f.write_text(json.dumps({
                "schema_version": SCHEMA_VERSION, "embedding_model": self.model_name, "dim": self.dim,
                "metric": "inner_product_l2_normalised", "ntotal": self.index.ntotal,
                "updated_at": _utcnow().isoformat(),
            }), encoding="utf-8")
            os.replace(tmp_m, self.metadata_path)
            os.replace(tmp_i, self.index_path)
            os.replace(tmp_f, self.manifest_path)
        finally:
            for tmp in (tmp_i, tmp_m, tmp_f):
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    # --------------------------------------------------------------- expiry
    def _expired(self, entry: dict, now: datetime) -> bool:
        # Evidence chunks are permanent. Only query anchors have a short routing lifetime;
        # their linked chunks remain available and are freshness-scored on every reuse.
        if entry.get("kind") != "query" or self.ttl_days <= 0:
            return False
        try:
            cached_at = datetime.fromisoformat(entry["cached_at"])
        except (KeyError, ValueError, TypeError):
            return True
        return now - cached_at > timedelta(days=self.ttl_days)

    def prune_expired(self) -> int:
        """Rebuild the index without expired entries (vectors are copied, not re-embedded)."""
        with self._lock:
            if not self.available and self.index is None:
                return 0
            now = _utcnow()
            keep = [i for i, m in enumerate(self.metadata) if not self._expired(m, now)]
            removed = len(self.metadata) - len(keep)
            if not removed:
                return 0
            vectors = np.vstack([self.index.reconstruct(i) for i in keep]) if keep else None
            self._empty_keep_dim()
            if vectors is not None:
                self.index.add(vectors.astype("float32"))
            self.metadata = [self.metadata[i] for i in keep]
            self._reindex()
            self._persist()
            logger.info("Pruned %d expired cache entries", removed)
            return removed

    def _empty_keep_dim(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)

    # --------------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, top_n: int = 30) -> List[dict]:
        """Nearest cached chunks (unexpired only), each with ``embedding_similarity`` in [0, 1]."""
        with self._lock:
            if not self.available or self.index.ntotal == 0:
                return []
            q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
            scores, ids = self.index.search(q, min(top_n, self.index.ntotal))
            now = _utcnow()
            hits: List[dict] = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                entry = self.metadata[idx]
                if self._expired(entry, now):
                    continue
                hit = dict(entry)
                hit.setdefault("kind", "chunk")
                hit["embedding_similarity"] = float(min(1.0, max(0.0, score)))
                hit["faiss_id"] = int(idx)
                hits.append(hit)
            return hits

    def get_chunks_by_keys(self, keys: List[str]) -> List[dict]:
        """Unexpired chunk entries for the given chunk keys (used to expand a matched query anchor).
        ``embedding_similarity`` is left as None: the caller scores them against the claim."""
        with self._lock:
            now = _utcnow()
            out: List[dict] = []
            for key in dict.fromkeys(keys):
                idx = self._hash_to_idx.get(key)
                if idx is None:
                    continue
                entry = self.metadata[idx]
                if entry.get("kind") == "query" or self._expired(entry, now):
                    continue
                item = dict(entry)
                item.update(kind="chunk", embedding_similarity=None, faiss_id=idx)
                out.append(item)
            return out

    def search_text(self, text: str, top_n: int = 30) -> List[dict]:
        if not self.available or self.embedder is None or not self.embedder.available:
            return []
        try:
            return self.search(self.embedder.encode([text])[0], top_n)
        except EmbeddingError as exc:
            logger.error("FAISS lookup skipped, embedding failed: %s", exc)
            return []

    # ---------------------------------------------------------------- write

    @staticmethod
    def _source_record(chunk: dict) -> dict:
        return {
            "url": chunk.get("url"),
            "domain": chunk.get("domain"),
            "title": chunk.get("title"),
            "published_date": chunk.get("published_date"),
            "date_source": chunk.get("date_source"),
            "author": chunk.get("author"),
        }

    def add_chunks(self, chunks: List[dict], query: Optional[str] = None,
                   query_vector: Optional[np.ndarray] = None) -> int:
        """Persist extracted chunks permanently and de-duplicate by SHA-256 content hash.

        Duplicate text updates the existing chunk's source provenance and last-seen metadata
        without adding a second vector. The seven-day TTL applies only to query anchors.
        Returns the number of new evidence chunks added.
        """
        with self._lock:
            if not self.available:
                return 0

            now_dt = _utcnow()
            now = now_dt.isoformat()
            fresh: List[dict] = []
            fresh_by_hash: Dict[str, dict] = {}
            keys: List[str] = []
            metadata_updated = False

            for chunk in chunks:
                text = (chunk.get("text") or "").strip()
                url = chunk.get("url") or ""
                if chunk.get("evidence_type") != "extracted" or not text or not url:
                    continue

                key = self._entry_hash(url, text)
                if key not in keys:
                    keys.append(key)
                source = self._source_record(chunk)

                existing_idx = self._hash_to_idx.get(key)
                if existing_idx is not None:
                    existing = self.metadata[existing_idx]
                    source_records = existing.setdefault("source_records", [])
                    if not source_records and existing.get("url"):
                        source_records.append(self._source_record(existing))
                    prior = next((record for record in source_records if record.get("url") == source.get("url")), None)
                    if prior is None:
                        source_records.append(source)
                    else:
                        prior.update({key: value for key, value in source.items() if value is not None})
                    # Refresh non-content metadata when the incoming source has a known date.
                    if source.get("published_date") and not existing.get("published_date"):
                        existing["published_date"] = source["published_date"]
                        existing["date_source"] = source.get("date_source")
                    existing["last_seen_at"] = now
                    metadata_updated = True
                    continue

                if key in fresh_by_hash:
                    primary = fresh_by_hash[key]
                    if source not in primary["source_records"]:
                        primary["source_records"].append(source)
                    continue

                item = dict(chunk)
                item["content_hash"] = self._content_hash(text)
                item["source_records"] = [source]
                item["last_seen_at"] = now
                fresh.append(item)
                fresh_by_hash[key] = item

            need = [item for item in fresh if item.get("_embedding") is None]
            if need:
                if self.embedder is None or not self.embedder.available:
                    fresh = [item for item in fresh if item.get("_embedding") is not None]
                else:
                    try:
                        vectors = self.embedder.encode([item["text"] for item in need])
                        for item, vector in zip(need, vectors):
                            item["_embedding"] = vector
                    except EmbeddingError as exc:
                        logger.error("Not caching %d chunks, embedding failed: %s", len(need), exc)
                        fresh = [item for item in fresh if item.get("_embedding") is not None]

            entries: List[dict] = []
            vectors: List[np.ndarray] = []
            for item in fresh:
                entry: Dict[str, object] = {key: item.get(key) for key in _ENTRY_FIELDS}
                entry.update(kind="chunk", evidence_type="extracted", cached_at=now)
                entries.append(entry)
                vectors.append(np.asarray(item["_embedding"], dtype="float32"))

            # Query anchors are the only entries governed by ttl_days. An expired anchor
            # is refreshed in place; its evidence chunks are never removed with it.
            anchor_updated = False
            if query and query_vector is not None and keys:
                query_key = self._query_hash(query)
                unique_keys = list(dict.fromkeys(keys))
                if query_key in self._hash_to_idx:
                    existing = self.metadata[self._hash_to_idx[query_key]]
                    merged = list(dict.fromkeys(list(existing.get("chunk_keys", [])) + unique_keys))
                    anchor_updated = merged != existing.get("chunk_keys")
                    existing["chunk_keys"] = merged
                    if self._expired(existing, now_dt):
                        existing["cached_at"] = now
                        anchor_updated = True
                else:
                    entries.append({
                        "kind": "query", "query": query, "chunk_keys": unique_keys,
                        "cached_at": now, "text": query, "url": "query:",
                    })
                    vectors.append(np.asarray(query_vector, dtype="float32"))

            if not entries and not anchor_updated:
                if metadata_updated:
                    self._persist()
                return 0

            if entries:
                start = len(self.metadata)
                self.index.add(np.vstack(vectors))
                self.metadata.extend(entries)
                for offset, entry in enumerate(entries):
                    self._hash_to_idx[self._meta_hash(entry)] = start + offset

            try:
                self._persist()
            except Exception as exc:
                logger.error("Could not persist FAISS cache (kept in memory): %s", exc)
            return len(fresh)

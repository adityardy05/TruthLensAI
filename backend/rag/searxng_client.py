"""Client for a self-hosted SearXNG instance (JSON API).

SearXNG only performs the web search. Ranking, fetching, extraction, cleaning,
scoring and selection all happen elsewhere in TruthLens.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

from backend.rag.text_utils import canonical_url, domain_of, is_internal_host

logger = logging.getLogger(__name__)


class SearXNGError(RuntimeError):
    """Raised when a search could not be completed. ``kind`` is a short machine-readable reason."""

    def __init__(self, kind: str, message: str = ""):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind


@dataclass
class SearchResponse:
    results: List[dict] = field(default_factory=list)
    unresponsive_engines: List[Tuple[str, str]] = field(default_factory=list)
    pages_fetched: int = 0


def normalize_result(raw: dict, query: str, rank: int, retrieved_at: str) -> Optional[dict]:
    """Map one raw SearXNG result to the internal schema (None if the URL is unusable)."""
    if not isinstance(raw, dict):
        return None
    url = str(raw.get("url") or "").strip()
    canon = canonical_url(url)
    if not canon or is_internal_host(domain_of(url)):
        return None
    engines = raw.get("engines")
    if not isinstance(engines, list) or not engines:
        engines = [raw["engine"]] if raw.get("engine") else []
    return {
        "url": url,
        "canonical_url": canon,
        "domain": domain_of(url),
        "title": str(raw.get("title") or "").strip(),
        "snippet": str(raw.get("content") or "").strip(),
        "engines": [str(e) for e in engines],
        "published_date": raw.get("publishedDate") or raw.get("pubdate") or None,
        "query": query,
        "retrieval_rank": rank,
        "retrieved_at": retrieved_at,
        "searxng_score": raw.get("score"),
    }


def merge_duplicates(results: List[dict]) -> List[dict]:
    """Collapse results with the same canonical URL, keeping the best rank, union of engines
    and the longest snippet."""
    merged: Dict[str, dict] = {}
    order: List[str] = []
    for item in results:
        key = item["canonical_url"]
        if key not in merged:
            merged[key] = dict(item)
            order.append(key)
            continue
        kept = merged[key]
        kept["engines"] = sorted(set(kept["engines"]) | set(item["engines"]))
        if len(item["snippet"]) > len(kept["snippet"]):
            kept["snippet"] = item["snippet"]
        if not kept["title"] and item["title"]:
            kept["title"] = item["title"]
        if not kept["published_date"] and item["published_date"]:
            kept["published_date"] = item["published_date"]
        kept["retrieval_rank"] = min(kept["retrieval_rank"], item["retrieval_rank"])
    return [merged[k] for k in order]


class SearXNGClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        categories: str = "general,news",
        max_pages: int = 1,
        results_per_query: int = 20,
        session: Optional[requests.Session] = None,
        user_agent: str = "TruthLensBot/1.0",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.categories = categories
        self.max_pages = max(1, max_pages)
        self.results_per_query = results_per_query
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", user_agent)

    def _get_page(self, query: str, page: int, language: Optional[str]) -> dict:
        params = {"q": query, "format": "json", "pageno": page}
        if self.categories:
            params["categories"] = self.categories
        if language and language.lower() != "en":
            params["language"] = language
        try:
            response = self.session.get(f"{self.base_url}/search", params=params, timeout=self.timeout)
        except requests.Timeout as exc:
            raise SearXNGError("timeout", str(exc)) from exc
        except requests.RequestException as exc:
            raise SearXNGError("connection", str(exc)) from exc
        if response.status_code == 403:
            raise SearXNGError("forbidden", "SearXNG refused the request (is the json format enabled in settings.yml?)")
        if response.status_code >= 400:
            raise SearXNGError(f"http_{response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise SearXNGError("invalid_json", str(exc)) from exc
        if not isinstance(data, dict):
            raise SearXNGError("invalid_json", "top-level JSON value is not an object")
        return data

    def search(self, query: str, language: Optional[str] = None) -> SearchResponse:
        """Run one query. Page 1 is always requested; further pages only while fewer than
        ``results_per_query`` unique URLs have been seen. Every result on a page is kept."""
        query = (query or "").strip()
        out = SearchResponse()
        if not query:
            return out
        retrieved_at = datetime.now(timezone.utc).isoformat()
        collected: List[dict] = []
        for page in range(1, self.max_pages + 1):
            try:
                data = self._get_page(query, page, language)
            except SearXNGError as exc:
                if page == 1:
                    raise
                logger.warning("SearXNG page %d failed for %r: %s", page, query, exc)
                break
            out.pages_fetched += 1
            for pair in data.get("unresponsive_engines") or []:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    out.unresponsive_engines.append((str(pair[0]), str(pair[1])))
            raw_results = data.get("results") or []
            if not isinstance(raw_results, list):
                raw_results = []
            for raw in raw_results:
                item = normalize_result(raw, query, len(collected) + 1, retrieved_at)
                if item:
                    collected.append(item)
            if len(merge_duplicates(collected)) >= self.results_per_query:
                break
        out.results = merge_duplicates(collected)
        if out.unresponsive_engines:
            logger.warning("SearXNG unresponsive engines for %r: %s", query, out.unresponsive_engines)
        return out

    def is_reachable(self, timeout: float = 2.0) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/", timeout=timeout)
            return response.status_code < 500
        except requests.RequestException:
            return False

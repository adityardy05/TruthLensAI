"""robots.txt check -> safe fetch -> Trafilatura, with search-snippet fallback.

For every shortlisted search result this produces exactly one *article record*:
  * ``evidence_type == "extracted"`` when the page was permitted, fetched and Trafilatura
    returned usable text; or
  * ``evidence_type == "snippet"`` when anything failed. The original SearXNG title/snippet is
    kept (never presented as a full article) and the status fields record why.

Status vocabularies:
  access_status:     OK | ROBOTS_DISALLOWED | ROBOTS_UNREACHABLE | SERVER_DENIED | UNSAFE_URL | NOT_ATTEMPTED
  fetch_status:      OK | HTTP_ERROR | TIMEOUT | TOO_LARGE | NON_HTML | REDIRECT_LIMIT | CONNECTION_ERROR | DNS_ERROR | NOT_ATTEMPTED
  extraction_status: OK | EMPTY | FAILED | NOT_ATTEMPTED
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone

from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from backend.core.config import RetrievalSettings
from backend.rag.robots import RobotsChecker
from backend.rag.safe_fetch import (
    FetchError,
    FetchHTTPError,
    UnresolvableHostError,
    UnsafeURLError,
    safe_get,
    validate_public_http_url,
)
from backend.rag.text_utils import canonical_url, domain_of

logger = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

_URL_DATE = re.compile(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:[/_.-]|$)")
SERVER_DENIED_CODES = (401, 403, 429)


def date_from_url(url: str, now: Optional[datetime] = None) -> Optional[str]:
    """Deterministic fallback: a /YYYY/MM/DD/ pattern in the URL path (labelled ``url_pattern``)."""
    match = _URL_DATE.search(urlparse(url or "").path)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        found = datetime(year, month, day)
    except ValueError:
        return None
    if found > (now or _utcnow_naive()) + timedelta(days=1):
        return None
    return found.strftime("%Y-%m-%d")


def default_extractor(body: bytes, url: str) -> Tuple[str, dict]:
    """Run Trafilatura on fetched HTML. Returns (text, metadata)."""
    import trafilatura

    # No ``deduplicate=True``: Trafilatura keeps a process-wide cache of seen segments, so later
    # calls would silently drop text already seen in other pages. We de-duplicate ourselves.
    try:
        text = trafilatura.extract(body, url=url, include_comments=False, include_tables=False,
                                   favor_precision=True)
    except TypeError:  # older Trafilatura without some keyword arguments
        text = trafilatura.extract(body, url=url)
    meta: dict = {}
    try:
        doc = trafilatura.extract_metadata(body, default_url=url)
        if doc is not None:
            meta = {"title": getattr(doc, "title", None), "date": getattr(doc, "date", None),
                    "author": getattr(doc, "author", None)}
    except Exception as exc:
        logger.debug("Trafilatura metadata failed for %s: %s", url, exc)
    return text or "", meta


class ArticleFetcher:
    def __init__(
        self,
        settings: RetrievalSettings,
        *,
        session: Optional[requests.Session] = None,
        robots: Optional[RobotsChecker] = None,
        resolver: Optional[Callable] = None,
        extractor: Callable[[bytes, str], Tuple[str, dict]] = default_extractor,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.s = settings
        self.session = session or requests.Session()
        self.resolver = resolver
        self.extractor = extractor
        self.sleep = sleep
        self.clock = clock
        self.robots = robots or RobotsChecker(
            self.session, user_agent=settings.crawler_user_agent,
            timeout=min(5.0, settings.fetch_timeout_seconds),
            cache_ttl=settings.robots_cache_ttl_seconds, resolver=resolver)
        self._next_slot: Dict[str, float] = {}
        self._slot_lock = threading.Lock()

    # ---------------------------------------------------------------- helpers
    def _throttle(self, host: str, crawl_delay: Optional[float]) -> None:
        delay = max(self.s.host_min_delay_seconds, min(crawl_delay or 0.0, self.s.robots_max_delay_seconds))
        if delay <= 0:
            return
        with self._slot_lock:
            now = self.clock()
            start = max(now, self._next_slot.get(host, 0.0))
            self._next_slot[host] = start + delay
        if start > now:
            self.sleep(start - now)

    def _record(self, cand: dict, **over) -> dict:
        url = cand["url"]
        snippet = cand.get("snippet", "") or ""
        title = cand.get("title", "") or ""
        text = snippet if len(snippet) >= self.s.snippet_min_chars else f"{title}. {snippet}".strip(". ")
        date, date_source = cand.get("published_date"), ("search" if cand.get("published_date") else "none")
        if not date:
            url_date = date_from_url(url)
            if url_date:
                date, date_source = url_date, "url_pattern"
        rec = {
            "article_id": hashlib.sha1(canonical_url(url).encode()).hexdigest()[:12],
            "url": url,
            "final_url": url,
            "domain": cand.get("domain") or domain_of(url),
            "title": title,
            "text": text,
            "published_date": date,
            "date_source": date_source,
            "author": None,
            "robots_allowed": None,
            "access_status": "NOT_ATTEMPTED",
            "fetch_status": "NOT_ATTEMPTED",
            "extraction_status": "NOT_ATTEMPTED",
            "fetch_error": None,
            "extraction_method": "search_snippet",
            "evidence_type": "snippet",
            "retrieval_rank": cand.get("retrieval_rank", 0),
            "search_relevance": cand.get("search_relevance"),
            "search_query": cand.get("query"),
            "search_engines": cand.get("engines", []),
            "origin": "searxng",
            "source_type": "web",
        }
        rec.update(over)
        return rec

    # ------------------------------------------------------------------ fetch
    def fetch_one(self, cand: dict) -> dict:
        url = cand["url"]
        try:
            host = validate_public_http_url(url, self.resolver)
        except UnresolvableHostError as exc:
            return self._record(cand, fetch_status="DNS_ERROR", fetch_error=str(exc))
        except UnsafeURLError as exc:
            return self._record(cand, access_status="UNSAFE_URL", fetch_error=str(exc))

        decision = self.robots.check(url)
        if not decision.may_fetch:
            status = "ROBOTS_UNREACHABLE" if decision.status == "unreachable" else "ROBOTS_DISALLOWED"
            return self._record(cand, access_status=status, robots_allowed=decision.robots_allowed,
                                fetch_error=f"robots.txt: {decision.status}")

        def on_hop(next_url: str) -> None:
            if domain_of(next_url) != domain_of(url):
                hop = self.robots.check(next_url)
                if not hop.may_fetch:
                    raise FetchError(f"redirect target blocked by robots.txt ({hop.status})")

        self._throttle(host, decision.crawl_delay)
        base = {"robots_allowed": decision.robots_allowed}
        try:
            result = safe_get(
                url, session=self.session, timeout=self.s.fetch_timeout_seconds,
                max_bytes=self.s.fetch_max_bytes, max_redirects=self.s.fetch_max_redirects,
                headers={"User-Agent": self.s.crawler_user_agent,
                         "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"},
                resolver=self.resolver, on_hop=on_hop)
        except FetchHTTPError as exc:
            denied = exc.status_code in SERVER_DENIED_CODES
            return self._record(cand, **base, access_status="SERVER_DENIED" if denied else "OK",
                                fetch_status="HTTP_ERROR", fetch_error=str(exc))
        except UnresolvableHostError as exc:  # a redirect target did not resolve
            return self._record(cand, **base, access_status="OK", fetch_status="DNS_ERROR", fetch_error=str(exc))
        except UnsafeURLError as exc:  # a redirect pointed at a non-public address
            return self._record(cand, **base, access_status="UNSAFE_URL", fetch_error=str(exc))
        except FetchError as exc:
            return self._record(cand, **base, access_status="OK",
                                fetch_status=getattr(exc, "status", "ERROR"), fetch_error=str(exc))

        try:
            text, meta = self.extractor(result.body, result.final_url)
        except Exception as exc:
            logger.warning("Extraction failed for %s: %s", url, exc)
            return self._record(cand, **base, access_status="OK", fetch_status="OK",
                                extraction_status="FAILED", fetch_error=f"extraction: {exc}",
                                final_url=result.final_url)
        text = (text or "").strip()
        if len(text) < self.s.min_article_chars:
            return self._record(cand, **base, access_status="OK", fetch_status="OK",
                                extraction_status="EMPTY", final_url=result.final_url)

        meta = meta or {}
        date, date_source = None, "none"
        for value, source in ((cand.get("published_date"), "search"), (meta.get("date"), "trafilatura"),
                              (date_from_url(result.final_url) or date_from_url(url), "url_pattern")):
            if value:
                date, date_source = value, source
                break
        return self._record(
            cand, **base, access_status="OK", fetch_status="OK", extraction_status="OK",
            final_url=result.final_url, text=text, evidence_type="extracted",
            extraction_method="trafilatura", title=meta.get("title") or cand.get("title", ""),
            published_date=date, date_source=date_source, author=meta.get("author") or None)

    def fetch_all(self, candidates: List[dict]) -> List[dict]:
        """Fetch candidates in parallel within a total time budget; results keep input order."""
        if not candidates:
            return []
        pool = ThreadPoolExecutor(max_workers=self.s.fetch_workers, thread_name_prefix="fetch")
        futures = [pool.submit(self.fetch_one, c) for c in candidates]
        done, _ = wait(futures, timeout=self.s.fetch_total_budget_seconds)
        pool.shutdown(wait=False, cancel_futures=True)
        out: List[dict] = []
        for cand, fut in zip(candidates, futures):
            if fut in done:
                try:
                    out.append(fut.result())
                    continue
                except Exception as exc:  # defensive: one bad URL must not sink the claim
                    logger.exception("Unexpected fetch error for %s", cand.get("url"))
                    out.append(self._record(cand, fetch_status="ERROR", fetch_error=str(exc)))
                    continue
            out.append(self._record(cand, fetch_status="TIMEOUT",
                                    fetch_error="total fetch budget exhausted"))
        return out

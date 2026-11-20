"""robots.txt compliance (RFC 9309-style) with a per-origin cache.

Policy (documented decision D6):
  * 200                         -> parse the rules and apply them.
  * 404 / 410 / other 4xx       -> robots.txt does not exist: crawling allowed, recorded as unavailable.
  * 401 / 403                   -> treated as "everything disallowed" (conservative, same as Python's stdlib).
  * 5xx, timeout, network error -> robots.txt unreachable: do NOT fetch the page (its search snippet is kept).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from backend.rag.safe_fetch import FetchError, FetchHTTPError, safe_get

logger = logging.getLogger(__name__)

ALLOWED = "allowed"
DISALLOWED = "disallowed"
FORBIDDEN = "forbidden"              # robots.txt itself answered 401/403
UNAVAILABLE_ALLOWED = "unavailable_allowed"
UNREACHABLE = "unreachable"
ROBOTS_MAX_BYTES = 512 * 1024


@dataclass(frozen=True)
class RobotsDecision:
    status: str
    crawl_delay: Optional[float] = None

    @property
    def may_fetch(self) -> bool:
        return self.status in (ALLOWED, UNAVAILABLE_ALLOWED)

    @property
    def robots_allowed(self) -> Optional[bool]:
        """True/False when robots.txt gave an explicit answer, None when it was unknown."""
        if self.status == ALLOWED:
            return True
        if self.status in (DISALLOWED, FORBIDDEN):
            return False
        return None


class RobotsChecker:
    def __init__(
        self,
        session: Optional[requests.Session] = None,
        *,
        user_agent: str = "TruthLensBot/1.0",
        timeout: float = 5.0,
        cache_ttl: int = 3600,
        resolver: Optional[Callable] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.session = session or requests.Session()
        self.user_agent = user_agent
        self.agent_token = user_agent.split("/")[0].strip() or "*"
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.resolver = resolver
        self.clock = clock
        self._cache: Dict[str, Tuple[float, Optional[RobotFileParser], str]] = {}
        self._lock = threading.Lock()

    def _load(self, origin: str) -> Tuple[Optional[RobotFileParser], str]:
        now = self.clock()
        with self._lock:
            cached = self._cache.get(origin)
            if cached and now - cached[0] < self.cache_ttl:
                return cached[1], cached[2]
        parser: Optional[RobotFileParser] = None
        try:
            result = safe_get(
                f"{origin}/robots.txt", session=self.session, timeout=self.timeout,
                max_bytes=ROBOTS_MAX_BYTES, max_redirects=3, resolver=self.resolver,
                headers={"User-Agent": self.user_agent, "Accept": "text/plain,*/*;q=0.5"},
                require_html=False, raise_for_status=False,
            )
            if result.status_code == 200:
                parser = RobotFileParser()
                parser.parse(result.body.decode("utf-8", errors="replace").splitlines())
                status = "ok"
            elif result.status_code in (401, 403):
                status = FORBIDDEN
            elif 400 <= result.status_code < 500:
                status = UNAVAILABLE_ALLOWED
            else:
                status = UNREACHABLE
        except FetchHTTPError:
            status = UNREACHABLE
        except FetchError as exc:
            logger.info("robots.txt unreachable for %s: %s", origin, exc)
            status = UNREACHABLE
        with self._lock:
            self._cache[origin] = (now, parser, status)
        return parser, status

    def check(self, url: str) -> RobotsDecision:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return RobotsDecision(UNREACHABLE)
        origin = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
        parser, status = self._load(origin)
        if status in (FORBIDDEN, UNAVAILABLE_ALLOWED, UNREACHABLE):
            return RobotsDecision(status)
        allowed = parser.can_fetch(self.agent_token, url)
        delay = None
        try:
            delay = parser.crawl_delay(self.agent_token)
        except Exception:
            pass
        return RobotsDecision(ALLOWED if allowed else DISALLOWED, float(delay) if delay else None)

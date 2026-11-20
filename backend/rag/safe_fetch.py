"""SSRF-resistant HTTP fetching shared by the article fetcher, robots.txt checker and URL extraction.

Every hop (including redirects) is validated *before* a request is sent: http(s) only, no
credentials in the URL, and every address the host resolves to must be a public (global) IP.
Redirects are followed manually and bounded; response size is capped while streaming.

Residual risk: DNS can change between our check and the connection (DNS rebinding). Closing that
fully needs IP pinning, which is not implemented.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests

REDIRECT_STATUSES = (301, 302, 303, 307, 308)
HTML_TYPES = ("text/html", "application/xhtml+xml")


class FetchError(Exception):
    """Base class; subclasses map to ``fetch_status`` values."""
    status = "ERROR"


class UnsafeURLError(FetchError, ValueError):
    status = "UNSAFE_URL"


class UnresolvableHostError(UnsafeURLError):
    """The host name did not resolve. Not an attack; reported as a DNS problem, not as unsafe."""
    status = "DNS_ERROR"


class FetchTimeout(FetchError):
    status = "TIMEOUT"


class FetchTooLarge(FetchError):
    status = "TOO_LARGE"


class NonHtmlContent(FetchError):
    status = "NON_HTML"


class RedirectLimit(FetchError):
    status = "REDIRECT_LIMIT"


class FetchConnectionError(FetchError):
    status = "CONNECTION_ERROR"


class FetchHTTPError(FetchError):
    status = "HTTP_ERROR"

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    body: bytes
    content_type: str
    redirects: int


def validate_public_http_url(url: str, resolver: Optional[Callable] = None) -> str:
    """Raise ``UnsafeURLError`` unless ``url`` is an http(s) URL whose host resolves only to public IPs.
    Returns the host name."""
    resolver = resolver or socket.getaddrinfo
    try:
        parsed = urlparse((url or "").strip())
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURLError(f"malformed URL: {exc}") from exc
    if parsed.scheme.lower() not in ("http", "https"):
        raise UnsafeURLError(f"unsupported scheme: {parsed.scheme or 'none'}")
    if parsed.username or parsed.password:
        raise UnsafeURLError("credentials in URL are not allowed")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise UnsafeURLError("missing host")
    if host == "localhost" or host.endswith(".localhost"):
        raise UnsafeURLError("localhost is not allowed")

    def check(ip: "ipaddress._BaseAddress") -> None:
        if not ip.is_global:
            raise UnsafeURLError(f"non-public address: {ip}")

    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        literal = None  # a host name; resolve it below
    if literal is not None:
        check(literal)
        return host
    try:
        infos = resolver(host, port or (443 if parsed.scheme.lower() == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UnresolvableHostError(f"cannot resolve host: {exc}") from exc
    if not infos:
        raise UnresolvableHostError("host did not resolve")
    for info in infos:
        try:
            check(ipaddress.ip_address(info[4][0].split("%", 1)[0]))
        except ValueError as exc:
            if isinstance(exc, UnsafeURLError):
                raise
            raise UnsafeURLError(f"unparseable address from resolver: {info[4][0]}") from exc
    return host


def safe_get(
    url: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 8.0,
    max_bytes: int = 2_000_000,
    max_redirects: int = 3,
    headers: Optional[Dict[str, str]] = None,
    resolver: Optional[Callable] = None,
    on_hop: Optional[Callable[[str], None]] = None,
    require_html: bool = True,
    raise_for_status: bool = True,
) -> FetchResult:
    """GET ``url`` safely. ``on_hop(next_url)`` is called for every redirect target before it is
    followed (the article fetcher uses it to re-check robots.txt when the host changes)."""
    session = session or requests.Session()
    current = url
    redirects = 0
    while True:
        validate_public_http_url(current, resolver)
        response = None
        try:
            try:
                response = session.get(current, headers=headers, timeout=timeout,
                                       allow_redirects=False, stream=True)
            except requests.Timeout as exc:
                raise FetchTimeout(str(exc)) from exc
            except requests.RequestException as exc:
                raise FetchConnectionError(str(exc)) from exc

            status = response.status_code
            if status in REDIRECT_STATUSES and response.headers.get("Location"):
                redirects += 1
                if redirects > max_redirects:
                    raise RedirectLimit(f"more than {max_redirects} redirects")
                current = urljoin(current, response.headers["Location"])
                if on_hop:
                    on_hop(current)
                continue

            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if status >= 400 and raise_for_status:
                raise FetchHTTPError(status)
            if status < 400 and require_html and content_type and content_type not in HTML_TYPES:
                raise NonHtmlContent(content_type)

            chunks: List[bytes] = []
            total = 0
            try:
                for part in response.iter_content(chunk_size=8192):
                    total += len(part)
                    if total > max_bytes:
                        raise FetchTooLarge(f"response larger than {max_bytes} bytes")
                    chunks.append(part)
            except requests.Timeout as exc:
                raise FetchTimeout(str(exc)) from exc
            except requests.RequestException as exc:
                raise FetchConnectionError(str(exc)) from exc
            return FetchResult(url, current, status, b"".join(chunks), content_type, redirects)
        finally:
            if response is not None:
                response.close()

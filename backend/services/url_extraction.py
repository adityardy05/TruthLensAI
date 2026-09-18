from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests
try:
    import trafilatura
except ImportError:  # optional until URL verification is invoked
    trafilatura = None


class UrlExtractionError(ValueError):
    pass


def _validate_public_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UrlExtractionError("A complete http or https URL is required.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise UrlExtractionError("Private or local URLs cannot be fetched.")
    except socket.gaierror as error:
        raise UrlExtractionError("The URL hostname could not be resolved.") from error
    return url


def extract_article_text(url: str, timeout: int = 20, max_bytes: int = 2_000_000) -> str:
    """Fetch a public article with bounded size and extract readable text."""
    if trafilatura is None:
        raise UrlExtractionError("URL extraction dependency is not installed.")
    url = _validate_public_http_url(url)
    try:
        response = requests.get(url, timeout=timeout, stream=True, allow_redirects=True,
                                headers={"User-Agent": "TruthLens/1.0"})
        response.raise_for_status()
        final_url = _validate_public_http_url(response.url)
        payload = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise UrlExtractionError("The URL response is too large to analyze.")
    except requests.RequestException as error:
        raise UrlExtractionError("Unable to retrieve the supplied URL.") from error
    text = trafilatura.extract(bytes(payload), url=final_url, include_links=False) or ""
    if not text.strip():
        raise UrlExtractionError("No readable article text could be extracted from this URL.")
    return text.strip()

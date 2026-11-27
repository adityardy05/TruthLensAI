from __future__ import annotations

from backend.rag.safe_fetch import (
    FetchError,
    FetchTooLarge,
    UnsafeURLError,
    safe_get,
    validate_public_http_url,
)

try:
    import trafilatura
except ImportError:  # optional until URL verification is invoked
    trafilatura = None


class UrlExtractionError(ValueError):
    pass


def _validate_public_http_url(url: str) -> str:
    try:
        validate_public_http_url(url)
    except UnsafeURLError as error:
        text = str(error)
        if "non-public" in text or "localhost" in text:
            raise UrlExtractionError("Private or local URLs cannot be fetched.") from error
        if "resolve" in text:
            raise UrlExtractionError("The URL hostname could not be resolved.") from error
        raise UrlExtractionError("A complete http or https URL is required.") from error
    return url


def extract_article_text(url: str, timeout: int = 20, max_bytes: int = 2_000_000) -> str:
    """Fetch a public article with bounded size and extract readable text.

    Uses the shared SSRF-safe fetcher: every redirect hop is validated *before* it is requested.
    (This is a user-supplied URL, not a crawl, so robots.txt is not consulted.)
    """
    if trafilatura is None:
        raise UrlExtractionError("URL extraction dependency is not installed.")
    url = _validate_public_http_url(url)
    try:
        result = safe_get(url, timeout=timeout, max_bytes=max_bytes, max_redirects=5,
                          headers={"User-Agent": "TruthLens/1.0"})
    except UnsafeURLError as error:
        raise UrlExtractionError("Private or local URLs cannot be fetched.") from error
    except FetchTooLarge as error:
        raise UrlExtractionError("The URL response is too large to analyze.") from error
    except FetchError as error:
        raise UrlExtractionError("Unable to retrieve the supplied URL.") from error
    text = trafilatura.extract(result.body, url=result.final_url, include_links=False) or ""
    if not text.strip():
        raise UrlExtractionError("No readable article text could be extracted from this URL.")
    return text.strip()


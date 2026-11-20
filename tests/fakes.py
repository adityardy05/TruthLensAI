"""Offline test doubles: no network, no model download."""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from backend.rag.embedding import EMBED_DIM, EmbeddingError


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (L2-normalised, 384-dim).
    Texts that share words get a positive cosine similarity."""

    model_name = "all-MiniLM-L6-v2"
    dim = EMBED_DIM

    def __init__(self, available: bool = True, fail: bool = False):
        self._available, self.fail = available, fail
        self.calls: List[int] = []

    @property
    def available(self) -> bool:
        return self._available

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.fail:
            raise EmbeddingError("boom")
        self.calls.append(len(texts))
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class FakeResponse:
    def __init__(self, status: int = 200, body: bytes | str = b"", headers: Optional[dict] = None, json_data=None):
        self.status_code = status
        self._body = body.encode() if isinstance(body, str) else body
        self.headers = headers if headers is not None else {"Content-Type": "text/html; charset=utf-8"}
        self._json = json_data
        self.closed = False

    def iter_content(self, chunk_size: int = 8192):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def close(self):
        self.closed = True


class FakeSession:
    """``routes`` maps an exact URL to a FakeResponse, an Exception (raised), or a callable
    ``(url, params) -> FakeResponse``. Unknown URLs raise ConnectionError."""

    def __init__(self, routes: Optional[Dict[str, object]] = None):
        import requests
        self._requests = requests
        self.routes = routes or {}
        self.calls: List[tuple] = []
        self.headers: dict = {}

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True, stream=False):
        self.calls.append((url, params))
        route = self.routes.get(url)
        if route is None:
            raise self._requests.ConnectionError(f"no route for {url}")
        if isinstance(route, Exception):
            raise route
        return route(url, params) if callable(route) else route


def public_resolver(mapping: Optional[Dict[str, str]] = None) -> Callable:
    """Fake DNS: every host resolves to a public IP unless ``mapping`` says otherwise."""
    mapping = mapping or {}

    def resolve(host, port, type=None):
        return [(2, 1, 6, "", (mapping.get(host, "93.184.216.34"), port))]

    return resolve


def article_html(title: str, paragraphs: List[str]) -> str:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f"<html><head><title>{title}</title></head><body><article><h1>{title}</h1>{body}</article></body></html>"


LONG_PARAGRAPHS = [
    "Health officials in Geneva said on Tuesday that there is no scientific evidence linking the vaccine to "
    "infertility, and a review of 14 clinical studies covering more than 40,000 participants found no difference "
    "in pregnancy rates between vaccinated and unvaccinated women.",
    "The World Health Organization has repeatedly stated that the claim circulating on social media is false. "
    "Researchers at three universities independently analysed fertility clinic records and reached the same result.",
    "Dr. Maria Santos, an epidemiologist who led one of the studies, explained that misinformation often spreads "
    "faster than corrected facts, which is why public health agencies publish regular fact-checking bulletins.",
    "The agency advised people to consult their doctor with any questions and to rely on information published "
    "by national health ministries and peer reviewed journals instead of forwarded messages.",
]

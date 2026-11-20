"""Small deterministic text helpers shared by the retrieval modules."""
from __future__ import annotations

import hashlib
import ipaddress
import re
import unicodedata
from typing import Iterable, List, Sequence, Set
from urllib.parse import parse_qsl, urlencode, urlparse

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its just me more most my no nor not now
    of off on once only or other our out over own same she should so some such than that the their them
    then there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your says said say
    """.split()
)

_TRACKING_PARAM = re.compile(
    r"^(utm_.*|fbclid|gclid|dclid|msclkid|mc_cid|mc_eid|igshid|ocid|cmpid)$", re.IGNORECASE
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u0964])\s+|\n+")


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens.

    Splits on whitespace, punctuation, symbols and control characters but keeps
    combining marks, so Devanagari and Telugu words are not cut at vowel signs
    (Python's ``\\w`` would split them).
    """
    if not text:
        return []
    tokens: List[str] = []
    buf: List[str] = []
    for ch in text.lower():
        if unicodedata.category(ch)[0] in "PSZC":
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def content_tokens(text: str) -> List[str]:
    """Tokens with filler words and one-letter noise removed (digits are kept)."""
    return [t for t in tokenize(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s and s.strip()]


def canonical_url(url: str) -> str:
    """Normalised form of a URL used ONLY as a de-duplication key (never fetched).

    Returns "" for anything that is not a valid http(s) URL.
    """
    try:
        parsed = urlparse((url or "").strip())
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return ""
    if parsed.scheme.lower() not in ("http", "https") or not host:
        return ""
    if host.startswith("www."):
        host = host[4:]
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = parsed.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                       if not _TRACKING_PARAM.match(k)])
    return f"https://{netloc}{path}" + (f"?{query}" if query else "")


def domain_of(url: str) -> str:
    try:
        host = (urlparse((url or "").strip()).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def is_internal_host(host: str) -> bool:
    """True for localhost and non-public IP literals (cheap, no DNS). Public search engines never
    return such URLs, so they are dropped from search results."""
    host = (host or "").strip().lower().strip("[]")
    if not host or host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return not ipaddress.ip_address(host.split("%", 1)[0]).is_global
    except ValueError:
        return False


def normalized_hash(text: str) -> str:
    return hashlib.sha1(" ".join(tokenize(text)).encode("utf-8")).hexdigest()


def shingles(tokens: Sequence[str], k: int = 5) -> Set[str]:
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: Set[str], b: Set[str]) -> float:
    """Share of the *smaller* set that appears in the other."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))

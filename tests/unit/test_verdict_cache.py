import sqlite3
import time

from backend.services.verdict_cache import VerdictCache


def test_cache_uses_sha256_of_exact_normalized_claim(tmp_path):
    cache = VerdictCache(tmp_path / "verdicts.sqlite3", ttl_days=7)
    response = {"verdict": "TRUE"}
    cache.set("A claim with spaces", response)
    assert cache.get(" a   CLAIM with spaces ") == response
    assert cache.get("A different claim") is None


def test_cache_expires_after_configured_ttl(tmp_path):
    path = tmp_path / "verdicts.sqlite3"
    cache = VerdictCache(path, ttl_days=7)
    cache.set("claim", {"verdict": "FALSE"})
    key = cache.claim_hash("claim")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE verdict_cache SET expires_at = ?", (int(time.time()) - 1,))
    assert cache.get("claim") is None


def test_zero_ttl_disables_verdict_cache(tmp_path):
    cache = VerdictCache(tmp_path / "verdicts.sqlite3", ttl_days=0)
    cache.set("claim", {"verdict": "TRUE"})
    assert cache.get("claim") is None

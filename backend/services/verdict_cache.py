"""Exact-claim verdict cache with a configurable, seven-day default TTL.

Only normalized exact claim hashes are used. There is deliberately no semantic or percentage
similarity matching between claims.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import closing
import unicodedata
from pathlib import Path
from typing import Any, Optional


class VerdictCache:
    def __init__(self, path: str | Path, ttl_days: int = 7):
        self.path = Path(path)
        self.ttl_days = max(0, int(ttl_days))
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS verdict_cache (
                    claim_hash TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_verdict_cache_expires ON verdict_cache(expires_at)")

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    @staticmethod
    def claim_hash(claim: str) -> str:
        normalized = " ".join(unicodedata.normalize("NFKC", claim or "").casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, claim: str) -> Optional[dict[str, Any]]:
        if self.ttl_days <= 0:
            return None
        key = self.claim_hash(claim)
        now = int(time.time())
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT response_json, expires_at FROM verdict_cache WHERE claim_hash = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            if int(row[1]) <= now:
                db.execute("DELETE FROM verdict_cache WHERE claim_hash = ?", (key,))
                return None
            try:
                value = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                db.execute("DELETE FROM verdict_cache WHERE claim_hash = ?", (key,))
                return None
            return value if isinstance(value, dict) else None

    def set(self, claim: str, response: dict[str, Any]) -> None:
        if self.ttl_days <= 0:
            return
        now = int(time.time())
        expires = now + self.ttl_days * 24 * 60 * 60
        payload = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        key = self.claim_hash(claim)
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                """INSERT INTO verdict_cache(claim_hash, response_json, created_at, expires_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(claim_hash) DO UPDATE SET
                     response_json=excluded.response_json,
                     created_at=excluded.created_at,
                     expires_at=excluded.expires_at""",
                (key, payload, now, expires),
            )
            db.execute("DELETE FROM verdict_cache WHERE expires_at <= ?", (now,))

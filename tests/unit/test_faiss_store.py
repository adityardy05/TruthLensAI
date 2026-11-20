import hashlib
import json
import pickle
from datetime import datetime, timedelta, timezone

import faiss
import numpy as np
import pytest

from backend.rag.embedding import EMBED_DIM
from backend.rag.faiss_store import FaissEvidenceStore
from tests.fakes import FakeEmbedder


def store(tmp_path, ttl=7, embedder=None):
    return FaissEvidenceStore(tmp_path / "i.faiss", tmp_path / "m.pkl", embedder or FakeEmbedder(), ttl_days=ttl)


def chunk(text, url="https://a.com/1", etype="extracted", idx=0):
    return {"text": text, "url": url, "title": "T", "domain": "a.com", "published_date": "2026-10-01",
            "date_source": "search", "article_id": "a1", "chunk_id": f"a1:{idx}", "extraction_method": "trafilatura",
            "evidence_type": etype, "author": None, "search_query": "q"}


def test_fresh_store_starts_empty_and_creates_manifest(tmp_path):
    s = store(tmp_path)
    assert s.available and s.count == 0 and s.search_text("anything") == []
    manifest = json.loads((tmp_path / "i.faiss.manifest.json").read_text())
    assert manifest["embedding_model"] == "all-MiniLM-L6-v2" and manifest["dim"] == EMBED_DIM and manifest["ntotal"] == 0


def test_add_and_search_roundtrip_with_cosine_semantics(tmp_path):
    s = store(tmp_path)
    assert s.add_chunks([chunk("whiskey does not cure covid"), chunk("pasta recipe with tomato", idx=1)]) == 2
    hits = s.search_text("whiskey does not cure covid", 5)
    assert hits[0]["text"] == "whiskey does not cure covid"
    assert hits[0]["embedding_similarity"] == pytest.approx(1.0, abs=1e-5)       # identical text -> cosine 1
    assert 0.0 <= hits[1]["embedding_similarity"] < 0.3
    assert hits[0]["url"] == "https://a.com/1" and hits[0]["chunk_id"] == "a1:0"


def test_index_and_metadata_stay_aligned_and_persist(tmp_path):
    s = store(tmp_path)
    s.add_chunks([chunk(f"fact number {i}", idx=i) for i in range(5)])
    reloaded = store(tmp_path)
    assert reloaded.count == 5 and reloaded.index.ntotal == 5
    assert reloaded.search_text("fact number 3", 1)[0]["text"] == "fact number 3"


def test_sha256_content_hash_deduplicates_across_source_urls_and_updates_provenance(tmp_path):
    s = store(tmp_path)
    first = chunk("same evidence text")
    second = chunk("same evidence text", url="https://b.example/story")
    assert s.add_chunks([first]) == 1
    assert s.add_chunks([second]) == 0
    assert s.count == 1
    entry = next(item for item in s.metadata if item.get("kind") == "chunk")
    expected = hashlib.sha256("same evidence text".encode("utf-8")).hexdigest()
    assert entry["content_hash"] == expected
    assert {source["url"] for source in entry["source_records"]} == {
        "https://a.com/1", "https://b.example/story"
    }


def test_snippets_empty_text_and_missing_url_are_never_cached(tmp_path):
    s = store(tmp_path)
    assert s.add_chunks([chunk("a snippet", etype="snippet"), chunk("  "), chunk("no url", url="")]) == 0
    assert s.count == 0


def test_cached_embeddings_are_reused_not_recomputed(tmp_path):
    emb = FakeEmbedder()
    s = store(tmp_path, embedder=emb)
    c = chunk("reuse my vector")
    c["_embedding"] = emb.encode(["reuse my vector"])[0]
    emb.calls.clear()
    s.add_chunks([c])
    assert emb.calls == []


def test_evidence_chunks_are_permanent_beyond_the_seven_day_anchor_ttl(tmp_path):
    s = store(tmp_path, ttl=7)
    s.add_chunks([chunk("old evidence about covid"), chunk("another old evidence item", idx=1)])
    for entry in s.metadata:
        entry["cached_at"] = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    assert len(s.search_text("evidence about covid", 5)) == 2
    assert s.prune_expired() == 0
    assert s.count == 2 and s.index.ntotal == 2
    assert store(tmp_path).count == 2


def test_zero_ttl_disables_expiry(tmp_path):
    s = store(tmp_path, ttl=0)
    s.add_chunks([chunk("evidence")])
    s.metadata[0]["cached_at"] = "2001-01-01T00:00:00+00:00"
    assert len(s.search_text("evidence", 1)) == 1


def test_legacy_index_without_manifest_is_quarantined_not_deleted(tmp_path):
    legacy = faiss.IndexFlatIP(EMBED_DIM)
    legacy.add(np.ones((3, EMBED_DIM), dtype="float32"))
    faiss.write_index(legacy, str(tmp_path / "i.faiss"))
    (tmp_path / "m.pkl").write_bytes(pickle.dumps([{"text": "liar"}] * 3))
    s = store(tmp_path)
    assert s.available and s.count == 0 and "manifest" in s.last_reset_reason
    assert len(list(tmp_path.glob("i.faiss.legacy-*"))) == 1 and len(list(tmp_path.glob("m.pkl.legacy-*"))) == 1


def test_misaligned_index_and_metadata_is_quarantined(tmp_path):
    s = store(tmp_path)
    s.add_chunks([chunk("one"), chunk("two", idx=1)])
    (tmp_path / "m.pkl").write_bytes(pickle.dumps([{"text": "only one"}]))
    s2 = store(tmp_path)
    assert s2.count == 0 and "misaligned" in s2.last_reset_reason


def test_embedding_model_mismatch_is_quarantined(tmp_path):
    s = store(tmp_path)
    s.add_chunks([chunk("one")])
    path = tmp_path / "i.faiss.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["embedding_model"] = "some-other-model"
    path.write_text(json.dumps(manifest))
    assert store(tmp_path).count == 0


def test_corrupt_files_do_not_crash(tmp_path):
    (tmp_path / "i.faiss").write_bytes(b"garbage")
    (tmp_path / "m.pkl").write_bytes(b"garbage")
    (tmp_path / "i.faiss.manifest.json").write_text("{not json")
    assert store(tmp_path).available


def test_unconfigured_store_is_unavailable_but_safe():
    s = FaissEvidenceStore(None, None, FakeEmbedder())
    assert not s.available and s.search_text("x") == [] and s.add_chunks([chunk("x")]) == 0


def test_search_survives_embedder_failure(tmp_path):
    s = store(tmp_path)
    s.add_chunks([chunk("something")])
    s.embedder = FakeEmbedder(fail=True)
    assert s.search_text("something") == []


# ------------------------------------------------------------- query anchors
def test_query_anchor_links_to_its_chunks_and_matches_the_claim_not_the_passage(tmp_path):
    emb = FakeEmbedder()
    s = store(tmp_path, embedder=emb)
    chunks = [chunk("long passage about hospitals studies and participants and outcomes " * 3, idx=i, url=f"https://a.com/{i}") for i in range(3)]
    q = "vaccine causes infertility"
    assert s.add_chunks(chunks, query=q, query_vector=emb.encode([q])[0]) == 3
    assert s.count == 4                                  # 3 chunks + 1 anchor
    hits = s.search(emb.encode([q])[0], 10)
    anchor = next(h for h in hits if h["kind"] == "query")
    assert anchor["embedding_similarity"] == pytest.approx(1.0, abs=1e-5) and len(anchor["chunk_keys"]) == 3
    assert max(h["embedding_similarity"] for h in hits if h["kind"] == "chunk") < 0.5   # claim vs long passage is low
    linked = s.get_chunks_by_keys(anchor["chunk_keys"])
    assert len(linked) == 3 and all(c["kind"] == "chunk" and c["embedding_similarity"] is None for c in linked)


def test_same_query_extends_one_anchor_instead_of_duplicating(tmp_path):
    emb = FakeEmbedder()
    s = store(tmp_path, embedder=emb)
    qv = emb.encode(["claim text"])[0]
    s.add_chunks([chunk("first passage", idx=0)], query="claim text", query_vector=qv)
    s.add_chunks([chunk("first passage", idx=0), chunk("second passage", idx=1, url="https://a.com/2")], query="claim text", query_vector=qv)
    anchors = [m for m in s.metadata if m.get("kind") == "query"]
    assert len(anchors) == 1 and len(anchors[0]["chunk_keys"]) == 2 and s.count == 3
    assert store(tmp_path, embedder=emb).count == 3      # persisted and re-loadable


def test_anchor_without_cacheable_chunks_is_not_created(tmp_path):
    emb = FakeEmbedder()
    s = store(tmp_path, embedder=emb)
    assert s.add_chunks([chunk("only a snippet", etype="snippet")], query="q", query_vector=emb.encode(["q"])[0]) == 0
    assert s.count == 0


def test_expired_anchor_is_ignored_and_pruned_but_its_evidence_is_kept(tmp_path):
    emb = FakeEmbedder()
    s = store(tmp_path, ttl=7, embedder=emb)
    qv = emb.encode(["old claim"])[0]
    s.add_chunks([chunk("passage")], query="old claim", query_vector=qv)
    for m in s.metadata:
        m["cached_at"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    assert not any(hit.get("kind") == "query" for hit in s.search(qv, 5))
    assert s.prune_expired() == 1
    assert s.count == 1 and s.index.ntotal == 1
    assert store(tmp_path, embedder=emb).count == 1

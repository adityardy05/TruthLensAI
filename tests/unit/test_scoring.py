from datetime import datetime, timedelta

import pytest

from backend.core.config import RetrievalSettings
from backend.rag.domain_credibility import HybridDomainCredibility
from backend.rag.scoring import compute_rd, compute_recency_score, quality_score, score_chunks, select_top_k
from tests.fakes import FakeEmbedder

S = RetrievalSettings()
NOW = datetime(2026, 11, 1)
CRED = HybridDomainCredibility()


def chunk(i=1, text=None, url=None, domain=None, etype="extracted", date="2026-10-25", aid=None, group=None, **kw):
    c = {"chunk_id": f"c{i}", "article_id": aid or f"a{i}", "url": url or f"https://site{i}.com/p", "domain": domain or f"site{i}.com",
         "title": "t", "text": text or "Health officials said the vaccine does not cause infertility. A study of 40,000 people confirmed it.",
         "evidence_type": etype, "published_date": date, "cleanup_flags": [], "retrieval_rank": i,
         "syndication_group_id": group or aid or f"a{i}", "origin": "searxng", "source_type": "web"}
    c.update(kw)
    return c


# ---------------------------------------------------------------- weights / R(d)
def test_default_weights():
    assert (S.weight_relevance, S.weight_quality, S.weight_credibility, S.weight_recency) == (0.40, 0.20, 0.25, 0.15)
    assert S.relevance_weight_embedding == 0.70 and S.relevance_weight_bm25 == 0.30
    assert S.weight_relevance + S.weight_quality + S.weight_credibility + S.weight_recency == pytest.approx(1.0)


def test_rd_formula_and_bounds():
    assert compute_rd(0.5, 0.8, 0.9, 1.0, S) == pytest.approx(0.4 * 0.5 + 0.2 * 0.8 + 0.25 * 0.9 + 0.15 * 1.0, abs=1e-4)
    assert compute_rd(0, 0, 0, 0, S) == 0.0 and compute_rd(1, 1, 1, 1, S) == 1.0
    assert compute_rd(2, 2, 2, 2, S) == 1.0   # clamped


# -------------------------------------------------------------------- recency
def test_recency_milestones():
    day = lambda n: (NOW - timedelta(days=n)).strftime("%Y-%m-%d")
    assert compute_recency_score(day(10), now=NOW) == 1.0
    assert compute_recency_score(day(365), now=NOW) == pytest.approx(0.6, abs=0.01)
    assert compute_recency_score(day(1825), now=NOW) == pytest.approx(0.1, abs=0.01)
    assert compute_recency_score(day(3000), now=NOW) == 0.1


@pytest.mark.parametrize("value", [None, "", "not a date", "2026-13-45", "2099-01-01", 12345])
def test_missing_invalid_or_future_dates_get_the_undated_score_not_a_high_one(value):
    assert compute_recency_score(value, undated=0.4, now=NOW) == 0.4


def test_timezone_and_format_variants():
    assert compute_recency_score("2026-10-30T12:00:00+05:30", now=NOW) == 1.0
    assert compute_recency_score("2026-10-30T12:00:00Z", now=NOW) == 1.0
    assert compute_recency_score("October 30, 2026", now=NOW) == 1.0


# -------------------------------------------------------------------- quality
def test_snippet_quality_is_lower_than_extracted_quality():
    text = "Health officials said the vaccine does not cause infertility. A study of 40,000 people confirmed it."
    assert quality_score(chunk(text=text, etype="snippet")) < quality_score(chunk(text=text, etype="extracted")) - 0.2


def test_quality_does_not_reward_length_beyond_adequacy_and_penalises_cleanup():
    base = "Officials in Geneva said the study of 40,000 people found nothing. Researchers confirmed it."
    medium, huge = chunk(text=base * 4), chunk(text=base * 40)
    assert quality_score(medium) == pytest.approx(quality_score(huge), abs=0.05)
    cleaned = chunk(text=base * 4, cleanup_flags=["boilerplate"])
    assert quality_score(cleaned) < quality_score(medium)
    assert 0.0 <= quality_score(chunk(text="x")) <= 1.0


# ----------------------------------------------------------------- score_chunks
def test_score_chunks_sets_all_components_in_range():
    chunks = [chunk(1, url="https://www.who.int/a", domain="who.int"),
              chunk(2, text="Completely unrelated cooking recipe with pasta and salt.", url="http://spam24.xyz/x", domain="spam24.xyz", date=None)]
    degraded = score_chunks("vaccine infertility claim", chunks, FakeEmbedder(), CRED, S, now=NOW)
    assert degraded == []
    for c in chunks:
        for k in ("bm25_score", "cosine_score", "relevance_score", "quality_score", "domain_credibility",
                  "credibility_source", "recency_score", "combined_reliability"):
            assert k in c
        assert all(0.0 <= c[k] <= 1.0 for k in ("bm25_score", "cosine_score", "relevance_score", "quality_score",
                                                "domain_credibility", "recency_score", "combined_reliability"))
    good, bad = chunks
    assert good["domain_credibility"] > bad["domain_credibility"]
    assert good["combined_reliability"] > bad["combined_reliability"]
    assert bad["recency_score"] == 0.4          # unknown domain / undated handled
    assert good["relevance_score"] == pytest.approx(0.7 * good["cosine_score"] + 0.3 * good["bm25_score"], abs=2e-4)


def test_existing_faiss_similarity_is_not_re_embedded():
    emb = FakeEmbedder()
    chunks = [chunk(1, embedding_similarity=0.82), chunk(2)]
    score_chunks("vaccine infertility", chunks, emb, CRED, S, now=NOW)
    assert emb.calls == [2]            # claim + the single chunk without a similarity
    assert chunks[0]["cosine_score"] == 0.82 and "_embedding" in chunks[1] and "_embedding" not in chunks[0]


def test_embedding_failure_degrades_to_bm25():
    chunks = [chunk(1)]
    degraded = score_chunks("vaccine infertility", chunks, FakeEmbedder(fail=True), CRED, S, now=NOW)
    assert degraded == ["embedder_unavailable"]
    assert chunks[0]["relevance_score"] == pytest.approx(chunks[0]["bm25_score"], abs=1e-3)


# ------------------------------------------------------------------- selection
def scored(n, **kw):
    out = []
    for i in range(1, n + 1):
        c = chunk(i, **kw)
        c["combined_reliability"] = round(1 - i * 0.05, 4)
        out.append(c)
    return out


def test_exactly_five_when_five_or_more_candidates():
    selected, info = select_top_k(scored(12), S)
    assert len(selected) == 5 and not info["insufficient_evidence"]
    assert [c["retrieval_rank"] for c in selected] == [1, 2, 3, 4, 5]
    assert selected == sorted(selected, key=lambda c: -c["combined_reliability"])


def test_fewer_than_five_is_reported_never_padded():
    selected, info = select_top_k(scored(3), S)
    assert len(selected) == 3 and info["insufficient_evidence"] and "3" in info["insufficient_reason"]
    assert select_top_k([], S)[0] == []


def test_low_quality_items_are_kept_when_needed_so_team_b_sees_weak_evidence():
    items = scored(5)
    for c, q in zip(items, ("extracted", "extracted", "extracted", "snippet", "snippet")):
        c["evidence_type"] = q
    items[3]["combined_reliability"] = items[4]["combined_reliability"] = 0.2
    selected, _ = select_top_k(items, S)
    assert [c["evidence_type"] for c in selected].count("snippet") == 2


def test_same_syndicated_story_not_selected_five_times():
    copies = [chunk(i, aid=f"x{i}", group="syn-1", domain=f"d{i}.com") for i in range(1, 6)]
    others = [chunk(10 + i, aid=f"o{i}", domain=f"o{i}.com") for i in range(4)]
    for c, r in zip(copies + others, [0.9, 0.89, 0.88, 0.87, 0.86, 0.5, 0.49, 0.48, 0.47]):
        c["combined_reliability"] = r
    selected, info = select_top_k(copies + others, S)
    assert sum(1 for c in selected if c["syndication_group_id"] == "syn-1") == 2
    assert len(selected) == 5 and info["independent_source_count"] >= 4


def test_caps_are_relaxed_rather_than_returning_fewer():
    items = [chunk(i, aid="same", domain="same.com", group="same") for i in range(1, 7)]
    for i, c in enumerate(items):
        c["combined_reliability"] = 0.9 - i * 0.01
    selected, info = select_top_k(items, S)
    assert len(selected) == 5 and info["independent_source_count"] == 1

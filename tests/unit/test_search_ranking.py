import pytest

from backend.rag.search_ranking import bm25_normalized, rank_search_results
from backend.rag.text_utils import content_tokens, tokenize
from tests.fakes import FakeEmbedder


def result(i, title, snippet, domain=None):
    return {"url": f"https://{domain or f'd{i}.com'}/p{i}", "canonical_url": f"https://{domain or f'd{i}.com'}/p{i}",
            "domain": domain or f"d{i}.com", "title": title, "snippet": snippet, "retrieval_rank": i,
            "engines": ["x"], "published_date": None, "query": "q", "retrieved_at": "t"}


def test_bm25_scores_are_bounded_and_not_forced_to_one():
    docs = ["whiskey does not cure covid", "weather forecast for tomorrow", "covid vaccine safety data"]
    scores = bm25_normalized("Does whiskey cure COVID", docs)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[2] > scores[1] == 0.0
    assert scores[0] < 1.0  # normalised by the theoretical bound, not by the pool maximum


def test_bm25_is_stable_on_tiny_corpora():
    assert bm25_normalized("covid", ["covid facts"])[0] > 0   # rank-bm25 gives <= 0 here
    assert bm25_normalized("covid", []) == []
    assert bm25_normalized("the of and", ["covid facts"]) == [0.0]  # only stopwords


def test_bm25_normalisation_is_comparable_across_pools():
    a = bm25_normalized("whiskey covid", ["whiskey covid cure myth", "other text here", "more other text"])[0]
    b = bm25_normalized("whiskey covid", ["whiskey covid cure myth", "whiskey covid cure myth", "whiskey covid cure myth"])[0]
    assert abs(a - b) < 0.35  # not rescaled to 1.0 per pool


def test_tokenizer_keeps_indic_words_whole():
    assert tokenize("कोविड वैक्सीन है।") == ["कोविड", "वैक्सीन", "है"]
    assert content_tokens("The vaccine is safe") == ["vaccine", "safe"]


def test_hybrid_score_matches_formula_and_keeps_components():
    results = [result(1, "Whiskey does not cure COVID", "WHO says alcohol does not cure covid"),
              result(2, "Cooking pasta", "Boil water and add salt")]
    selected, scored, degraded = rank_search_results("whiskey cure covid", results, FakeEmbedder(), top_n=10)
    assert degraded == []
    top = selected[0]
    assert top["title"].startswith("Whiskey")
    assert top["search_relevance"] == pytest.approx(0.7 * top["embedding_similarity"] + 0.3 * top["bm25_norm"], abs=2e-4)
    assert selected[0]["search_relevance"] > selected[1]["search_relevance"]
    assert top["url"] and top["snippet"] and top["query"]  # original metadata preserved


def test_top_ten_selection():
    results = [result(i, f"covid item {i}", f"covid text {i}") for i in range(25)]
    selected, scored, _ = rank_search_results("covid", results, FakeEmbedder(), top_n=10)
    assert len(selected) == 10 and len(scored) == 25


def test_domain_cap_with_backfill():
    results = [result(i, "covid claim fact", "covid fact check", domain="same.com") for i in range(5)]
    results += [result(10 + i, "covid claim", "weak", domain=f"o{i}.com") for i in range(3)]
    selected, _, _ = rank_search_results("covid claim fact", results, FakeEmbedder(), top_n=6, max_per_domain=2)
    assert len(selected) == 6
    first_four = [r["domain"] for r in selected[:4]]
    assert first_four.count("same.com") <= 2   # cap honoured while alternatives exist
    only_same = [result(i, "covid", "covid", domain="same.com") for i in range(5)]
    relaxed, _, _ = rank_search_results("covid", only_same, FakeEmbedder(), top_n=4, max_per_domain=2)
    assert len(relaxed) == 4                  # cap relaxed rather than returning fewer


@pytest.mark.parametrize("embedder", [FakeEmbedder(available=False), FakeEmbedder(fail=True)])
def test_degrades_to_bm25_only_when_embeddings_unavailable(embedder):
    results = [result(1, "whiskey covid cure", "covid whiskey"), result(2, "pasta", "salt")]
    selected, _, degraded = rank_search_results("whiskey covid", results, embedder)
    assert "embedder_unavailable" in degraded
    assert selected[0]["search_relevance"] == pytest.approx(selected[0]["bm25_norm"], abs=1e-3)


def test_empty_input():
    assert rank_search_results("x", [], FakeEmbedder()) == ([], [], [])

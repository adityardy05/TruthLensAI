from backend.rag.query_builder import build_queries, clean_claim


def test_clean_claim_strips_noise():
    raw = "FORWARDED AS RECEIVED ❤️ Check https://fake.example/x COVID   vaccine\u200b causes infertility!! please share"
    out = clean_claim(raw)
    assert "http" not in out and "❤" not in out and "\u200b" not in out
    assert "please share" not in out.lower() and "forwarded" not in out.lower()
    assert "COVID vaccine causes infertility" in out


def test_english_claim_gives_single_query():
    assert build_queries("Whiskey cures COVID-19", None, "en") == ["Whiskey cures COVID-19"]


def test_non_english_claim_adds_original_language_query():
    queries = build_queries("Vaccine causes infertility", "टीका बांझपन का कारण बनता है", "hi")
    assert queries == ["Vaccine causes infertility", "टीका बांझपन का कारण बनता है"]


def test_identical_queries_are_deduplicated_and_capped():
    assert build_queries("same text", "same text", "hi") == ["same text"]
    assert len(build_queries("a b", "c d", "hi", max_queries=1)) == 1


def test_very_long_input_is_shortened_to_a_usable_query():
    article = "The government announced a new policy about taxes and public spending today. " * 100
    query = build_queries(article, None, "en", max_chars=200)[0]
    assert 0 < len(query) <= 200


def test_empty_input_gives_no_queries():
    assert build_queries("   ", None, "en") == []

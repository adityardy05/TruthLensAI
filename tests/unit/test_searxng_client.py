import pytest
import requests

from backend.rag.searxng_client import SearXNGClient, SearXNGError, normalize_result
from tests.fakes import FakeResponse, FakeSession

BASE = "http://searx.test"
URL = f"{BASE}/search"


def raw(i, **over):
    item = {"url": f"https://site{i}.com/a{i}", "title": f"Title {i}", "content": f"Snippet {i}",
            "engines": ["google cse"], "score": 1.0}
    item.update(over)
    return item


def client(routes, **kw):
    return SearXNGClient(BASE, session=FakeSession(routes), **kw)


def test_success_normalizes_results():
    c = client({URL: FakeResponse(json_data={"results": [raw(1), raw(2)], "unresponsive_engines": [["brave", "Suspended"]]})})
    resp = c.search("covid vaccine")
    assert [r["url"] for r in resp.results] == ["https://site1.com/a1", "https://site2.com/a2"]
    first = resp.results[0]
    for key in ("url", "canonical_url", "domain", "title", "snippet", "engines", "published_date",
                "query", "retrieval_rank", "retrieved_at"):
        assert key in first
    assert first["domain"] == "site1.com" and first["query"] == "covid vaccine"
    assert resp.unresponsive_engines == [("brave", "Suspended")]


def test_request_parameters_include_json_format_categories_and_language():
    session = FakeSession({URL: FakeResponse(json_data={"results": []})})
    SearXNGClient(BASE, session=session, categories="general,news").search("q", language="hi")
    params = session.calls[0][1]
    assert params["format"] == "json" and params["categories"] == "general,news"
    assert params["language"] == "hi" and params["pageno"] == 1


def test_english_does_not_send_language():
    session = FakeSession({URL: FakeResponse(json_data={"results": []})})
    SearXNGClient(BASE, session=session).search("q", language="en")
    assert "language" not in session.calls[0][1]


def test_empty_results():
    assert client({URL: FakeResponse(json_data={"results": []})}).search("x").results == []


@pytest.mark.parametrize("route,kind", [
    (requests.Timeout("slow"), "timeout"),
    (requests.ConnectionError("down"), "connection"),
    (FakeResponse(status=500), "http_500"),
    (FakeResponse(status=403), "forbidden"),
    (FakeResponse(status=200, json_data=None), "invalid_json"),
    (FakeResponse(status=200, json_data=["not", "an", "object"]), "invalid_json"),
])
def test_failures_raise_searxng_error(route, kind):
    with pytest.raises(SearXNGError) as exc:
        client({URL: route}).search("x")
    assert exc.value.kind == kind


def test_duplicate_urls_are_merged_with_engines_and_best_rank():
    dup = raw(1, engines=["bing news"], content="A much longer snippet for the same page")
    tracked = raw(1, url="https://www.site1.com/a1/?utm_source=x")
    c = client({URL: FakeResponse(json_data={"results": [raw(1), dup, tracked, raw(2)]})})
    results = c.search("x").results
    assert len(results) == 2
    merged = results[0]
    assert set(merged["engines"]) == {"google cse", "bing news"}
    assert merged["snippet"] == "A much longer snippet for the same page"
    assert merged["retrieval_rank"] == 1


def test_missing_fields_and_malformed_urls():
    bad = [{"url": "not a url", "title": "x"}, {"title": "no url"}, "garbage",
           {"url": "ftp://x.org/f", "title": "ftp"}, {"url": "https://ok.com/p"}]
    results = client({URL: FakeResponse(json_data={"results": bad})}).search("x").results
    assert len(results) == 1
    assert results[0]["title"] == "" and results[0]["snippet"] == "" and results[0]["engines"] == []


def test_single_engine_field_is_used_when_engines_missing():
    item = {"url": "https://a.com/x", "engine": "bing news"}
    results = client({URL: FakeResponse(json_data={"results": [item]})}).search("x").results
    assert results[0]["engines"] == ["bing news"]


def test_every_result_on_a_page_is_kept_even_beyond_target():
    page = {"results": [raw(i) for i in range(30)]}
    results = client({URL: FakeResponse(json_data=page)}, results_per_query=20).search("x").results
    assert len(results) == 30


def test_second_page_requested_only_when_needed():
    pages = {1: {"results": [raw(i) for i in range(8)]}, 2: {"results": [raw(i) for i in range(8, 16)]}}
    session = FakeSession({URL: lambda url, params: FakeResponse(json_data=pages[params["pageno"]])})
    c = SearXNGClient(BASE, session=session, max_pages=2, results_per_query=20)
    assert len(c.search("x").results) == 16
    assert [call[1]["pageno"] for call in session.calls] == [1, 2]

    full = FakeSession({URL: lambda url, params: FakeResponse(json_data={"results": [raw(i) for i in range(25)]})})
    SearXNGClient(BASE, session=full, max_pages=3, results_per_query=20).search("x")
    assert len(full.calls) == 1


def test_later_page_failure_keeps_page_one():
    def route(url, params):
        if params["pageno"] == 1:
            return FakeResponse(json_data={"results": [raw(1)]})
        raise requests.Timeout("slow")

    c = SearXNGClient(BASE, session=FakeSession({URL: route}), max_pages=2, results_per_query=20)
    assert len(c.search("x").results) == 1


def test_multiple_queries_each_get_their_own_results():
    c = client({URL: lambda url, params: FakeResponse(json_data={"results": [raw(len(params["q"]))]})})
    assert c.search("a").results[0]["query"] == "a"
    assert c.search("abc").results[0]["query"] == "abc"


def test_normalize_result_rejects_invalid():
    assert normalize_result({"url": "javascript:alert(1)"}, "q", 1, "now") is None
    assert normalize_result(None, "q", 1, "now") is None


def test_reachability_check():
    assert client({f"{BASE}/": FakeResponse(200)}).is_reachable()
    assert not client({f"{BASE}/": requests.ConnectionError("x")}).is_reachable()


def test_internal_and_private_urls_in_results_are_dropped():
    bad = [{"url": "http://127.0.0.1:8080/admin"}, {"url": "http://localhost/x"}, {"url": "http://10.0.0.7/a"},
           {"url": "http://[::1]/x"}, {"url": "http://169.254.169.254/latest"}, {"url": "https://ok.example/page"}]
    results = client({URL: FakeResponse(json_data={"results": bad})}).search("x").results
    assert [r["domain"] for r in results] == ["ok.example"]

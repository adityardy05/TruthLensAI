import requests

from backend.core.config import RetrievalSettings
from backend.rag.article_fetcher import ArticleFetcher, date_from_url
from tests.fakes import LONG_PARAGRAPHS, FakeResponse, FakeSession, article_html, public_resolver

S = RetrievalSettings(host_min_delay_seconds=0, fetch_workers=3, fetch_total_budget_seconds=5)
TXT = {"Content-Type": "text/plain"}
ALLOW = FakeResponse(200, "User-agent: *\nAllow: /", TXT)


def cand(url="https://news.example/2026/10/20/story", **over):
    c = {"url": url, "domain": "news.example", "title": "Search title", "snippet": "A short search snippet about the claim.",
         "retrieval_rank": 1, "query": "q", "engines": ["google cse"], "published_date": None, "search_relevance": 0.5}
    c.update(over)
    return c


def fetcher(routes, **kw):
    session = FakeSession(routes)
    return ArticleFetcher(S, session=session, resolver=public_resolver(), **kw), session


def routes_for(url, page, robots=ALLOW):
    host = "/".join(url.split("/")[:3])
    return {f"{host}/robots.txt": robots, url: page}


def test_successful_trafilatura_extraction():
    url = "https://news.example/2026/10/20/story"
    page = FakeResponse(200, article_html("Vaccine fertility fact check", LONG_PARAGRAPHS))
    f, _ = fetcher(routes_for(url, page))
    rec = f.fetch_one(cand(url))
    assert rec["evidence_type"] == "extracted" and rec["extraction_method"] == "trafilatura"
    assert rec["access_status"] == "OK" and rec["fetch_status"] == "OK" and rec["extraction_status"] == "OK"
    assert rec["robots_allowed"] is True and "infertility" in rec["text"]
    assert rec["title"] and rec["domain"] == "news.example" and rec["url"] == url


def test_robots_disallowed_keeps_snippet_and_does_not_fetch():
    url = "https://news.example/story"
    routes = routes_for(url, FakeResponse(200, "x"), robots=FakeResponse(200, "User-agent: *\nDisallow: /", TXT))
    f, session = fetcher(routes)
    rec = f.fetch_one(cand(url))
    assert rec["evidence_type"] == "snippet" and rec["access_status"] == "ROBOTS_DISALLOWED"
    assert rec["robots_allowed"] is False and rec["fetch_status"] == "NOT_ATTEMPTED"
    assert rec["text"] == "A short search snippet about the claim."
    assert url not in [c[0] for c in session.calls]


def test_robots_unreachable_skips_page_but_keeps_snippet():
    url = "https://news.example/story"
    f, _ = fetcher(routes_for(url, FakeResponse(200, "x"), robots=FakeResponse(503, b"", TXT)))
    rec = f.fetch_one(cand(url))
    assert rec["access_status"] == "ROBOTS_UNREACHABLE" and rec["evidence_type"] == "snippet"


def test_403_is_server_denied_not_robots_denied_and_not_retried():
    url = "https://news.example/story"
    f, session = fetcher(routes_for(url, FakeResponse(403, b"forbidden")))
    rec = f.fetch_one(cand(url))
    assert rec["access_status"] == "SERVER_DENIED" and rec["fetch_status"] == "HTTP_ERROR"
    assert rec["robots_allowed"] is True and rec["evidence_type"] == "snippet"
    assert [c[0] for c in session.calls].count(url) == 1


def test_other_http_error_and_timeout_fall_back_to_snippet():
    url = "https://news.example/story"
    f, _ = fetcher(routes_for(url, FakeResponse(500)))
    assert f.fetch_one(cand(url))["fetch_status"] == "HTTP_ERROR"
    f, _ = fetcher(routes_for(url, requests.Timeout("slow")))
    rec = f.fetch_one(cand(url))
    assert rec["fetch_status"] == "TIMEOUT" and rec["evidence_type"] == "snippet"


def test_empty_extraction_and_extractor_failure_fall_back():
    url = "https://news.example/story"
    page = FakeResponse(200, "<html><body><div>js</div></body></html>")
    f, _ = fetcher(routes_for(url, page))
    rec = f.fetch_one(cand(url))
    assert rec["extraction_status"] == "EMPTY" and rec["evidence_type"] == "snippet"

    def boom(body, u):
        raise RuntimeError("parser crash")
    f, _ = fetcher(routes_for(url, FakeResponse(200, "<html></html>")), extractor=boom)
    rec = f.fetch_one(cand(url))
    assert rec["extraction_status"] == "FAILED" and rec["evidence_type"] == "snippet"


def test_unsafe_url_is_never_fetched():
    f, session = fetcher({})
    rec = f.fetch_one(cand("http://169.254.169.254/latest"))
    assert rec["access_status"] == "UNSAFE_URL" and session.calls == []


def test_redirect_to_a_robots_blocked_domain_is_stopped():
    url = "https://news.example/story"
    routes = {
        "https://news.example/robots.txt": ALLOW, url: FakeResponse(302, headers={"Location": "https://other.example/x"}),
        "https://other.example/robots.txt": FakeResponse(200, "User-agent: *\nDisallow: /", TXT),
        "https://other.example/x": FakeResponse(200, article_html("t", LONG_PARAGRAPHS)),
    }
    f, session = fetcher(routes)
    rec = f.fetch_one(cand(url))
    assert rec["evidence_type"] == "snippet"
    assert "https://other.example/x" not in [c[0] for c in session.calls]


def test_date_sources_in_priority_order():
    assert date_from_url("https://x.com/2026/10/20/a") == "2026-10-20"
    assert date_from_url("https://x.com/2026-10-20-a") == "2026-10-20"
    assert date_from_url("https://x.com/2099/01/01/a") is None   # future
    assert date_from_url("https://x.com/2026/13/40/a") is None   # invalid
    url = "https://news.example/story"
    f, _ = fetcher(routes_for(url, FakeResponse(200, article_html("Title", LONG_PARAGRAPHS))),
                   extractor=lambda b, u: ("x" * 300, {"date": "2026-09-01", "title": "T"}))
    rec = f.fetch_one(cand(url, published_date="2026-10-01"))
    assert rec["published_date"] == "2026-10-01" and rec["date_source"] == "search"
    rec = f.fetch_one(cand(url))
    assert rec["published_date"] == "2026-09-01" and rec["date_source"] == "trafilatura"
    f2, _ = fetcher(routes_for("https://news.example/2026/10/20/s", FakeResponse(200, "<html></html>")),
                    extractor=lambda b, u: ("x" * 300, {}))
    rec = f2.fetch_one(cand("https://news.example/2026/10/20/s"))
    assert rec["published_date"] == "2026-10-20" and rec["date_source"] == "url_pattern"
    undated = fetcher(routes_for(url, FakeResponse(403)))[0].fetch_one(cand(url))
    assert undated["published_date"] is None and undated["date_source"] == "none"


def test_fetch_all_preserves_order_and_isolates_failures():
    urls = [f"https://s{i}.example/story" for i in range(4)]
    routes = {}
    for i, u in enumerate(urls):
        routes.update(routes_for(u, FakeResponse(200, article_html(f"T{i}", LONG_PARAGRAPHS)) if i != 2 else FakeResponse(500)))
    f, _ = fetcher(routes)
    out = f.fetch_all([cand(u, domain=f"s{i}.example") for i, u in enumerate(urls)])
    assert [r["url"] for r in out] == urls
    assert [r["evidence_type"] for r in out] == ["extracted", "extracted", "snippet", "extracted"]


def test_per_host_rate_limit_sleeps_between_requests():
    sleeps, now = [], [0.0]
    s = RetrievalSettings(host_min_delay_seconds=1.0, fetch_workers=1)
    session = FakeSession({**routes_for("https://a.example/1", FakeResponse(500)), "https://a.example/2": FakeResponse(500)})
    f = ArticleFetcher(s, session=session, resolver=public_resolver(), sleep=sleeps.append, clock=lambda: now[0])
    f.fetch_one(cand("https://a.example/1")); f.fetch_one(cand("https://a.example/2"))
    assert sleeps and abs(sleeps[0] - 1.0) < 1e-6


def test_unresolvable_host_is_a_dns_error_not_an_unsafe_url():
    def nxdomain(host, port, type=None):
        raise OSError("Name or service not known")
    session = FakeSession({})
    f = ArticleFetcher(S, session=session, resolver=nxdomain)
    rec = f.fetch_one(cand("https://does-not-exist.example/story"))
    assert rec["fetch_status"] == "DNS_ERROR" and rec["access_status"] == "NOT_ATTEMPTED"
    assert rec["evidence_type"] == "snippet" and session.calls == []

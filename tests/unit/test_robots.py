import requests

from backend.rag.robots import RobotsChecker
from tests.fakes import FakeResponse, FakeSession, public_resolver

TXT = {"Content-Type": "text/plain"}


def checker(routes, **kw):
    session = FakeSession(routes)
    return RobotsChecker(session, user_agent="TruthLensBot/1.0", resolver=public_resolver(), **kw), session


def test_allowed_url():
    c, _ = checker({"https://a.com/robots.txt": FakeResponse(200, "User-agent: *\nDisallow: /private/", TXT)})
    d = c.check("https://a.com/news/story")
    assert d.may_fetch and d.robots_allowed is True and d.status == "allowed"


def test_disallowed_url_and_agent_specific_rules():
    c, _ = checker({"https://a.com/robots.txt": FakeResponse(
        200, "User-agent: TruthLensBot\nDisallow: /\n\nUser-agent: *\nAllow: /", TXT)})
    d = c.check("https://a.com/news/story")
    assert not d.may_fetch and d.robots_allowed is False and d.status == "disallowed"


def test_missing_robots_txt_allows_but_is_recorded_as_unknown():
    c, _ = checker({"https://a.com/robots.txt": FakeResponse(404, b"", TXT)})
    d = c.check("https://a.com/x")
    assert d.may_fetch and d.robots_allowed is None and d.status == "unavailable_allowed"


def test_robots_forbidden_is_treated_as_disallow_all():
    for code in (401, 403):
        c, _ = checker({"https://a.com/robots.txt": FakeResponse(code, b"", TXT)})
        d = c.check("https://a.com/x")
        assert not d.may_fetch and d.robots_allowed is False


def test_server_error_or_timeout_means_unreachable_do_not_fetch():
    for route in (FakeResponse(503, b"", TXT), requests.Timeout("slow"), requests.ConnectionError("down")):
        c, _ = checker({"https://a.com/robots.txt": route})
        d = c.check("https://a.com/x")
        assert not d.may_fetch and d.status == "unreachable" and d.robots_allowed is None


def test_invalid_robots_content_is_treated_as_no_rules():
    c, _ = checker({"https://a.com/robots.txt": FakeResponse(200, "<html>not a robots file</html>\n\x00\x01", TXT)})
    assert c.check("https://a.com/x").may_fetch


def test_result_is_cached_per_origin_and_expires():
    now = [1000.0]
    c, session = checker({"https://a.com/robots.txt": FakeResponse(200, "User-agent: *\nAllow: /", TXT)},
                         cache_ttl=60, clock=lambda: now[0])
    c.check("https://a.com/one"); c.check("https://a.com/two")
    assert len(session.calls) == 1
    now[0] += 120
    c.check("https://a.com/three")
    assert len(session.calls) == 2


def test_different_domains_are_checked_independently():
    c, _ = checker({
        "https://a.com/robots.txt": FakeResponse(200, "User-agent: *\nDisallow: /", TXT),
        "https://b.com/robots.txt": FakeResponse(200, "User-agent: *\nAllow: /", TXT)})
    assert not c.check("https://a.com/x").may_fetch
    assert c.check("https://b.com/x").may_fetch


def test_crawl_delay_is_exposed():
    c, _ = checker({"https://a.com/robots.txt": FakeResponse(200, "User-agent: *\nCrawl-delay: 3", TXT)})
    assert c.check("https://a.com/x").crawl_delay == 3.0


def test_redirected_robots_txt_is_followed():
    c, _ = checker({"https://a.com/robots.txt": FakeResponse(301, headers={"Location": "https://www.a.com/robots.txt"}),
                    "https://www.a.com/robots.txt": FakeResponse(200, "User-agent: *\nDisallow: /", TXT)})
    assert not c.check("https://a.com/x").may_fetch

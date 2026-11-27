import pytest
import requests

from backend.rag.safe_fetch import (
    FetchHTTPError,
    FetchTimeout,
    FetchTooLarge,
    NonHtmlContent,
    RedirectLimit,
    UnsafeURLError,
    safe_get,
    validate_public_http_url,
)
from tests.fakes import FakeResponse, FakeSession, public_resolver

RES = public_resolver()


@pytest.mark.parametrize("url", [
    "http://localhost/admin", "http://127.0.0.1/", "http://10.0.0.5/x", "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data", "http://[::1]/", "ftp://example.com/f", "file:///etc/passwd",
    "javascript:alert(1)", "http://user:pass@example.com/", "https:///nohost", "http://svc.localhost/",
])
def test_unsafe_urls_rejected(url):
    with pytest.raises(UnsafeURLError):
        validate_public_http_url(url, RES)


def test_hostname_resolving_to_private_address_is_rejected():
    with pytest.raises(UnsafeURLError):
        validate_public_http_url("https://intranet.corp/x", public_resolver({"intranet.corp": "10.1.2.3"}))


def test_unresolvable_host_rejected():
    def boom(host, port, type=None):
        raise OSError("nxdomain")
    with pytest.raises(UnsafeURLError):
        validate_public_http_url("https://nope.invalid/", boom)


def test_public_url_accepted():
    assert validate_public_http_url("https://www.who.int/x", RES) == "www.who.int"


def test_successful_fetch():
    s = FakeSession({"https://a.com/x": FakeResponse(200, b"<html>hello</html>")})
    r = safe_get("https://a.com/x", session=s, resolver=RES)
    assert r.status_code == 200 and r.body == b"<html>hello</html>" and r.redirects == 0


def test_redirect_to_private_address_is_blocked_before_it_is_requested():
    s = FakeSession({"https://a.com/x": FakeResponse(302, headers={"Location": "http://169.254.169.254/secret"})})
    with pytest.raises(UnsafeURLError):
        safe_get("https://a.com/x", session=s, resolver=RES)
    assert [c[0] for c in s.calls] == ["https://a.com/x"]  # the internal URL was never requested


def test_relative_redirect_is_followed_and_hook_called():
    s = FakeSession({"https://a.com/x": FakeResponse(301, headers={"Location": "/y"}),
                     "https://a.com/y": FakeResponse(200, b"ok")})
    hops = []
    r = safe_get("https://a.com/x", session=s, resolver=RES, on_hop=hops.append)
    assert r.final_url == "https://a.com/y" and r.redirects == 1 and hops == ["https://a.com/y"]


def test_redirect_limit():
    s = FakeSession({"https://a.com/x": FakeResponse(302, headers={"Location": "https://a.com/x"})})
    with pytest.raises(RedirectLimit):
        safe_get("https://a.com/x", session=s, resolver=RES, max_redirects=2)


def test_on_hop_can_abort():
    s = FakeSession({"https://a.com/x": FakeResponse(302, headers={"Location": "https://b.com/y"})})

    def refuse(url):
        raise RuntimeError("blocked")
    with pytest.raises(RuntimeError):
        safe_get("https://a.com/x", session=s, resolver=RES, on_hop=refuse)


def test_http_errors_timeouts_size_and_content_type():
    with pytest.raises(FetchHTTPError) as e:
        safe_get("https://a.com/x", session=FakeSession({"https://a.com/x": FakeResponse(404)}), resolver=RES)
    assert e.value.status_code == 404
    with pytest.raises(FetchTimeout):
        safe_get("https://a.com/x", session=FakeSession({"https://a.com/x": requests.Timeout()}), resolver=RES)
    with pytest.raises(FetchTooLarge):
        safe_get("https://a.com/x", session=FakeSession({"https://a.com/x": FakeResponse(200, b"x" * 5000)}),
                 resolver=RES, max_bytes=1000)
    pdf = FakeResponse(200, b"%PDF", headers={"Content-Type": "application/pdf"})
    with pytest.raises(NonHtmlContent):
        safe_get("https://a.com/x", session=FakeSession({"https://a.com/x": pdf}), resolver=RES)


def test_responses_are_closed():
    resp = FakeResponse(200, b"ok")
    safe_get("https://a.com/x", session=FakeSession({"https://a.com/x": resp}), resolver=RES)
    assert resp.closed

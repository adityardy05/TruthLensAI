from unittest.mock import Mock, patch

from bot.backend_client import BackendClient


@patch("bot.backend_client.requests.post")
def test_bot_uses_backend_api(mock_post):
    response = Mock(ok=True)
    response.json.return_value = {"verdict": "TRUE"}
    mock_post.return_value = response
    assert BackendClient("http://backend").verify_text("claim")["verdict"] == "TRUE"
    assert mock_post.call_args.args[0].endswith("/api/v1/verify")

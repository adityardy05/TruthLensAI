import unittest
from unittest.mock import Mock, patch

from flask import Flask

from backend.api.verification import api
from backend.services.verification_service import VerificationService
from bot.backend_client import BackendClient


class FakeRetriever:
    def process_claim(self, claim, original_language, top_k=5):
        return {"sub_claims": [claim], "evidence": [{"source_domain": "example.org", "combined_reliability": 0.8}],
                "metadata": {"num_sources_kept": 1}}


class UnifiedArchitectureTests(unittest.TestCase):
    @patch("backend.services.verification_service.run_graph")
    def test_service_composes_retrieval_and_graph(self, graph):
        graph.return_value = {"verdict": "TRUE", "final_confidence": 85, "justification": "Supported.",
                              "recommendation": "Likely accurate", "persona_insights": {}, "rounds_executed": 1}
        result = VerificationService(FakeRetriever(), 1024).verify_text("A verifiable claim.")
        self.assertEqual(result["verdict"], "TRUE")
        graph.assert_called_once()

    def test_api_contract(self):
        app = Flask(__name__)
        app.extensions["verification_service"] = type("Service", (), {
            "verify_text": lambda _, claim, original_language=None, top_k=5: {"original_claim": claim, "verdict": "UNVERIFIABLE", "confidence": 0, "evidence": []}
        })()
        app.extensions["retriever"] = type("Retriever", (), {"faiss_index": None, "scraper": type("Scraper", (), {"api_key": None})()})()
        app.register_blueprint(api)
        self.assertEqual(app.test_client().post("/api/v1/verify", json={"claim": "test"}).get_json()["verdict"], "UNVERIFIABLE")
        self.assertEqual(app.test_client().get("/api/v1/health").status_code, 200)

    @patch("bot.backend_client.requests.post")
    def test_bot_calls_flask_api(self, post):
        response = Mock(ok=True)
        response.json.return_value = {"verdict": "TRUE"}
        post.return_value = response
        self.assertEqual(BackendClient("http://backend").verify_text("claim")["verdict"], "TRUE")
        self.assertTrue(post.call_args.args[0].endswith("/api/v1/verify"))


if __name__ == "__main__":
    unittest.main()

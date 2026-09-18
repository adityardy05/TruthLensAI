"""
================================================================
VERIFICATION SUITE: UNIFIED FLASK API ENDPOINTS
================================================================
"""

import json
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask

from backend.api.verification import api


class FakeVerificationService:
    def verify_text(self, claim, original_language="en", top_k=5):
        return {
            "verdict": "TRUE",
            "confidence": 0.91,
            "claim": claim,
            "justification": "Test verification result.",
            "evidence": [
                {
                    "source_domain": "example.org",
                    "combined_reliability": 0.8
                }
            ],
            "metadata": {
                "num_sources_kept": 1
            }
        }


class TestUnifiedApiEndpoints(unittest.TestCase):

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(api)

        fake_scraper = type(
            "FakeScraper",
            (),
            {"api_key": "test-key"}
        )()

        fake_retriever = type(
            "FakeRetriever",
            (),
            {
                "faiss_index": object(),
                "scraper": fake_scraper
            }
        )()

        self.app.extensions["retriever"] = fake_retriever
        self.app.extensions["verification_service"] = FakeVerificationService()

        self.client = self.app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        """Test GET /api/v1/health."""
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["faiss_loaded"])
        self.assertTrue(data["tavily_configured"])

    def test_verify_endpoint(self):
        """Test POST /api/v1/verify."""
        payload = {
            "claim": "Does drinking alcohol cure COVID-19?",
            "original_language": "en",
            "top_k": 5
        }

        response = self.client.post(
            "/api/v1/verify",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()

        self.assertIn("verdict", data)
        self.assertIn("confidence", data)
        self.assertIn("claim", data)
        self.assertIn("justification", data)
        self.assertIn("evidence", data)

        self.assertEqual(data["verdict"], "TRUE")


if __name__ == "__main__":
    unittest.main()

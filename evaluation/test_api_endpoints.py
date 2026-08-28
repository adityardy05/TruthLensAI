"""
================================================================
VERIFICATION SUITE: FLASK API ENDPOINTS (TruthLens v2.0)
================================================================
"""

import os
import sys
import json
import unittest
from unittest.mock import patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.app import app
import api.app as app_module

class TestTeamAApiEndpoints(unittest.TestCase):

    def setUp(self):
        if app is not None:
            self.app = app.test_client()
            self.app.testing = True
        else:
            self.app = None

    def test_health_endpoint(self):
        """Test GET /health returns 200 OK with expected fields."""
        if self.app is None:
            self.skipTest("Flask not installed")
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["architecture_spec"], "TruthLens v2.0")

    def test_retrieve_endpoint(self):
        """Test POST /retrieve returns 200 OK with Inter-Team Data Contract JSON."""
        if self.app is None:
            self.skipTest("Flask not installed")
        payload = {
            "claim": "Does drinking alcohol cure COVID-19?",
            "original_language": "en",
            "top_k": 5
        }
        response = self.app.post('/retrieve', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIn("claim", data)
        self.assertIn("normalized_claim", data)
        self.assertIn("sub_claims", data)
        self.assertIn("evidence", data)
        self.assertIn("metadata", data)

    @patch("api.app.requests.post")
    def test_forward_to_team_b_sends_contract(self, mock_post):
        """Test the Team B handoff sends the complete contract as JSON."""
        mock_post.return_value.raise_for_status.return_value = None
        contract = {"claim": "example", "evidence": []}
        original_url = app_module.TEAM_B_URL
        app_module.TEAM_B_URL = "http://127.0.0.1:8000/analyze"
        try:
            self.assertIsNone(app_module.forward_to_team_b(contract))
        finally:
            app_module.TEAM_B_URL = original_url

        mock_post.assert_called_once_with(
            "http://127.0.0.1:8000/analyze",
            json=contract,
            timeout=app_module.TEAM_B_TIMEOUT
        )

if __name__ == "__main__":
    unittest.main()

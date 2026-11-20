"""
================================================================
VERIFICATION SUITE: TEAM A PIPELINE (SearXNG architecture)
================================================================

Offline tests (no network, no model download) for:
    1. Hybrid Domain Credibility Evaluator (3 layers)
    2. Recency scoring
    3. R(d) = 0.40*Relevance + 0.20*Quality + 0.25*Credibility + 0.15*Recency
    4. Team A -> Team B data contract (process_claim output)
================================================================
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core.config import RetrievalSettings
from backend.rag.domain_credibility import HybridDomainCredibility
from backend.rag.retriever import TeamARetrievalPipeline
from backend.rag.scoring import compute_rd, compute_recency_score, score_chunks
from backend.rag.searxng_client import SearchResponse
from tests.fakes import FakeEmbedder


class _NoStore:
    available = False
    count = 0


class _NoSearch:
    def search(self, query, language=None):
        return SearchResponse()


class _NoFetch:
    def fetch_all(self, candidates):
        return []


class TestTeamAPipeline(unittest.TestCase):

    def setUp(self):
        self.domain_evaluator = HybridDomainCredibility()
        self.settings = RetrievalSettings()
        self.pipeline = TeamARetrievalPipeline(
            settings=self.settings, embedder=FakeEmbedder(), store=_NoStore(),
            searxng=_NoSearch(), fetcher=_NoFetch(),
        )

    def test_layer1_hardlist(self):
        """Test trusted and untrusted hardlist domains."""

        score_who, layer_who = self.domain_evaluator.evaluate(
            "https://www.who.int/news/item"
        )

        self.assertGreaterEqual(score_who, 0.90)
        self.assertEqual(layer_who, "hardlist_trusted")

        score_tw, layer_tw = self.domain_evaluator.evaluate(
            "https://twitter.com/someuser/status/12345"
        )

        self.assertLess(score_tw, 0.40)
        self.assertEqual(layer_tw, "hardlist_untrusted")

    def test_layer2_mbfc_cache(self):
        """Test MBFC cached domain rating."""

        score_wsj, layer_wsj = self.domain_evaluator.evaluate(
            "https://www.wsj.com/articles/news"
        )

        self.assertEqual(layer_wsj, "mbfc_cache")
        self.assertGreaterEqual(score_wsj, 0.80)

    def test_layer3_dynamic_fallback(self):
        """Test dynamic fallback domain heuristic."""

        score_gov, layer_gov = self.domain_evaluator.evaluate(
            "https://health.karnataka.gov.in/page"
        )

        self.assertEqual(layer_gov, "dynamic_fallback")
        self.assertGreaterEqual(score_gov, 0.90)

        score_spam, layer_spam = self.domain_evaluator.evaluate(
            "http://breaking-viral-daily24.xyz/fake"
        )

        self.assertEqual(layer_spam, "dynamic_fallback")
        self.assertLess(score_spam, 0.35)


    def test_recency_scoring(self):
        """Test recency scoring milestones."""

        from datetime import datetime, timedelta

        now = datetime.now()

        d_20d = (
            now - timedelta(days=20)
        ).strftime("%Y-%m-%d")

        score_20d = compute_recency_score(d_20d)

        self.assertEqual(score_20d, 1.00)

        d_1yr = (
            now - timedelta(days=365)
        ).strftime("%Y-%m-%d")

        score_1yr = compute_recency_score(d_1yr)

        self.assertAlmostEqual(
            score_1yr,
            0.60,
            delta=0.05
        )

        d_6yr = (
            now - timedelta(days=2000)
        ).strftime("%Y-%m-%d")

        score_6yr = compute_recency_score(d_6yr)

        self.assertEqual(score_6yr, 0.10)

    def test_rd_formula_uses_agreed_weights(self):
        """R(d) = 0.40*Relevance + 0.20*Quality + 0.25*Credibility + 0.15*Recency."""
        chunk = {
            "text": "Alcohol and whiskey do not cure COVID-19 according to medical experts. WHO said so.",
            "url": "https://www.who.int/statement", "domain": "who.int", "evidence_type": "extracted",
            "published_date": "2026-10-20", "cleanup_flags": [],
        }
        score_chunks("Does whiskey cure COVID-19?", [chunk], FakeEmbedder(), self.domain_evaluator, self.settings)
        expected = round(
            0.40 * chunk["relevance_score"] + 0.20 * chunk["quality_score"]
            + 0.25 * chunk["domain_credibility"] + 0.15 * chunk["recency_score"], 4)
        self.assertAlmostEqual(chunk["combined_reliability"], expected, delta=0.001)
        self.assertAlmostEqual(compute_rd(1, 1, 1, 1, self.settings), 1.0, places=4)
        self.assertAlmostEqual(compute_rd(0, 0, 0, 0, self.settings), 0.0, places=4)

    def test_inter_team_data_contract_schema(self):
        """Team A output contract, produced with all external services faked."""
        result = self.pipeline.process_claim("Vaccines cause autism in children.", original_language="en", top_k=3)
        for key in ["claim", "original_language", "normalized_claim", "sub_claims", "evidence", "metadata"]:
            self.assertIn(key, result, f"Missing required top-level key '{key}'")
        self.assertIsInstance(result["evidence"], list)
        self.assertIsInstance(result["metadata"], dict)
        self.assertEqual(result["evidence"], [])  # nothing retrievable -> honest empty result
        self.assertTrue(result["metadata"]["insufficient_evidence"])


if __name__ == "__main__":
    unittest.main()


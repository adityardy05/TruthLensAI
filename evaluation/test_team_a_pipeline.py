"""
================================================================
VERIFICATION SUITE: TEAM A PIPELINE (TruthLens v2.0)
================================================================

Tests Team A implementation against TruthLens v2.0 Architecture requirements:
    1. Hybrid Domain Credibility Evaluator (3 Layers: Hardlist, MBFC, Dynamic Fallback)
    2. Recency Scoring Function (s₄landmarks: <30d=1.0, <1yr=0.6, >5yr=0.1)
    3. 4-Factor R(d) Evidence Reliability Scoring Formula:
       R(d) = 0.25·s₁ + 0.25·s₂ + 0.25·s₃ + 0.25·s₄
    4. Inter-Team Data Contract Schema Compliance (Module A -> Module B)
================================================================
"""

"""
================================================================
VERIFICATION SUITE: TEAM A PIPELINE (Unified Architecture)
================================================================

Tests the unified Team A retrieval implementation:
    1. Hybrid Domain Credibility Evaluator
    2. Recency Scoring
    3. 4-Factor Evidence Reliability Scoring
    4. Inter-Team Data Contract Schema
================================================================
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.rag.domain_credibility import HybridDomainCredibility
from backend.rag.retriever import (
    compute_recency_score,
    TeamARetrievalPipeline,
)


class TestTeamAPipeline(unittest.TestCase):

    def setUp(self):
        self.domain_evaluator = HybridDomainCredibility()

        index_path = os.path.join(
            BASE_DIR,
            "indexes",
            "truthlens.faiss"
        )

        metadata_path = os.path.join(
            BASE_DIR,
            "indexes",
            "metadata.pkl"
        )

        self.pipeline = TeamARetrievalPipeline(
            index_path=index_path,
            metadata_path=metadata_path
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

    def test_4_factor_scoring_formula(self):
        """Verify the 4-factor reliability formula."""

        raw_evidence = [
            {
                "source_id": "test_1",
                "source_url": "https://www.who.int/statement",
                "title": "WHO statement on COVID remedies",
                "text_snippet": (
                    "Alcohol and whiskey do not cure COVID-19 "
                    "according to medical experts."
                ),
                "published_date": "2026-08-01",
                "author": "WHO Staff",
            }
        ]

        claim = "Does whiskey cure COVID-19?"

        scored = self.pipeline.score_and_rank_evidence(
            claim,
            raw_evidence,
            top_k=1
        )

        self.assertEqual(len(scored), 1)

        item = scored[0]

        s1 = item["bm25_score"]
        s2 = item["cosine_score"]
        s3 = item["domain_credibility"]
        s4 = item["recency_score"]

        expected_rd = round(
            0.25 * s1
            + 0.25 * s2
            + 0.25 * s3
            + 0.25 * s4,
            4
        )

        actual_rd = item["combined_reliability"]

        self.assertAlmostEqual(
            expected_rd,
            actual_rd,
            delta=0.001
        )

    def test_inter_team_data_contract_schema(self):
        """Verify the Team A output contract."""

        claim = "Vaccines cause autism in children."

        result = self.pipeline.process_claim(
            claim,
            original_language="en",
            top_k=3
        )

        required_top_keys = [
            "claim",
            "original_language",
            "normalized_claim",
            "sub_claims",
            "evidence",
            "metadata",
        ]

        for key in required_top_keys:
            self.assertIn(
                key,
                result,
                f"Missing required top-level key '{key}'"
            )

        self.assertIsInstance(
            result["evidence"],
            list
        )

        self.assertIsInstance(
            result["metadata"],
            dict
        )

        if result["evidence"]:

            evidence_item = result["evidence"][0]

            required_evidence_keys = [
                "source_domain",
                "author",
                "organization",
                "content",
                "url",
                "bm25_score",
                "cosine_score",
                "domain_credibility",
                "credibility_source",
                "recency_score",
                "combined_reliability",
                "retrieval_rank",
            ]

            for e_key in required_evidence_keys:
                self.assertIn(
                    e_key,
                    evidence_item,
                    f"Missing evidence key '{e_key}'"
                )


if __name__ == "__main__":
    unittest.main()

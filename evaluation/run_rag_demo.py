"""
================================================================
RAG PIPELINE VERIFICATION & DEMONSTRATION SCRIPT (TruthLens v2.0)
================================================================

Tests end-to-end RAG retrieval, sub-claim decomposition, 4-factor evidence
scoring, domain credibility, and JSON contract formatting for sample claims.
================================================================
"""

"""
================================================================
RAG PIPELINE VERIFICATION & DEMONSTRATION SCRIPT
================================================================

Tests the unified Team A RAG pipeline:
    1. RAG retrieval
    2. Sub-claim decomposition
    3. 4-factor evidence scoring
    4. Domain credibility
    5. Inter-Team Data Contract formatting
================================================================
"""

import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.rag.retriever import TeamARetrievalPipeline


INDEX_PATH = os.path.join(
    BASE_DIR,
    "indexes",
    "truthlens.faiss"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "indexes",
    "metadata.pkl"
)


def test_rag_claim(claim_text: str, original_lang: str = "en"):
    print("=" * 75)
    print(f"TESTING RAG CLAIM: '{claim_text}'")
    print("=" * 75)

    pipeline = TeamARetrievalPipeline(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    raw_demo_evidence = [
        {
            "source_id": "web_1",
            "source_url": (
                "https://www.who.int/emergencies/diseases/"
                "novel-coronavirus-2019/advice-for-public/myth-busters"
            ),
            "title": (
                "Coronavirus disease (COVID-19) advice for the public: "
                "Mythbusters"
            ),
            "source_type": "web",
            "text_snippet": (
                "Drinking alcohol does not protect against COVID-19 "
                "and can be dangerous. Consuming alcohol or whiskey "
                "does not cure virus infection."
            ),
            "published_date": "2026-08-01",
            "author": "World Health Organization",
            "organization": "WHO"
        },
        {
            "source_id": "web_2",
            "source_url": (
                "https://www.reuters.com/article/"
                "uk-factcheck-alcohol-covid19"
            ),
            "title": "Fact Check: Alcohol does not cure COVID-19",
            "source_type": "web",
            "text_snippet": (
                "False claim: Drinking high-proof alcohol or whiskey "
                "kills coronavirus in the throat or body. Health "
                "officials confirm alcohol consumption offers no protection."
            ),
            "published_date": "2026-07-15",
            "author": "Reuters Fact Check Team",
            "organization": "Reuters"
        },
        {
            "source_id": "web_3",
            "source_url": (
                "http://viral-news24-7breaking.xyz/whiskey-cure"
            ),
            "title": "Shocking secret remedy: Whiskey cures COVID",
            "source_type": "web",
            "text_snippet": (
                "Whiskey destroys all viruses instantly in your body "
                "when taken daily."
            ),
            "published_date": "2020-03-10",
            "author": "Anonymous",
            "organization": "Viral News 24"
        }
    ]

    t0 = time.time()

    scored_evidence = pipeline.score_and_rank_evidence(
        claim_text,
        raw_demo_evidence,
        top_k=5
    )

    t1 = time.time()

    sub_claims = pipeline.generate_sub_claims(claim_text)

    result = {
        "claim": claim_text,
        "original_language": original_lang,
        "normalized_claim": claim_text,
        "sub_claims": sub_claims,
        "evidence": scored_evidence,
        "metadata": {
            "retrieval_time_ms": int((t1 - t0) * 1000),
            "num_sources_total": len(raw_demo_evidence),
            "num_sources_kept": len(scored_evidence),
            "top_source_domain": (
                scored_evidence[0]["source_domain"]
                if scored_evidence
                else "N/A"
            )
        }
    }

    print(
        "\n[RAG Engine] Retrieval & 4-Factor R(d) "
        f"Scoring Completed in {(t1 - t0) * 1000:.2f} ms"
    )

    print(f"  Original Language   : {result['original_language']}")
    print(f"  Normalized Claim    : {result['normalized_claim']}")
    print(f"  Generated Sub-claims: {result['sub_claims']}")
    print(
        f"  Total Sources Found : "
        f"{result['metadata']['num_sources_total']}"
    )
    print(
        f"  Top Sources Kept    : "
        f"{result['metadata']['num_sources_kept']}"
    )
    print(
        f"  Top Source Domain   : "
        f"{result['metadata']['top_source_domain']}"
    )

    print("\n--- RETRIEVED EVIDENCE & 4-FACTOR R(d) SCORES ---")

    for item in result["evidence"]:
        print(
            f"\n[Rank #{item['retrieval_rank']}] "
            f"Domain: {item['source_domain']} | "
            f"Source Layer: {item['credibility_source']}"
        )

        print(
            f"  - Combined Reliability R(d) : "
            f"{item['combined_reliability']}"
        )

        print(
            f"  - BM25 Score (s1)           : "
            f"{item['bm25_score']}"
        )

        print(
            f"  - Cosine Similarity (s2)    : "
            f"{item['cosine_score']}"
        )

        print(
            f"  - Domain Credibility (s3)  : "
            f"{item['domain_credibility']}"
        )

        print(
            f"  - Recency Score (s4)       : "
            f"{item['recency_score']}"
        )

        print(f"  - URL                       : {item['url']}")

        print(
            f"  - Content Snippet           : "
            f"{item['content'][:140]}..."
        )

    print("\n--- FULL INTER-TEAM DATA CONTRACT JSON ---")

    print(
        json.dumps(
            result,
            indent=2,
            default=str
        )
    )

    print("=" * 75)


if __name__ == "__main__":
    sample_claim = (
        "Does drinking alcohol or whiskey cure COVID-19?"
    )

    test_rag_claim(sample_claim)

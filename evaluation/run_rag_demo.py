"""
Live demonstration of the Team A retrieval pipeline (SearXNG architecture).

Requires a running SearXNG instance (SEARXNG_BASE_URL) and the all-MiniLM-L6-v2 model.
Run from the repository root:

    python -m evaluation.run_rag_demo "Does whiskey cure COVID-19?"
"""
from __future__ import annotations

import json
import sys

from backend.core.config import load_settings
from backend.rag.retriever import TeamARetrievalPipeline


def main(argv: list[str]) -> int:
    claim = " ".join(argv[1:]).strip() or "Does whiskey cure COVID-19?"
    settings = load_settings()
    pipeline = TeamARetrievalPipeline(
        index_path=str(settings.faiss_index_path),
        metadata_path=str(settings.faiss_metadata_path),
        settings=settings.retrieval,
    )
    print("Health:", json.dumps(pipeline.health()))
    result = pipeline.process_claim(claim, original_language="en")
    meta = result["metadata"]
    print(f"\nClaim: {claim}")
    print(f"Route: {meta['route']} (best FAISS similarity {meta['best_faiss_similarity']}, "
          f"threshold {meta['faiss_threshold']})")
    print(f"Queries: {meta['queries']}  | raw results: {meta['raw_results']}  | "
          f"extracted: {meta['fetched_extracted']}  | snippet fallbacks: {meta['snippet_fallbacks']}")
    if meta["unresponsive_engines"]:
        print("Unresponsive engines:", meta["unresponsive_engines"])
    if meta["insufficient_evidence"]:
        print("INSUFFICIENT EVIDENCE:", meta["insufficient_reason"])
    for e in result["evidence"]:
        print(f"\n#{e['retrieval_rank']} {e['source_domain']}  [{e['evidence_type']}]  R(d)={e['combined_reliability']}")
        print(f"   relevance={e['relevance_score']} quality={e['quality_score']} "
              f"credibility={e['domain_credibility']} ({e['credibility_source']}) recency={e['recency_score']}")
        print(f"   {e['url']}")
        print(f"   {e['content'][:200]}")
    print(f"\nRetrieval time: {meta['retrieval_time_ms']} ms  {meta['stage_timings_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


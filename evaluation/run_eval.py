"""
================================================================
RAG RETRIEVAL EVALUATION — UNIFIED ARCHITECTURE
================================================================
Runs TeamARetrievalPipeline over the canonical FAISS index and
reports standard IR metrics using two ground-truth definitions:

  1. Known-item:
     Relevant = the exact source statement itself.
     Tests embedding/index fidelity.

  2. Topical:
     Relevant = other documents sharing the same `subject` tag.
     Tests semantic/topical relevance quality.

Metric math is reused from evaluation/evaluator.py.
================================================================
"""

import os
import random
import sys
import time

# Allow imports from project root when this script is run directly.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.evaluator import RetrievalEvaluator as RE
from backend.rag.retriever import TeamARetrievalPipeline


# ----------------------------------------------------------------
# Canonical unified-architecture paths
# ----------------------------------------------------------------

INDEX_PATH = os.path.join(
    BASE_DIR,
    "indexes",
    "truthlens.faiss",
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "indexes",
    "metadata.pkl",
)


# ----------------------------------------------------------------
# Evaluation configuration
# ----------------------------------------------------------------

SAMPLE_SIZE = 200
TOP_K = 5
SEED = 42


def parse_faiss_source_id(source_id):
    """
    Convert a source ID such as 'faiss_2452' into corpus index 2452.

    Returns -1 when the source ID cannot be parsed.
    """
    if not isinstance(source_id, str):
        return -1

    if not source_id.startswith("faiss_"):
        return -1

    try:
        return int(source_id.split("_", 1)[1])
    except (ValueError, IndexError):
        return -1


def average_metric(metric_fn, retrieved_lists, relevant_lists):
    """
    Safely calculate the average of a per-query metric.
    """
    if not retrieved_lists:
        return 0.0

    scores = [
        metric_fn(retrieved, relevant, k=TOP_K)
        for retrieved, relevant in zip(
            retrieved_lists,
            relevant_lists,
        )
    ]

    return sum(scores) / len(scores)


def main():
    random.seed(SEED)

    # ------------------------------------------------------------
    # Validate required index files
    # ------------------------------------------------------------

    if not os.path.isfile(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}\n"
            "Run: python scripts/build_index.py"
        )

    if not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError(
            f"FAISS metadata not found: {METADATA_PATH}\n"
            "Run: python scripts/build_index.py"
        )

    # ------------------------------------------------------------
    # Initialize unified Team A retrieval pipeline
    # ------------------------------------------------------------

    print("\nInitializing unified Team A retrieval pipeline...")

    pipeline = TeamARetrievalPipeline(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
    )

    meta = pipeline.metadata
    corpus_size = len(meta)

    if corpus_size == 0:
        raise RuntimeError("FAISS metadata contains zero documents.")

    # ------------------------------------------------------------
    # Build subject -> corpus indices lookup
    # ------------------------------------------------------------

    subject_to_idxs = {}

    for i, document in enumerate(meta):
        subject_field = str(
            document.get("subject", "") or ""
        )

        subjects = [
            subject.strip().lower()
            for subject in subject_field.split(",")
            if subject.strip()
        ]

        for subject in subjects:
            subject_to_idxs.setdefault(
                subject,
                set(),
            ).add(i)

    # ------------------------------------------------------------
    # Select evaluation queries
    # ------------------------------------------------------------

    sample_idxs = random.sample(
        range(corpus_size),
        min(SAMPLE_SIZE, corpus_size),
    )

    # Known-item results
    known_item_retrieved = []
    known_item_relevant = []

    # Topical results
    topical_retrieved = []
    topical_relevant = []

    # Operational metrics
    returned_counts = []
    latencies = []

    # ------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------

    for query_index in sample_idxs:

        query_document = meta[query_index]

        query = (
            query_document.get("statement", "")
            or query_document.get("cleaned_text", "")
        )

        if not isinstance(query, str) or not query.strip():
            continue

        # Measure retrieval latency only.
        start_time = time.perf_counter()

        results = pipeline.retrieve_faiss(
            query,
            top_n=TOP_K,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        latencies.append(latency_ms)
        returned_counts.append(len(results))

        # --------------------------------------------------------
        # Map retrieval results back to corpus indices
        # --------------------------------------------------------

        retrieved_indices = []

        for result in results:
            corpus_index = parse_faiss_source_id(
                result.get("source_id")
            )

            if 0 <= corpus_index < corpus_size:
                retrieved_indices.append(corpus_index)

        # --------------------------------------------------------
        # 1. Known-item ground truth
        #
        # The source document used to create the query is relevant.
        # --------------------------------------------------------

        known_item_retrieved.append(retrieved_indices)
        known_item_relevant.append([query_index])

        # --------------------------------------------------------
        # 2. Topical ground truth
        #
        # Other documents sharing any subject tag are relevant.
        # The query document itself is excluded.
        # --------------------------------------------------------

        relevant_indices = set()

        subject_field = str(
            query_document.get("subject", "") or ""
        )

        subjects = [
            subject.strip().lower()
            for subject in subject_field.split(",")
            if subject.strip()
        ]

        for subject in subjects:
            relevant_indices |= subject_to_idxs.get(
                subject,
                set(),
            )

        relevant_indices.discard(query_index)

        topical_retrieved.append(retrieved_indices)
        topical_relevant.append(list(relevant_indices))

    # ------------------------------------------------------------
    # Validate evaluation results
    # ------------------------------------------------------------

    query_count = len(known_item_retrieved)

    if query_count == 0:
        raise RuntimeError(
            "No valid evaluation queries were produced."
        )

    # ------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------

    known_precision = average_metric(
        RE.precision_at_k,
        known_item_retrieved,
        known_item_relevant,
    )

    known_recall = average_metric(
        RE.recall_at_k,
        known_item_retrieved,
        known_item_relevant,
    )

    known_mrr = RE.mrr(
        known_item_retrieved,
        known_item_relevant,
    )

    known_map = RE.map_score(
        known_item_retrieved,
        known_item_relevant,
    )

    topical_precision = average_metric(
        RE.precision_at_k,
        topical_retrieved,
        topical_relevant,
    )

    topical_recall = average_metric(
        RE.recall_at_k,
        topical_retrieved,
        topical_relevant,
    )

    topical_mrr = RE.mrr(
        topical_retrieved,
        topical_relevant,
    )

    topical_map = RE.map_score(
        topical_retrieved,
        topical_relevant,
    )

    # ------------------------------------------------------------
    # Operational metrics
    # ------------------------------------------------------------

    average_results = (
        sum(returned_counts) / len(returned_counts)
        if returned_counts
        else 0.0
    )

    empty_queries = sum(
        1
        for count in returned_counts
        if count == 0
    )

    empty_percentage = (
        100.0 * empty_queries / len(returned_counts)
        if returned_counts
        else 0.0
    )

    sorted_latencies = sorted(latencies)

    latency_mean = (
        sum(sorted_latencies) / len(sorted_latencies)
        if sorted_latencies
        else 0.0
    )

    if sorted_latencies:
        p50_index = len(sorted_latencies) // 2
        p95_index = min(
            int(len(sorted_latencies) * 0.95),
            len(sorted_latencies) - 1,
        )

        latency_p50 = sorted_latencies[p50_index]
        latency_p95 = sorted_latencies[p95_index]
    else:
        latency_p50 = 0.0
        latency_p95 = 0.0

    # ------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print(
        f"RAG RETRIEVAL EVALUATION "
        f"(n={query_count} queries, K={TOP_K})"
    )
    print("=" * 65)

    print(
        "\n--- 1. KNOWN-ITEM RETRIEVAL "
        "(relevant = source statement) ---"
    )

    print(
        f"  Precision@{TOP_K} : {known_precision:.4f}"
    )

    print(
        f"  Recall@{TOP_K}    : {known_recall:.4f}"
    )

    print(
        f"  MRR              : {known_mrr:.4f}"
    )

    print(
        f"  MAP              : {known_map:.4f}"
    )

    print(
        "\n--- 2. TOPICAL RETRIEVAL "
        "(relevant = same subject tag) ---"
    )

    print(
        f"  Precision@{TOP_K} : {topical_precision:.4f}"
    )

    print(
        f"  Recall@{TOP_K}    : {topical_recall:.4f}"
    )

    print(
        f"  MRR              : {topical_mrr:.4f}"
    )

    print(
        f"  MAP              : {topical_map:.4f}"
    )

    print("\n--- SYSTEM / OPERATIONAL METRICS ---")

    print(
        f"  Corpus size            : "
        f"{corpus_size} documents"
    )

    print(
        f"  Avg results returned   : "
        f"{average_results:.2f} / {TOP_K}"
    )

    print(
        f"  Empty (gated) queries  : "
        f"{empty_queries} "
        f"({empty_percentage:.1f}%)"
    )

    print(
        f"  Latency mean           : "
        f"{latency_mean:.1f} ms"
    )

    print(
        f"  Latency p50 / p95      : "
        f"{latency_p50:.1f} ms / "
        f"{latency_p95:.1f} ms"
    )

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

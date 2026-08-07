"""
================================================================
RAG RETRIEVAL EVALUATION (real index)
================================================================
Runs the EnhancedRetrievalAgent over the compiled FAISS index and
reports standard IR metrics using two ground-truth definitions:

  1. Known-item  : relevant = the exact source statement itself.
                   Tests embedding/index fidelity (can it find the doc?).
  2. Topical     : relevant = other docs sharing the same `subject` tag.
                   Tests semantic/topical relevance quality.

Metric math is reused from evaluation/evaluator.py (RetrievalEvaluator).
================================================================
"""

import os
import sys
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluator import RetrievalEvaluator as RE
from src.embeddings.retriever import EnhancedRetrievalAgent

INDEX_PATH    = os.path.join(os.path.dirname(__file__), "..", "index", "fake_news.faiss")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "index", "metadata.pkl")

SAMPLE_SIZE = 200
TOP_K       = 5
SEED        = 42


def main():
    random.seed(SEED)

    agent = EnhancedRetrievalAgent(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        bm25_weight=0.3,
        semantic_weight=0.7,
        temporal_weight=0.1,
        credibility_weight=0.15,
        confidence_threshold=0.4,
        temporal_decay_days=365,
    )

    meta = agent.metadata
    N = len(meta)

    # ── Build lookup maps ────────────────────────────────────────
    # Composite key -> corpus index (to map a search result back to a doc id)
    key_to_idx = {}
    for i, d in enumerate(meta):
        key = (d.get("statement", ""), d.get("speaker", ""), str(d.get("context", "")))
        key_to_idx.setdefault(key, i)

    # subject tag -> set of corpus indices sharing that subject
    subject_to_idxs = {}
    for i, d in enumerate(meta):
        subj_field = str(d.get("subject", "") or "")
        for subj in [s.strip().lower() for s in subj_field.split(",") if s.strip()]:
            subject_to_idxs.setdefault(subj, set()).add(i)

    # ── Sample queries ───────────────────────────────────────────
    sample_idxs = random.sample(range(N), min(SAMPLE_SIZE, N))

    ki_retrieved, ki_relevant = [], []   # known-item
    tp_retrieved, tp_relevant = [], []   # topical
    returned_counts = []
    latencies = []

    for qi in sample_idxs:
        qdoc = meta[qi]
        query = qdoc.get("statement", "") or qdoc.get("cleaned_text", "")
        if not query.strip():
            continue

        t0 = time.perf_counter()
        results = agent.search(query, top_k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        returned_counts.append(len(results))

        # Map each returned result back to a corpus index
        retrieved_idxs = []
        for r in results:
            key = (r.get("statement", ""), r.get("speaker", ""), str(r.get("context", "")))
            retrieved_idxs.append(key_to_idx.get(key, -1))

        # --- Known-item ground truth: the query doc itself ---
        ki_retrieved.append(retrieved_idxs)
        ki_relevant.append([qi])

        # --- Topical ground truth: docs sharing any subject (excl. self) ---
        rel = set()
        subj_field = str(qdoc.get("subject", "") or "")
        for subj in [s.strip().lower() for s in subj_field.split(",") if s.strip()]:
            rel |= subject_to_idxs.get(subj, set())
        rel.discard(qi)
        tp_retrieved.append(retrieved_idxs)
        tp_relevant.append(list(rel))

    # ── Aggregate metrics ────────────────────────────────────────
    def avg(fn, R, G):
        return sum(fn(r, g, k=TOP_K) for r, g in zip(R, G)) / len(R)

    print("\n" + "=" * 65)
    print(f"RAG RETRIEVAL EVALUATION  (n={len(ki_retrieved)} queries, K={TOP_K})")
    print("=" * 65)

    print("\n--- 1. KNOWN-ITEM RETRIEVAL (relevant = source statement) ---")
    print(f"  Precision@{TOP_K} : {avg(RE.precision_at_k, ki_retrieved, ki_relevant):.4f}")
    print(f"  Recall@{TOP_K}    : {avg(RE.recall_at_k,    ki_retrieved, ki_relevant):.4f}")
    print(f"  MRR          : {RE.mrr(ki_retrieved, ki_relevant):.4f}")
    print(f"  MAP          : {RE.map_score(ki_retrieved, ki_relevant):.4f}")

    print("\n--- 2. TOPICAL RETRIEVAL (relevant = same subject tag) ---")
    print(f"  Precision@{TOP_K} : {avg(RE.precision_at_k, tp_retrieved, tp_relevant):.4f}")
    print(f"  Recall@{TOP_K}    : {avg(RE.recall_at_k,    tp_retrieved, tp_relevant):.4f}")
    print(f"  MRR          : {RE.mrr(tp_retrieved, tp_relevant):.4f}")
    print(f"  MAP          : {RE.map_score(tp_retrieved, tp_relevant):.4f}")

    print("\n--- SYSTEM / OPERATIONAL METRICS ---")
    print(f"  Corpus size            : {N} documents")
    print(f"  Avg results returned   : {sum(returned_counts)/len(returned_counts):.2f} / {TOP_K}")
    empty = sum(1 for c in returned_counts if c == 0)
    print(f"  Empty (gated) queries  : {empty} ({100*empty/len(returned_counts):.1f}%)")
    lat = sorted(latencies)
    print(f"  Latency mean           : {sum(lat)/len(lat):.1f} ms")
    print(f"  Latency p50 / p95      : {lat[len(lat)//2]:.1f} ms / {lat[int(len(lat)*0.95)]:.1f} ms")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

"""
================================================================
ENHANCED RETRIEVAL AGENT
================================================================

Implements a hybrid RAG retrieval pipeline with four layers:

    1. BM25 Lexical Search       — Keyword/exact-match scoring
    2. Semantic FAISS Search     — Dense vector similarity scoring
    3. Hybrid Score Fusion       — Combines BM25 + Semantic (weighted)
    4. Temporal Filtering        — Penalises older / undated evidence
    5. Credibility Scoring       — Boosts evidence from trusted sources
    6. Confidence-Gated Return   — Only returns results above a threshold

Flow:
    Query
      |
      v
    [BM25 Lexical Score]  +  [FAISS Semantic Score]
                               |
                               v
                    [Hybrid Fusion Score]
                               |
                               v
                    [Temporal Decay Penalty]
                               |
                               v
                    [Credibility Source Boost]
                               |
                               v
                    [Confidence Gate Filter]
                               |
                               v
                    Top-K Final Results

================================================================
"""

import faiss
import math
import pickle
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# BM25 — pure-Python, no C extensions required
from rank_bm25 import BM25Okapi


# ================================================================
# CREDIBILITY DICTIONARY
# Maps source domains to a credibility score between 0.0 and 1.0.
# Scores represent editorial standards and fact-checking reputation.
# URL credibility verification is handled upstream by the Credibility Agent;
# this dictionary provides a passive boost during retrieval scoring.
# ================================================================

CREDIBILITY_SCORES: Dict[str, float] = {
    # Tier 1 — Highly trusted, non-partisan fact-checkers and newswires
    "reuters.com":      1.00,
    "apnews.com":       1.00,
    "factcheck.org":    1.00,
    "snopes.com":       0.95,
    "bbc.com":          0.95,
    "npr.org":          0.90,

    # Tier 2 — Reputable mainstream outlets
    "theguardian.com":  0.85,
    "nytimes.com":      0.85,
    "washingtonpost.com": 0.85,
    "bbc.co.uk":        0.85,
    "aljazeera.com":    0.80,
    "thehindu.com":     0.80,
    "ndtv.com":         0.75,
    "hindustantimes.com": 0.70,

    # Tier 3 — General / neutral (default for unknown sources)
    "__default__":      0.50,

    # Tier 4 — Low-credibility or known misinformation sources
    # Add known bad-actor domains here with scores < 0.3 if needed
}


def get_credibility_score(url: str) -> float:
    """
    Looks up the credibility score for a source URL.
    Iterates over known domains and matches by substring.
    Falls back to the default score if no domain is recognized.

    :param url: The source URL string.
    :return: Credibility float in [0.0, 1.0].
    """
    if not isinstance(url, str) or not url.strip():
        return CREDIBILITY_SCORES["__default__"]

    url_lower = url.lower()
    for domain, score in CREDIBILITY_SCORES.items():
        if domain == "__default__":
            continue
        if domain in url_lower:
            return score

    return CREDIBILITY_SCORES["__default__"]


# ================================================================
# TEMPORAL SCORING
# ================================================================

def compute_temporal_score(publish_date_str: Optional[str], decay_days: int = 365) -> float:
    """
    Assigns a temporal relevance score between 0.0 and 1.0.

    - Recent articles score closer to 1.0.
    - Articles older than `decay_days` decay exponentially toward 0.0.
    - Missing or unparseable dates receive a neutral score of 0.5.

    Uses exponential decay:  score = e^(-age_in_days / decay_days)

    :param publish_date_str: Publication date string (ISO format preferred).
    :param decay_days:        Half-life denominator (default 365 days).
    :return: Temporal score float in [0.0, 1.0].
    """
    if not publish_date_str or publish_date_str in ("No Date", "", None):
        return 0.5  # Neutral score for undated evidence

    # Try common date formats
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]

    pub_date = None
    for fmt in formats:
        try:
            pub_date = datetime.strptime(str(publish_date_str).strip(), fmt)
            break
        except ValueError:
            continue

    if pub_date is None:
        return 0.5  # Could not parse date — use neutral score

    now = datetime.now()
    age_in_days = max((now - pub_date).days, 0)
    return math.exp(-age_in_days / decay_days)


# ================================================================
# ENHANCED RETRIEVAL AGENT
# ================================================================

class EnhancedRetrievalAgent:
    """
    Hybrid RAG Retrieval Agent combining:
        - BM25 (TF-IDF keyword matching)
        - Semantic FAISS vector search
        - Temporal decay scoring
        - Source credibility boosting
        - Confidence-gated result filtering

    Parameters
    ----------
    index_path        : Path to the compiled FAISS index (.faiss file).
    metadata_path     : Path to the metadata pickle file (.pkl file).
    model_name        : SentenceTransformer model (must match the one used to build the index).
    bm25_weight       : Weight for BM25 score in hybrid fusion (0.0–1.0).
    semantic_weight   : Weight for semantic FAISS score in hybrid fusion (0.0–1.0).
    temporal_weight   : Weight of temporal score in final scoring.
    credibility_weight: Weight of source credibility in final scoring.
    confidence_threshold: Minimum final score to include a result (confidence gate).
    temporal_decay_days : Age in days at which temporal score = e^-1 ≈ 0.37.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        temporal_weight: float = 0.1,
        credibility_weight: float = 0.15,
        confidence_threshold: float = 0.4,
        temporal_decay_days: int = 365,
    ):
        # ── Weights validation ──────────────────────────────────
        assert abs(bm25_weight + semantic_weight - 1.0) < 1e-6, (
            "bm25_weight + semantic_weight must equal 1.0"
        )

        self.bm25_weight         = bm25_weight
        self.semantic_weight     = semantic_weight
        self.temporal_weight     = temporal_weight
        self.credibility_weight  = credibility_weight
        self.confidence_threshold = confidence_threshold
        self.temporal_decay_days = temporal_decay_days

        # ── Load Sentence Transformer ───────────────────────────
        print(f"[Retriever] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        # ── Load FAISS Index ────────────────────────────────────
        print(f"[Retriever] Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)

        # ── Load Metadata ───────────────────────────────────────
        print(f"[Retriever] Loading metadata from {metadata_path}...")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # ── Build BM25 Index ────────────────────────────────────
        print("[Retriever] Building BM25 index from corpus...")
        self._build_bm25_index()

        print(f"[Retriever] Ready. Corpus size: {len(self.metadata)} documents.\n")

    # ──────────────────────────────────────────────────────────────
    # BM25 Index Construction
    # ──────────────────────────────────────────────────────────────

    def _build_bm25_index(self):
        """
        Tokenizes each document in the metadata corpus and builds
        a BM25Okapi index for lexical keyword retrieval.

        Uses the 'cleaned_text' field. Falls back to 'statement' if
        'cleaned_text' is absent (for raw LIAR dataset entries).
        """
        tokenized_corpus = []
        for doc in self.metadata:
            text = doc.get("cleaned_text") or doc.get("statement", "")
            tokens = str(text).lower().split()
            tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(tokenized_corpus)

    # ──────────────────────────────────────────────────────────────
    # BM25 Lexical Search
    # ──────────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_n: int) -> Dict[int, float]:
        """
        Runs BM25 on the query and returns a dict of {doc_index: normalised_score}.
        Scores are normalised to [0.0, 1.0] by dividing by the max BM25 score.

        :param query:  Raw query string.
        :param top_n:  Number of top results to consider.
        :return: Dict mapping corpus index → normalised BM25 score.
        """
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Normalise scores to [0, 1]
        max_score = np.max(scores) if np.max(scores) > 0 else 1.0
        normalised = scores / max_score

        # Grab the top_n indices sorted by descending score
        top_indices = np.argsort(normalised)[::-1][:top_n]
        return {int(idx): float(normalised[idx]) for idx in top_indices}

    # ──────────────────────────────────────────────────────────────
    # FAISS Semantic Search
    # ──────────────────────────────────────────────────────────────

    def _semantic_search(self, query: str, top_n: int) -> Dict[int, float]:
        """
        Embeds the query and runs FAISS similarity search.
        Returns a dict of {doc_index: normalised_semantic_score}.

        FAISS returns L2 distances (lower = more similar).
        We convert to similarity scores using: score = 1 / (1 + distance).

        :param query:  Raw query string.
        :param top_n:  Number of top FAISS results.
        :return: Dict mapping corpus index → semantic similarity score.
        """
        query_embedding = np.array(
            self.model.encode([query])
        ).astype("float32")

        distances, indices = self.index.search(query_embedding, top_n)

        results = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            # Convert L2 distance to similarity (0-1 range)
            similarity = 1.0 / (1.0 + float(dist))
            results[int(idx)] = similarity

        return results

    # ──────────────────────────────────────────────────────────────
    # Hybrid Fusion
    # ──────────────────────────────────────────────────────────────

    def _fuse_scores(
        self,
        bm25_scores: Dict[int, float],
        semantic_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Fuses BM25 and semantic scores using a weighted linear combination:

            hybrid_score = (bm25_weight × bm25_score) + (semantic_weight × semantic_score)

        Considers the union of both result sets so that results appearing
        in only one retriever still have a chance to surface.

        :param bm25_scores:     Dict {doc_idx: normalised_bm25_score}
        :param semantic_scores: Dict {doc_idx: semantic_similarity_score}
        :return: Dict {doc_idx: hybrid_score}
        """
        all_indices = set(bm25_scores.keys()) | set(semantic_scores.keys())
        fused = {}
        for idx in all_indices:
            b_score = bm25_scores.get(idx, 0.0)
            s_score = semantic_scores.get(idx, 0.0)
            fused[idx] = (self.bm25_weight * b_score) + (self.semantic_weight * s_score)
        return fused

    # ──────────────────────────────────────────────────────────────
    # Final Scoring (Temporal + Credibility Boost)
    # ──────────────────────────────────────────────────────────────

    def _compute_final_score(
        self,
        hybrid_score: float,
        publish_date: str,
        source_url: str,
    ) -> float:
        """
        Applies temporal decay and credibility boost on top of the hybrid score.

        Final Formula:
            final = hybrid_score
                  + (temporal_weight  × temporal_score)
                  + (credibility_weight × credibility_score)

        The result is clipped to [0.0, 1.0].

        :param hybrid_score:  Fused BM25 + semantic score.
        :param publish_date:  Article publish date string.
        :param source_url:    Article source URL.
        :return: Final composite score in [0.0, 1.0].
        """
        temporal_score    = compute_temporal_score(publish_date, self.temporal_decay_days)
        credibility_score = get_credibility_score(source_url)

        final = (
            hybrid_score
            + (self.temporal_weight   * temporal_score)
            + (self.credibility_weight * credibility_score)
        )
        return min(max(final, 0.0), 1.0)  # Clip to [0, 1]

    # ──────────────────────────────────────────────────────────────
    # Public Search API
    # ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with temporal filtering, credibility scoring,
        and confidence gating.

        Steps:
            1. BM25 lexical search over `candidate_pool` documents.
            2. FAISS semantic search over `candidate_pool` documents.
            3. Fuse scores with weighted combination.
            4. Apply temporal decay and credibility boost.
            5. Filter out results below `confidence_threshold`.
            6. Return top `top_k` results sorted by final score.

        :param query:          The claim or question text.
        :param top_k:          Maximum number of results to return.
        :param candidate_pool: How many candidates each retriever considers
                               before fusion (larger = more thorough, slower).
        :return: List of result dicts, sorted by descending final_score.
        """
        if not query or not query.strip():
            return []

        # Step 1 & 2: BM25 + Semantic search
        bm25_scores     = self._bm25_search(query, top_n=candidate_pool)
        semantic_scores = self._semantic_search(query, top_n=candidate_pool)

        # Step 3: Hybrid fusion
        fused_scores = self._fuse_scores(bm25_scores, semantic_scores)

        # Step 4 & 5: Apply temporal + credibility boost, then confidence gate
        results = []
        for idx, hybrid_score in fused_scores.items():
            doc = self.metadata[idx]

            publish_date = doc.get("publish_date") or doc.get("context", "")
            source_url   = doc.get("url", "")

            final_score = self._compute_final_score(hybrid_score, publish_date, source_url)

            # Confidence gate — skip weak evidence
            if final_score < self.confidence_threshold:
                continue

            results.append({
                "final_score":        round(final_score, 4),
                "hybrid_score":       round(hybrid_score, 4),
                "bm25_score":         round(bm25_scores.get(idx, 0.0), 4),
                "semantic_score":     round(semantic_scores.get(idx, 0.0), 4),
                "temporal_score":     round(
                    compute_temporal_score(publish_date, self.temporal_decay_days), 4
                ),
                "credibility_score":  round(get_credibility_score(source_url), 4),
                "label":              doc.get("label", "unknown"),
                "statement":          doc.get("statement", doc.get("cleaned_text", "")),
                "speaker":            doc.get("speaker", ""),
                "context":            doc.get("context", ""),
                "source_url":         source_url,
                "publish_date":       publish_date,
            })

        # Step 6: Sort by final score descending, return top_k
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]


# ================================================================
# Quick test — run: python src/embeddings/retriever.py
# ================================================================

if __name__ == "__main__":
    INDEX_PATH    = "index/fake_news.faiss"
    METADATA_PATH = "index/metadata.pkl"

    try:
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

        test_queries = [
            "The economy bled $24 billion due to the government shutdown.",
            "Vaccines cause autism in children.",
            "Hillary Clinton email scandal corruption.",
        ]

        for query in test_queries:
            print("=" * 65)
            print(f"QUERY: {query}")
            print("=" * 65)
            results = agent.search(query, top_k=3)

            if not results:
                print("  [Confidence Gate] No results met the confidence threshold.\n")
                continue

            for i, r in enumerate(results, 1):
                print(f"  Match #{i}")
                print(f"    Final Score  : {r['final_score']}  "
                      f"(BM25={r['bm25_score']} | Sem={r['semantic_score']} | "
                      f"Temp={r['temporal_score']} | Cred={r['credibility_score']})")
                print(f"    Label        : {r['label']}")
                print(f"    Statement    : {str(r['statement'])[:120]}...")
                print(f"    Speaker      : {r['speaker']}")
                print(f"    Context      : {r['context']}")
                print("-" * 65)
            print()

    except FileNotFoundError:
        print(
            "Error: FAISS index or metadata not found.\n"
            "Run `python src/embeddings/generate_index.py` first."
        )

"""
================================================================
TEAM A: DATA INGESTION & RETRIEVAL ENGINE (TruthLens v2.0)
================================================================

Implements Team A's Module A RAG Retrieval and 4-Factor Evidence Scoring:

    Formula: R(d) = 0.25·s₁ + 0.25·s₂ + 0.25·s₃ + 0.25·s₄

    Where:
        s₁ = BM25 Lexical Score (normalized [0, 1])
        s₂ = Cosine Similarity (sentence-transformers all-MiniLM-L6-v2)
        s₃ = Hybrid Domain Credibility (Hardlist → MBFC cache → Dynamic fallback)
        s₄ = Recency Score (<30 days = 1.0, <1 yr = 0.6, >5 yrs = 0.1)

Combines:
    - Live Web Evidence via Tavily Search API + Trafilatura fallback
    - FAISS Vector DB search over historical claims database
================================================================
"""

import os
import sys
import math
import time
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# Optional dependency fallbacks
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    BM25Okapi = None
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_ST = True
except ImportError:
    SentenceTransformer = None
    util = None
    HAS_ST = False

# TF-IDF+SVD fallback (sklearn — always available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize as sk_normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Add project root to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.rag.domain_credibility import HybridDomainCredibility
from backend.rag.web_retriever import HybridScraper

# ================================================================
# RECENCY SCORING ENGINE
# s₄ = recency_score(pub_date)
# Spec: <30 days = 1.0, <1 yr (365 days) = 0.6, >5 yrs (1825 days) = 0.1
# ================================================================

def compute_recency_score(publish_date_str: Optional[str]) -> float:
    """
    Computes recency score s₄ according to TruthLens v2.0 specification:
      - < 30 days old     : 1.00
      - < 365 days (1 yr) : Smooth decay 1.00 -> 0.60
      - < 1825 days (5 yr): Smooth decay 0.60 -> 0.10
      - > 1825 days (>5 yr): 0.10
      - Undated / Unknown  : 0.50 (Neutral)
    """
    if not publish_date_str or str(publish_date_str).strip() in ("No Date", "None", ""):
        return 0.50

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]

    pub_date = None
    date_clean = str(publish_date_str).strip()
    for fmt in formats:
        try:
            pub_date = datetime.strptime(date_clean, fmt)
            break
        except ValueError:
            continue

    if pub_date is None:
        # Try extracting standard ISO date YYYY-MM-DD via regex
        import re
        match = re.search(r'\d{4}-\d{2}-\d{2}', date_clean)
        if match:
            try:
                pub_date = datetime.strptime(match.group(0), "%Y-%m-%d")
            except ValueError:
                return 0.50
        else:
            return 0.50

    now = datetime.now()
    age_days = max((now - pub_date).days, 0)

    if age_days <= 30:
        return 1.00
    elif age_days <= 365:
        # Linear interpolation from 1.00 (day 30) to 0.60 (day 365)
        return round(1.00 - (0.40 * (age_days - 30) / (365 - 30)), 4)
    elif age_days <= 1825:
        # Linear interpolation from 0.60 (day 365) to 0.10 (day 1825)
        return round(0.60 - (0.50 * (age_days - 365) / (1825 - 365)), 4)
    else:
        return 0.10

# ================================================================
# TEAM A RETRIEVAL PIPELINE CLASS
# ================================================================

class TeamARetrievalPipeline:
    """
    Module A: Data Ingestion & Retrieval Pipeline for Team A.
    Executes live web search + FAISS retrieval and scores evidence using the 4-factor formula:
        R(d) = 0.25·BM25 + 0.25·cosine + 0.25·domain + 0.25·recency
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        tavily_api_key: Optional[str] = None
    ):
        if HAS_ST:
            print(f"[TeamA Engine] Initializing Sentence Transformer: '{model_name}'...")
            self.embedding_model = SentenceTransformer(model_name)
            self.tfidf_svd_model = None
        else:
            print("[TeamA Engine] SentenceTransformer not available; loading TF-IDF+SVD fallback...")
            self.embedding_model = None
            self.tfidf_svd_model = None
            # Try to load pre-built TF-IDF+SVD model
            _tfidf_path = os.path.join(BASE_DIR, "data", "tfidf_svd_model.pkl")
            if HAS_SKLEARN and os.path.exists(_tfidf_path):
                try:
                    with open(_tfidf_path, "rb") as _f:
                        self.tfidf_svd_model = pickle.load(_f)
                    print(f"[TeamA Engine] Loaded TF-IDF+SVD model from: {_tfidf_path}")
                except Exception as _e:
                    print(f"[TeamA Engine] Warning: Could not load TF-IDF+SVD model: {_e}")
            else:
                print("[TeamA Engine] No TF-IDF+SVD model found. FAISS search will be skipped.")

        self.domain_evaluator = HybridDomainCredibility()

        # Initialize Web Scraper (Tavily + Trafilatura)
        self.scraper = HybridScraper(api_key=tavily_api_key)

        # Initialize FAISS & BM25 local corpus if present
        self.faiss_index = None
        self.metadata = []
        self.bm25_index = None

        if HAS_FAISS and index_path and metadata_path and os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                print(f"[TeamA Engine] Loading FAISS index: {index_path}")
                self.faiss_index = faiss.read_index(index_path)

                print(f"[TeamA Engine] Loading FAISS metadata: {metadata_path}")
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)

                if HAS_BM25:
                    tokenized_corpus = []
                    for doc in self.metadata:
                        text = doc.get("cleaned_text") or doc.get("statement", "")
                        tokenized_corpus.append(str(text).lower().split())
                    self.bm25_index = BM25Okapi(tokenized_corpus)
                    print(f"[TeamA Engine] Built BM25 index over {len(self.metadata)} historical documents.")
            except Exception as e:
                print(f"[TeamA Engine] Warning: Failed to load FAISS index or metadata: {e}")

    def generate_sub_claims(self, claim: str) -> List[str]:
        """
        Decomposes the main claim into heuristic sub-claims for targeted search & verification.
        """
        claim_clean = claim.strip()
        sub_claims = [claim_clean]

        # Generate simple sub-queries if multi-word claim
        words = claim_clean.split()
        if len(words) > 6:
            # Generate key phrase sub-claim
            sub_claims.append(" ".join(words[:len(words)//2]))
            sub_claims.append(" ".join(words[len(words)//2:]))

        return list(dict.fromkeys(sub_claims))  # Preserve unique order

    def retrieve_live_web(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves live web evidence using Tavily API with Trafilatura fallback.
        """
        try:
            if not self.scraper.api_key:
                print("[TeamA Engine] Tavily API key not set. Skipping live web retrieval.")
                return []
            return self.scraper.search_and_extract(query=query, max_results=max_results)
        except Exception as e:
            print(f"[TeamA Engine] Live web retrieval error: {e}")
            return []

    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """
        Encodes a query string into a 384-dim vector.
        Uses sentence-transformers if available, else TF-IDF+SVD fallback.
        """
        if self.embedding_model is not None:
            vec = np.array(self.embedding_model.encode([query])).astype("float32")
            faiss.normalize_L2(vec)
            return vec
        elif self.tfidf_svd_model is not None:
            tfidf = self.tfidf_svd_model["tfidf"]
            svd   = self.tfidf_svd_model["svd"]
            sparse = tfidf.transform([query])
            dense  = svd.transform(sparse).astype("float32")
            vec    = sk_normalize(dense, norm="l2")
            return vec
        return None

    def retrieve_faiss(self, query: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves historical claim evidence from FAISS vector store.
        Uses sentence-transformers or TF-IDF+SVD fallback for query encoding.
        """
        if not self.faiss_index or not self.metadata:
            return []

        try:
            query_emb = self._encode_query(query)
            if query_emb is None:
                print("[TeamA Engine] No query encoder available. Skipping FAISS search.")
                return []
            distances, indices = self.faiss_index.search(query_emb, top_n)

            faiss_results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1 or idx >= len(self.metadata):
                    continue
                doc = self.metadata[idx]
                faiss_results.append({
                    "source_id": f"faiss_{idx}",
                    "source_url": doc.get("url", "http://liar-dataset.internal"),
                    "title": doc.get("statement", "")[:80],
                    "source_type": "faiss_dataset",
                    "text_snippet": doc.get("cleaned_text") or doc.get("statement", ""),
                    "published_date": doc.get("publish_date") or doc.get("context", ""),
                    "author": doc.get("speaker", "Unknown Speaker"),
                    "organization": doc.get("job_title", "Dataset Record")
                })
            return faiss_results
        except Exception as e:
            print(f"[TeamA Engine] FAISS search error: {e}")
            return []

    def score_and_rank_evidence(
        self,
        claim: str,
        raw_evidence: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Applies the 4-factor scoring formula R(d) = 0.25·s₁ + 0.25·s₂ + 0.25·s₃ + 0.25·s₄ to raw evidence items.

        Returns top_k items sorted by combined_reliability R(d) descending.
        """
        if not raw_evidence:
            return []

        snippets = [item.get("text_snippet", "") for item in raw_evidence]
        tokenized_snippets = [str(s).lower().split() for s in snippets]
        claim_tokens = set(claim.lower().split())

        # 1. s₁: BM25 Lexical Score (with word-overlap fallback)
        if HAS_BM25 and len(tokenized_snippets) > 0:
            bm25_engine = BM25Okapi(tokenized_snippets)
            bm25_raw_scores = bm25_engine.get_scores(claim.lower().split())
            max_bm25 = np.max(bm25_raw_scores) if len(bm25_raw_scores) > 0 and np.max(bm25_raw_scores) > 0 else 1.0
            bm25_norm_scores = [round(float(score / max_bm25), 4) for score in bm25_raw_scores]
        else:
            # Fallback word overlap ratio
            bm25_norm_scores = []
            for snip_tokens in tokenized_snippets:
                snip_set = set(snip_tokens)
                overlap = len(claim_tokens.intersection(snip_set))
                ratio = overlap / len(claim_tokens) if claim_tokens else 0.0
                bm25_norm_scores.append(round(min(ratio * 1.5, 1.0), 4))

        # 2. s₂: Cosine Similarity Score (with Jaccard similarity fallback)
        if HAS_ST and self.embedding_model is not None:
            claim_embedding = self.embedding_model.encode(claim, convert_to_tensor=True)
            snippet_embeddings = self.embedding_model.encode(snippets, convert_to_tensor=True)
            cosine_sims = util.cos_sim(claim_embedding, snippet_embeddings)[0].tolist()
            cosine_norm_scores = [round(max(float(sim), 0.0), 4) for sim in cosine_sims]
        else:
            # Fallback Jaccard index
            cosine_norm_scores = []
            for snip_tokens in tokenized_snippets:
                snip_set = set(snip_tokens)
                union_len = len(claim_tokens.union(snip_set))
                inter_len = len(claim_tokens.intersection(snip_set))
                jaccard = inter_len / union_len if union_len > 0 else 0.0
                cosine_norm_scores.append(round(min(jaccard * 2.0, 1.0), 4))

        processed_evidence = []
        for idx, item in enumerate(raw_evidence):
            url = item.get("source_url", "")
            domain = self.domain_evaluator.extract_domain(url) or "unknown"

            # 3. s₃: Domain Credibility Score (Hybrid: Hardlist -> MBFC -> Dynamic)
            domain_cred, cred_source = self.domain_evaluator.evaluate(url)

            # 4. s₄: Recency Score
            pub_date = item.get("published_date")
            recency_score = compute_recency_score(pub_date)

            s1 = bm25_norm_scores[idx]
            s2 = cosine_norm_scores[idx]
            s3 = round(domain_cred, 4)
            s4 = recency_score

            # Combined Reliability: R(d) = 0.25·s₁ + 0.25·s₂ + 0.25·s₃ + 0.25·s₄
            combined_reliability = round(0.25 * s1 + 0.25 * s2 + 0.25 * s3 + 0.25 * s4, 4)

            # Extract author / organization
            author = item.get("author") or self._extract_author(item)
            org = item.get("organization") or domain.split(".")[0].upper()

            processed_evidence.append({
                "source_domain": domain,
                "author": author,
                "organization": org,
                "content": item.get("text_snippet", ""),
                "url": url,
                "bm25_score": s1,
                "cosine_score": s2,
                "domain_credibility": s3,
                "credibility_source": cred_source,
                "recency_score": s4,
                "combined_reliability": combined_reliability,
                "retrieval_rank": 0  # To be set after sorting
            })

        # Sort by combined_reliability descending
        processed_evidence.sort(key=lambda x: x["combined_reliability"], reverse=True)

        # Truncate to top_k and assign retrieval_rank
        top_evidence = processed_evidence[:top_k]
        for rank, item in enumerate(top_evidence, start=1):
            item["retrieval_rank"] = rank

        return top_evidence

    def _extract_author(self, item: Dict[str, Any]) -> str:
        domain = self.domain_evaluator.extract_domain(item.get("source_url", ""))
        if "who.int" in domain:
            return "World Health Organization"
        elif "reuters" in domain:
            return "Reuters Staff"
        elif "apnews" in domain:
            return "Associated Press"
        elif "bbc" in domain:
            return "BBC News"
        elif "factcheck" in domain:
            return "FactCheck.org Staff"
        elif "snopes" in domain:
            return "Snopes Fact Checker"
        return "Editorial Team"

    def process_claim(
        self,
        claim: str,
        original_language: str = "en",
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Executes full Module A pipeline:
          1. Sub-claim decomposition
          2. Multi-source evidence retrieval (Tavily live web + FAISS)
          3. 4-factor scoring R(d) = 0.25·s₁ + 0.25·s₂ + 0.25·s₃ + 0.25·s₄
          4. Adheres strictly to Module A -> Module B Inter-Team Data Contract.
        """
        start_time = time.time()

        normalized_claim = claim.strip()
        sub_claims = self.generate_sub_claims(normalized_claim)

        # 1. Retrieve raw evidence
        raw_web_evidence = self.retrieve_live_web(normalized_claim, max_results=top_k * 2)
        raw_faiss_evidence = self.retrieve_faiss(normalized_claim, top_n=top_k * 2)

        raw_all = raw_web_evidence + raw_faiss_evidence

        # Deduplicate evidence by content/URL
        seen_urls = set()
        unique_raw = []
        for item in raw_all:
            u = item.get("source_url", "")
            if u not in seen_urls:
                seen_urls.add(u)
                unique_raw.append(item)

        # 2. Apply 4-Factor R(d) Scoring & Ranking
        scored_evidence = self.score_and_rank_evidence(
            claim=normalized_claim,
            raw_evidence=unique_raw,
            top_k=top_k
        )

        # Indexes are immutable while the service is running.  Cache and index
        # maintenance are deliberately offline scripts, not request side effects.
        elapsed_ms = int((time.time() - start_time) * 1000)
        top_domain = scored_evidence[0]["source_domain"] if scored_evidence else "N/A"

        # 4. Format exact Inter-Team Data Contract JSON (Module A -> Module B)
        output_contract = {
            "claim": normalized_claim,
            "original_language": original_language,
            "normalized_claim": normalized_claim,
            "sub_claims": sub_claims,
            "evidence": scored_evidence,
            "metadata": {
                "retrieval_time_ms": elapsed_ms,
                "num_sources_total": len(unique_raw),
                "num_sources_kept": len(scored_evidence),
                "top_source_domain": top_domain
            }
        }

        return output_contract

# ── Alias for backward compatibility ────────────────────────────
EvidenceRetrievalPipeline = TeamARetrievalPipeline

# Quick standalone test execution
if __name__ == "__main__":
    INDEX_PATH    = os.path.join(BASE_DIR, "indexes", "truthlens.faiss")
    METADATA_PATH = os.path.join(BASE_DIR, "indexes", "metadata.pkl")
    API_KEY       = os.getenv("TAVILY_API_KEY", "")

    pipeline = TeamARetrievalPipeline(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        tavily_api_key=API_KEY
    )

    test_claim = "COVID-19 vaccines cause infertility"
    print(f"\n--- Processing Claim: '{test_claim}' ---")
    result = pipeline.process_claim(test_claim, top_k=5)

    import json
    print(json.dumps(result, indent=2, default=str))

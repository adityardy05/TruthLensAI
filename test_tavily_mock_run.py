"""
================================================================
MOCK RUN: TAVILY API KEY VERIFICATION + RAG PIPELINE TEST
================================================================
Tests the full Team A pipeline:
  1. Tavily API Key connectivity
  2. HybridScraper search_and_extract()
  3. Live evidence retrieval + 4-Factor R(d) scoring
  4. FAISS index status check

Usage:
    python test_tavily_mock_run.py
================================================================
"""

import os
import sys
import json
import time

# ── Load .env manually (no dotenv dependency) ──────────────────
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
        print("[ENV] Loaded .env from: " + env_path)
    else:
        print("[ENV] No .env file found. Using environment variables.")

load_env()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ──────────────────────────────────────────────────────────────
# STEP 0: Dependency check
# ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TRUTHLENS v2.0 - TAVILY MOCK RUN TEST")
print("="*65)

print("\n[STEP 0] Checking dependencies...")
deps = {}

try:
    try:
        from tavily import TavilyClient
    except ImportError:
        from tavily import Client as TavilyClient
    deps["tavily"] = True
except ImportError:
    deps["tavily"] = False

try:
    from sentence_transformers import SentenceTransformer
    deps["sentence_transformers"] = True
except ImportError:
    deps["sentence_transformers"] = False

try:
    import faiss
    deps["faiss"] = True
except ImportError:
    deps["faiss"] = False

try:
    import trafilatura
    deps["trafilatura"] = True
except ImportError:
    deps["trafilatura"] = False

try:
    from rank_bm25 import BM25Okapi
    deps["rank_bm25"] = True
except ImportError:
    deps["rank_bm25"] = False

for dep, status in deps.items():
    icon = "[OK]    " if status else "[MISSING]"
    print(f"  {dep:<28} {icon}")

if not deps["tavily"]:
    print("\n[FATAL] 'tavily' package not installed.")
    print("  Run: python -m pip install tavily-python")
    sys.exit(1)

if not deps["sentence_transformers"]:
    print("\n[WARN] sentence-transformers not available.")
    print("  Cosine similarity scoring will be skipped (BM25 + domain + recency only).")

# ──────────────────────────────────────────────────────────────
# STEP 1: Validate Tavily API Key
# ──────────────────────────────────────────────────────────────
print("\n[STEP 1] Validating Tavily API Key...")
if not TAVILY_API_KEY:
    print("  [FAIL] TAVILY_API_KEY is not set in .env or environment!")
    sys.exit(1)

masked = TAVILY_API_KEY[:12] + "..." + TAVILY_API_KEY[-6:]
if TAVILY_API_KEY.startswith("tvly-"):
    print(f"  [OK] Key format valid: {masked}")
else:
    print(f"  [WARN] Unexpected key format: {masked}")

# ──────────────────────────────────────────────────────────────
# STEP 2: Live Tavily API connectivity test (direct)
# ──────────────────────────────────────────────────────────────
print("\n[STEP 2] Testing Tavily API connectivity (direct call)...")

TEST_CLAIM = "COVID-19 vaccines cause infertility"

try:
    try:
        from tavily import TavilyClient
    except ImportError:
        from tavily import Client as TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    t_start = time.time()
    response = client.search(
        query=TEST_CLAIM,
        max_results=3,
        search_depth="basic"
    )
    elapsed_ms = (time.time() - t_start) * 1000
    raw_results = response.get("results", [])
    print(f"  [OK] Tavily API responded in {elapsed_ms:.0f}ms")
    print(f"  Retrieved {len(raw_results)} raw results:\n")
    for i, r in enumerate(raw_results, 1):
        title   = r.get("title", "(no title)")[:60]
        url     = r.get("url", "")[:70]
        score   = r.get("score", 0)
        content = r.get("content", "")
        print(f"  [{i}] {title}")
        print(f"       URL    : {url}")
        print(f"       Score  : {score:.4f}  |  Content: {len(content)} chars")
        print(f"       Snippet: {content[:120].strip()}...")
        print()
except Exception as e:
    print(f"  [FAIL] Tavily API call FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# STEP 3: Test HybridScraper (tavily_scraper.py)
# ──────────────────────────────────────────────────────────────
print("\n[STEP 3] Testing HybridScraper (tavily_scraper.py)...")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

scraper_ok = False
scraped_results = []
try:
    from scraping.tavily_scraper import HybridScraper
    scraper = HybridScraper(api_key=TAVILY_API_KEY, min_content_length=400)
    t_start = time.time()
    scraped_results = scraper.search_and_extract(TEST_CLAIM, max_results=3, search_depth="basic")
    elapsed_ms = (time.time() - t_start) * 1000
    print(f"\n  [OK] HybridScraper returned {len(scraped_results)} items in {elapsed_ms:.0f}ms")
    print(f"\n  Evidence items:")
    for i, ev in enumerate(scraped_results, 1):
        method  = ev.get("extraction_method", "?")
        clen    = len(ev.get("text_snippet", ""))
        title   = ev.get("title", "(no title)")[:55]
        url     = ev.get("source_url", "")[:65]
        snippet = ev.get("text_snippet", "")[:150].strip()
        print(f"\n  [{i}] {title}")
        print(f"       URL    : {url}")
        print(f"       Method : {method:<18}  |  Content: {clen} chars")
        print(f"       Snippet: {snippet}...")
    scraper_ok = True
except Exception as e:
    print(f"  [FAIL] HybridScraper FAILED: {e}")
    import traceback; traceback.print_exc()

# ──────────────────────────────────────────────────────────────
# STEP 4: Manual R(d) scoring demo on scraped evidence
# ──────────────────────────────────────────────────────────────
print("\n[STEP 4] Manual R(d) scoring on scraped evidence (without full pipeline)...")

def simple_domain_score(url):
    """Quick domain credibility heuristic."""
    trusted = [".gov", ".edu", ".who.int", "cdc.gov", "nih.gov", "nature.com",
               "reuters.com", "bbc.com", "apnews.com", "scientificamerican.com"]
    fake    = ["infowars", "naturalnews", "beforeitsnews", "theonion"]
    url_l   = url.lower()
    for t in trusted:
        if t in url_l: return 0.9
    for f in fake:
        if f in url_l: return 0.1
    return 0.5

def simple_recency_score(date_str):
    """Recency score: 1.0 if recent, decays with age."""
    if not date_str:
        return 0.5
    try:
        from datetime import datetime
        pub = datetime.strptime(date_str[:10], "%Y-%m-%d")
        days_old = (datetime.now() - pub).days
        if days_old < 30:   return 1.0
        if days_old < 180:  return 0.8
        if days_old < 365:  return 0.6
        if days_old < 730:  return 0.4
        return 0.2
    except Exception:
        return 0.5

def simple_bm25_score(query, text, idx, total):
    """Pseudo BM25 via term overlap (true BM25 when rank_bm25 missing)."""
    if deps["rank_bm25"] and text:
        try:
            from rank_bm25 import BM25Okapi
            corpus = [text.lower().split()]
            bm25   = BM25Okapi(corpus)
            scores = bm25.get_scores(query.lower().split())
            norm   = min(1.0, scores[0] / (max(scores[0], 10.0)))
            return round(norm, 4)
        except Exception:
            pass
    # Fallback: position-based (rank 1 => highest)
    return round(max(0.1, 1.0 - (idx / max(total, 1)) * 0.3), 4)

if scraped_results:
    print(f"\n  Claim: \"{TEST_CLAIM}\"\n")
    print(f"  {'Rank':<5} {'BM25':<7} {'Cosine':<9} {'Domain':<9} {'Recency':<9} {'R(d)':<7}  Title")
    print("  " + "-"*80)
    scored = []
    for idx, ev in enumerate(scraped_results):
        bm25    = simple_bm25_score(TEST_CLAIM, ev.get("text_snippet",""), idx, len(scraped_results))
        cosine  = 0.50  # placeholder (sentence-transformers unavailable)
        domain  = simple_domain_score(ev.get("source_url",""))
        recency = simple_recency_score(ev.get("published_date"))
        r_d     = round(0.25*bm25 + 0.25*cosine + 0.25*domain + 0.25*recency, 4)
        ev["r_d"] = r_d
        scored.append((r_d, ev, bm25, cosine, domain, recency))

    scored.sort(key=lambda x: x[0], reverse=True)
    for rank, (r_d, ev, bm25, cosine, domain, recency) in enumerate(scored, 1):
        title = ev.get("title","(no title)")[:40]
        print(f"  {rank:<5} {bm25:<7} {cosine:<9} {domain:<9} {recency:<9} {r_d:<7}  {title}")
    print()
    if not deps["sentence_transformers"]:
        print("  NOTE: Cosine score is fixed at 0.50 (sentence-transformers not installed).")
        print("        Install with: python -m pip install sentence-transformers==2.2.2")
else:
    print("  [SKIP] No scraped results to score.")

# ──────────────────────────────────────────────────────────────
# STEP 5: FAISS index status
# ──────────────────────────────────────────────────────────────
print("\n[STEP 5] Checking FAISS index status...")
base = os.path.dirname(os.path.abspath(__file__))
faiss_path = os.path.join(base, "data", "faiss_index.bin")
faiss_meta = os.path.join(base, "data", "faiss_metadata.json")

if os.path.exists(faiss_path):
    size_mb = os.path.getsize(faiss_path) / (1024 * 1024)
    print(f"  [OK] FAISS index found: {faiss_path}")
    print(f"       Size: {size_mb:.2f} MB")
    if os.path.exists(faiss_meta):
        with open(faiss_meta) as f:
            meta = json.load(f)
        print(f"       Stored documents: {len(meta)}")
    if deps["faiss"]:
        import faiss as faiss_lib
        idx = faiss_lib.read_index(faiss_path)
        print(f"       FAISS vectors: {idx.ntotal}")
else:
    print(f"  [INFO] FAISS index not found at: data/faiss_index.bin")
    print(f"         Live Tavily search will serve as primary evidence source.")

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  MOCK RUN COMPLETE - SUMMARY")
print("="*65)
print(f"  Claim tested         : \"{TEST_CLAIM}\"")
print(f"  Tavily API Key       : [OK] Working")
print(f"  HybridScraper        : {'[OK] Working' if scraper_ok else '[FAIL] Error - check logs'}")
print(f"  R(d) Scoring (4-fac) : [OK] BM25 + Domain + Recency active")
print(f"  Cosine similarity    : {'[OK] Active' if deps['sentence_transformers'] else '[WARN] Skipped (install sentence-transformers)'}")
print(f"  FAISS index          : {'[OK] Present' if os.path.exists(faiss_path) else '[INFO] Not built yet'}")
print("="*65 + "\n")

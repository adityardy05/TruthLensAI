"""
================================================================
FAISS INDEX BUILDER — TruthLens v2.0 (Team A)
================================================================
Reads data/processed/cleaned_news.csv and builds:
  - data/faiss_index.bin      : FAISS FlatIP vector index
  - data/faiss_metadata.pkl   : Pickle list of doc dicts

Uses sentence-transformers all-MiniLM-L6-v2 for embeddings.
Falls back to TF-IDF + SVD if sentence-transformers unavailable.

Usage:
    python build_faiss_index.py
================================================================
"""

import os
import sys
import csv
import time
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_INDEX = os.path.join(DATA_DIR, "faiss_index.bin")
OUT_META  = os.path.join(DATA_DIR, "faiss_metadata.pkl")
CSV_PATH  = os.path.join(DATA_DIR, "processed", "cleaned_news.csv")

EMBED_DIM   = 384          # all-MiniLM-L6-v2 dim
BATCH_SIZE  = 256          # how many docs to embed at once
MAX_DOCS    = 5000         # cap for speed (set None for all 12k)

print("\n" + "="*65)
print("  FAISS INDEX BUILDER — TruthLens v2.0")
print("="*65)

# ── Check dependencies ─────────────────────────────────────────
HAS_FAISS = False
HAS_ST    = False

try:
    import faiss
    HAS_FAISS = True
    print("[OK]  faiss installed")
except ImportError:
    print("[WARN] faiss not installed — installing now...")
    os.system(f"{sys.executable} -m pip install faiss-cpu 2>&1")
    try:
        import faiss
        HAS_FAISS = True
        print("[OK]  faiss-cpu installed successfully")
    except ImportError:
        print("[FAIL] Could not install faiss-cpu. Aborting.")
        sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
    print("[OK]  sentence-transformers available")
except ImportError:
    print("[WARN] sentence-transformers not found — will use TF-IDF fallback")

# ── Load CSV ───────────────────────────────────────────────────
print(f"\n[STEP 1] Loading dataset: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    print(f"[FAIL] CSV not found at: {CSV_PATH}")
    sys.exit(1)

docs = []
with open(CSV_PATH, encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = (row.get("cleaned_text") or row.get("statement") or "").strip()
        if text:
            docs.append({
                "id":               row.get("id", ""),
                "label":            row.get("label", ""),
                "statement":        row.get("statement", ""),
                "cleaned_text":     text,
                "speaker":          row.get("speaker", ""),
                "subject":          row.get("subject", ""),
                "context":          row.get("context", ""),
                "url":              f"http://liar-dataset.internal/{row.get('id','')}",
            })
        if MAX_DOCS and len(docs) >= MAX_DOCS:
            break

total_docs = len(docs)
print(f"  Loaded {total_docs} documents (capped at {MAX_DOCS or 'all'})")

# ── Generate Embeddings ────────────────────────────────────────
print(f"\n[STEP 2] Generating embeddings (dim={EMBED_DIM})...")

if HAS_ST:
    print("  Using sentence-transformers all-MiniLM-L6-v2 ...")
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    texts  = [d["cleaned_text"] for d in docs]
    t0     = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")
    elapsed = time.time() - t0
    print(f"  Embedding done in {elapsed:.1f}s — shape: {vectors.shape}")
else:
    # TF-IDF + TruncatedSVD fallback (sklearn)
    print("  Falling back to TF-IDF + TruncatedSVD (384 dims)...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        texts = [d["cleaned_text"] for d in docs]
        t0    = time.time()

        print("  Fitting TF-IDF vectorizer...")
        tfidf  = TfidfVectorizer(max_features=10000, sublinear_tf=True, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(texts)

        print("  Running TruncatedSVD (LSA) to 384 dims...")
        svd    = TruncatedSVD(n_components=EMBED_DIM, random_state=42)
        dense  = svd.fit_transform(tfidf_matrix).astype("float32")
        vectors = normalize(dense, norm="l2")

        elapsed = time.time() - t0
        print(f"  TF-IDF+SVD done in {elapsed:.1f}s — shape: {vectors.shape}")

        # Persist vectorizer for query-time use
        vect_path = os.path.join(DATA_DIR, "tfidf_svd_model.pkl")
        with open(vect_path, "wb") as f:
            pickle.dump({"tfidf": tfidf, "svd": svd}, f)
        print(f"  Saved TF-IDF+SVD model to: {vect_path}")

    except Exception as e:
        print(f"[FAIL] TF-IDF fallback failed: {e}")
        sys.exit(1)

# ── Build FAISS Index ──────────────────────────────────────────
print(f"\n[STEP 3] Building FAISS FlatIP index ({total_docs} vectors x {EMBED_DIM} dims)...")

# L2-normalize for cosine similarity via Inner Product
faiss.normalize_L2(vectors)

index = faiss.IndexFlatIP(EMBED_DIM)
index.add(vectors)

print(f"  Index built — total vectors: {index.ntotal}")

# ── Save to disk ───────────────────────────────────────────────
print(f"\n[STEP 4] Saving index and metadata...")
os.makedirs(DATA_DIR, exist_ok=True)

faiss.write_index(index, OUT_INDEX)
print(f"  Saved FAISS index  : {OUT_INDEX}  ({os.path.getsize(OUT_INDEX)/1024/1024:.2f} MB)")

with open(OUT_META, "wb") as f:
    pickle.dump(docs, f)
print(f"  Saved metadata PKL : {OUT_META}  ({os.path.getsize(OUT_META)/1024:.1f} KB)")

# ── Quick Sanity Check ─────────────────────────────────────────
print(f"\n[STEP 5] Quick sanity search test...")
TEST_QUERY = "COVID-19 vaccines cause infertility"

if HAS_ST:
    q_vec = model.encode([TEST_QUERY], convert_to_numpy=True).astype("float32")
else:
    from sklearn.preprocessing import normalize as sk_normalize
    with open(os.path.join(DATA_DIR, "tfidf_svd_model.pkl"), "rb") as f:
        vmod = pickle.load(f)
    q_tfidf = vmod["tfidf"].transform([TEST_QUERY])
    q_dense = vmod["svd"].transform(q_tfidf).astype("float32")
    q_vec   = sk_normalize(q_dense, norm="l2")

faiss.normalize_L2(q_vec)
dists, idxs = index.search(q_vec, 5)

print(f"  Query: \"{TEST_QUERY}\"")
print(f"  Top-5 FAISS matches:\n")
for rank, (idx, dist) in enumerate(zip(idxs[0], dists[0]), 1):
    if idx == -1: continue
    doc = docs[idx]
    print(f"  [{rank}] Score: {dist:.4f}  Label: {doc['label']:<12}  Speaker: {doc['speaker'][:20]}")
    print(f"       {doc['statement'][:80]}")
    print()

print("="*65)
print("  FAISS INDEX BUILD COMPLETE")
print(f"  Documents indexed : {index.ntotal}")
print(f"  Index file        : {OUT_INDEX}")
print(f"  Metadata file     : {OUT_META}")
print(f"  Embed method      : {'sentence-transformers' if HAS_ST else 'TF-IDF+SVD fallback'}")
print("="*65 + "\n")

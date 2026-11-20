# TruthLens Architecture (current)

> The older `v2.0-ARCHITECTURE.md` and `docs/project_documentation.md` describe a previous design
> (LIAR dataset index, Tavily). This file reflects the code.

## Components

| Layer | Implementation |
|---|---|
| Web/API | Flask (`backend/app.py`, `backend/api/verification.py`): `/api/v1/verify`, `/verify/image`, `/verify/url`, `/health` |
| Service | `backend/services/verification_service.py` (language detection, translation to English, Team A -> Team B) |
| Team A | `backend/rag/*` (below) |
| Team B | LangGraph (`backend/agents/team_b`): 3 personas in round 1, confidence gate 0.80, round 2 Fact Checker (R(d) >= 0.70), round 3 Logical Analyst (R(d) < 0.70), final judgment |
| Bot | `bot/` Telegram client calling the Flask API |

## Team A retrieval pipeline

| Module | Responsibility |
|---|---|
| `query_builder.py` | Deterministic cleaning and query construction (English query, plus the original-language query for non-English claims). No LLM. |
| `faiss_store.py` | FAISS store (`IndexFlatIP`, cosine) of permanent extracted evidence chunks and short-lived query anchors; SHA-256 content deduplication, manifest, aligned metadata |
| `searxng_client.py` | SearXNG JSON API client, result normalisation, URL de-duplication |
| `search_ranking.py` | `SearchRelevance = 0.70*MiniLM + 0.30*BM25` over title+snippet, Top 10 (max 3 per domain) |
| `robots.py`, `safe_fetch.py`, `article_fetcher.py` | robots.txt policy, SSRF-safe fetching, Trafilatura extraction, snippet fallback |
| `evidence_processing.py` | Chunking, prompt-injection and boilerplate cleanup, near-duplicate and syndication handling |
| `scoring.py` | `R(d) = 0.40 Relevance + 0.20 Quality + 0.25 Credibility + 0.15 Recency`, Top-5 selection |
| `domain_credibility.py` | Hardlist, MBFC cache (31 entries) and a heuristic fallback |
| `retriever.py` | Orchestrator; `process_claim(claim, original_language, top_k, original_claim=None)`; stale cache evidence triggers SearXNG |

### Routing
The claim is embedded and searched in FAISS. If the best cosine similarity is **>= 0.75**
(`FAISS_SIMILARITY_THRESHOLD`) the cached evidence is reused and SearXNG is not called. Otherwise SearXNG
is queried. 0.75 is a routing threshold only: it is not R(d) and not a truth probability.
The index holds two kinds of entries: evidence chunks and *query anchors* (the embedding of the claim
that retrieved a batch of chunks, linked to them). A claim is a short sentence and chunks are long
passages, so claim-to-chunk similarity is usually low; claim-to-earlier-claim similarity is a
like-for-like comparison. Evidence chunks and their embeddings are retained permanently. Only query anchors expire after `FAISS_CACHE_TTL_DAYS` (7). On every reuse, publication-date recency is recalculated; evidence below `FAISS_MIN_RECENCY_SCORE` (default 0.60, the existing one-year recency boundary) triggers a fresh SearXNG lookup. Search snippets are never cached. The separate exact-normalized-claim verdict cache uses SHA-256 and `VERDICT_CACHE_TTL_DAYS` (7); there is no semantic claim-similarity matching.

### Evidence types
`evidence_type` is `extracted` (full article text) or `snippet` (SearXNG title/snippet kept when robots.txt
disallows, the server refuses, the page cannot be extracted, or the fetch fails). Status fields:
`access_status`, `fetch_status`, `extraction_status`, `robots_allowed`, `fetch_error`.

### robots.txt policy
200: apply the rules. 404/other 4xx: allowed (recorded as unavailable). 401/403 on robots.txt: treated as
disallowed. 5xx/timeout: do not fetch (snippet kept). A 403 on the article itself is `SERVER_DENIED`, not a
robots denial, and is not retried.

### R(d)
All components are in [0, 1] and are returned on every evidence item. R(d) ranks; it never deletes. The final
Top 5 may mix strong, weak and snippet evidence so that Team B can reason over weak or conflicting items. At
most 2 chunks per URL, domain and syndication group are chosen, relaxed when there are too few candidates.
Fewer than 5 items are returned (with `insufficient_evidence`) rather than inventing evidence.

## Verdict cache\n\n`backend/services/verdict_cache.py` caches completed, verifiable responses by SHA-256 of the normalized exact claim. Entries expire after seven days. A cache hit reuses the previous verdict and evidence response; a miss runs Team A and Team B. No 90% or other semantic claim matching is performed.\n\n## Known limitations
* The embedding model is English-centric (`all-MiniLM-L6-v2`). Non-English claims are translated to English
  before retrieval; the original-language query is only an extra search query.
* SearXNG results depend on which engines are reachable; blocked engines are logged per request.
* DNS rebinding between the safety check and the connection is not fully closed (no IP pinning).
* R(d) values are lower on average than with the old formula; Team B's 0.70 / 0.80 thresholds were not
  recalibrated.


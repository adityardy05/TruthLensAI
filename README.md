# TruthLens

A multilingual, multi-agent fake-news verification system with a web UI and a Telegram bot.

```
claim -> Team A (evidence retrieval) -> Team B (LangGraph multi-agent reasoning) -> verdict
```

## Evidence retrieval (Team A)

```
claim -> query building -> FAISS cache lookup
      -> similarity >= 0.75 + fresh evidence : reuse cached evidence
      -> otherwise                           : SearXNG -> MiniLM + BM25 ranking -> Top 10 -> robots.txt -> fetch
                              -> Trafilatura (snippet fallback) -> chunk -> injection cleanup
      -> SHA-256 de-duplicate -> R(d) scoring -> Top 5 -> Team B
\nCompleted exact-claim verdicts are cached for 7 days; evidence chunks and embeddings are retained permanently.
```

See [architecture.md](architecture.md) for the details.

## Run

```powershell
# 1. SearXNG must be running with the JSON format enabled (search.formats: [html, json]).
#    Default URL: http://127.0.0.1:8888  (set SEARXNG_BASE_URL otherwise)
# 2. Configure
copy config\.env.example .env
# 3. Install and start
pip install -r requirements.txt
python -m scripts.init_index        # optional: creates the empty FAISS cache
python -m backend.app               # web UI + API on http://127.0.0.1:5000
```

## Tests

```powershell
python -m pytest tests evaluation/test_team_a_pipeline.py evaluation/test_api_endpoints.py
```

The retrieval tests are fully offline (mocked HTTP, fake embedder). Five tests in
`backend/agents/team_b/tests` fail on the original code as well and are unrelated to retrieval.

# API

`GET /api/v1/health` reports service readiness.

`POST /api/v1/verify` accepts `{ "claim": "..." }`.

`POST /api/v1/verify/image` accepts multipart `image` or `file`.

`POST /api/v1/verify/url` accepts `{ "url": "https://..." }`.

All verification endpoints return the original and normalized claim, detected language, subclaims, scored evidence, verdict, confidence, justification, recommendation, persona insights, and execution metadata.

# TruthLens v2.0 — Team A Retrieval & OCR API Documentation

This document specifies the endpoints, request/response formats, and scoring formulas for **Team A (Data Ingestion, OCR & Evidence Retrieval Module)** in TruthLens v2.0 Architecture.

---

## 1. Overview & Architectural Role

Team A owns **Module 0 (OCR & Multilingual Processing)** and **Module A (Data Ingestion & Retrieval)**.

Team A provides high-precision RAG retrieval and calculates the full 4-Factor Evidence Reliability score $R(d)$ for every retrieved evidence document:

$$R(d) = 0.25 \cdot s_1 + 0.25 \cdot s_2 + 0.25 \cdot s_3 + 0.25 \cdot s_4$$

- $s_1$: **BM25 Lexical Score** (normalized $[0, 1]$)
- $s_2$: **Semantic Cosine Similarity** (`sentence-transformers/all-MiniLM-L6-v2` 384-dim)
- $s_3$: **Hybrid Domain Credibility** (Hardlist $\rightarrow$ MBFC Cache $\rightarrow$ Dynamic Fallback)
- $s_4$: **Recency Score** ($<30\text{ days} = 1.0$, $<1\text{ year} = 0.6$, $>5\text{ years} = 0.1$)

Team B (Agentic Reasoning) consumes Team A's output directly without re-scoring evidence credibility.

---

## 2. Base URL

```text
http://127.0.0.1:5000
```

### Team B forwarding

Every successful `/retrieve` and `/process_image` request POSTs the same Module A -> Module B contract to Team B. The default Team B endpoint is `http://127.0.0.1:8000/analyze`. Set `TEAM_B_URL` in the project `.env` file if Team B uses another host or route.

```text
TEAM_B_URL=http://127.0.0.1:8000/analyze
TEAM_B_TIMEOUT=30
```

The Team A endpoint still returns the contract to its caller. If Team B is unavailable or returns a non-2xx response, Team A returns `502` with a forwarding error.

---

## 3. Endpoints

### A. Health Check
* **URL:** `/health`
* **Method:** `GET`
* **Response:**
```json
{
  "status": "healthy",
  "service": "TruthLens Team A - Data Ingestion & Retrieval API",
  "version": "v2.0",
  "architecture_spec": "TruthLens v2.0",
  "faiss_loaded": true,
  "tavily_configured": true
}
```

---

### B. Retrieve Evidence (`/retrieve` - Main Endpoint)
* **URL:** `/retrieve`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`

#### Request Body:
```json
{
  "claim": "Does whiskey cure COVID-19?",
  "original_language": "hindi",
  "top_k": 10
}
```

#### Response Body (Inter-Team Data Contract Module A $\rightarrow$ Module B):
```json
{
  "claim": "Does whiskey cure COVID-19?",
  "original_language": "hindi",
  "normalized_claim": "Does whiskey cure COVID-19?",
  "sub_claims": [
    "Does whiskey cure COVID-19?",
    "Does whiskey cure",
    "COVID-19?"
  ],
  "evidence": [
    {
      "source_domain": "who.int",
      "author": "World Health Organization",
      "organization": "WHO",
      "content": "Drinking alcohol does not protect against COVID-19 and can be dangerous...",
      "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
      "bm25_score": 0.88,
      "cosine_score": 0.92,
      "domain_credibility": 0.98,
      "credibility_source": "hardlist_trusted",
      "recency_score": 0.90,
      "combined_reliability": 0.92,
      "retrieval_rank": 1
    }
  ],
  "metadata": {
    "retrieval_time_ms": 3120,
    "num_sources_total": 20,
    "num_sources_kept": 10,
    "top_source_domain": "who.int"
  }
}
```

---

### C. Process Image Claim (`/process_image`)
* **URL:** `/process_image`
* **Method:** `POST`
* **Body:** `multipart/form-data` with key `file` or `image` containing the image file.
* **Process:** OpenCV Preprocessing $\rightarrow$ Multi-language Tesseract OCR $\rightarrow$ Language Detection $\rightarrow$ Google Translator $\rightarrow$ 4-Factor RAG Retrieval.

---

## 4. Integration Example for Team B / Team C (Python)

```python
import requests

payload = {
    "claim": "क्या व्हिस्की से COVID ठीक होता है?",
    "original_language": "hindi",
    "top_k": 5
}

response = requests.post("http://127.0.0.1:5000/retrieve", json=payload)
data = response.json()

# Access scored evidence
for item in data["evidence"]:
    print(f"Domain: {item['source_domain']} | R(d) Score: {item['combined_reliability']}")
```

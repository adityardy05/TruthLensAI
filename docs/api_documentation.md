# Fake News Detection: Retrieval API Documentation

This document explains how **Team B (Classification)** can interact with the Retrieval module's Flask API.

## 1. Overview
The Retrieval API is a lightweight microservice. Its only job is to receive a suspicious claim (query), search our pre-compiled FAISS vector database (containing the cleaned LIAR dataset), and instantly return the Top-K most relevant factual articles to serve as evidence.

Team B will feed this returned evidence into their final Classification Model (e.g., BERT/RoBERTa) to determine if the claim is `Real` or `Fake`.

## 2. Base URL
If running locally, the API will be hosted at:
```text
http://127.0.0.1:5000
```

## 3. Endpoints

### A. Health Check
* **URL:** `/health`
* **Method:** `GET`
* **Purpose:** Verify the API is running and the FAISS index is loaded.

**Response:**
```json
{
  "status": "healthy",
  "message": "Retrieval API is active."
}
```

---

### B. Retrieve Evidence (Main Endpoint)
* **URL:** `/retrieve`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`
* **Purpose:** Submit a claim and get back factual evidence articles.

#### Request Body
The API expects a JSON payload containing the `query` (the claim to fact-check) and optionally `top_k` (how many articles you want back; defaults to 5).

```json
{
  "query": "The economy bled $24 billion due to the government shutdown.",
  "top_k": 3
}
```

#### Success Response (200 OK)
Returns a JSON object containing the `retrieved_articles`. Each article includes a `score` (L2 distance, where lower is better), along with the metadata (title, text, URL).

```json
{
  "query": "The economy bled $24 billion due to the government shutdown.",
  "top_k_requested": 3,
  "retrieved_articles": [
    {
      "score": 0.8412,
      "title": "Government shutdown costs economy billions",
      "text": "According to new reports, the economy bled roughly 24 billion dollars...",
      "url": "http://example-news.com/article1",
      "publish_date": "2013-10-18"
    },
    {
      "score": 1.1054,
      "title": "Financial impacts of the shutdown",
      "text": "The 16-day federal closure had massive impacts on Wall Street...",
      "url": "http://example-news.com/article2",
      "publish_date": "2013-10-19"
    }
  ]
}
```

#### Error Responses
* **400 Bad Request:** If the JSON is improperly formatted or the `query` field is missing.
* **500 Internal Server Error:** If the FAISS index fails to search or the server crashes.

## 4. How Team B Can Use This in Python
Team B can integrate this API directly into their classification scripts using the `requests` library. Here is an example of how they will pull evidence before classifying:

```python
import requests

# 1. The claim Team B wants to evaluate
claim_to_check = "Drinking bleach cures the virus."

# 2. Make a request to Team A's Retrieval API
api_url = "http://127.0.0.1:5000/retrieve"
payload = {
    "query": claim_to_check,
    "top_k": 5
}

response = requests.post(api_url, json=payload)

if response.status_code == 200:
    data = response.json()
    evidence_articles = data['retrieved_articles']
    
    # 3. Team B concatenates the evidence text
    full_evidence = " ".join([article['text'] for article in evidence_articles])
    
    # 4. Team B passes the original claim + the full evidence into their deep learning model!
    # final_prediction = my_classification_model.predict(claim_to_check, full_evidence)
else:
    print("Error fetching evidence:", response.text)
```

## 5. Running the API Server
To start the API, simply navigate to the root of the project and run:
```bash
python api/app.py
```
*(Make sure you have installed Flask: `pip install Flask`)*

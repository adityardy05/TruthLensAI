# Fake News Detection: Retrieval (RAG) Module
**Comprehensive Project Documentation**

---

## 1. Abstract
The rapid proliferation of misinformation on digital platforms necessitates automated, highly accurate verification systems. This project details the design and implementation of the **Retrieval (RAG) Module** for a Fake News Detection system. By leveraging Dense Vector Embeddings and FAISS (Facebook AI Similarity Search), the module acts as a high-speed factual evidence retriever. It accepts raw, suspicious claims and queries a pre-compiled Knowledge Base of trusted news sources to supply contextually relevant facts to a downstream Classification model, thereby dramatically reducing AI hallucinations and improving classification accuracy.

---

## 2. Introduction
Detecting fake news computationally is a complex challenge. Traditional machine learning models often struggle because they lack real-world context; they attempt to classify a claim based solely on linguistic patterns rather than factual reality. 

To solve this, our architecture utilizes **Retrieval-Augmented Generation (RAG)**. The Retrieval Module functions as an automated research assistant. Before the classifier makes a decision on a claim, the Retrieval Module searches a curated Knowledge Base and retrieves the Top-K most relevant factual articles. This ensures the classifier grounds its final verdict in verified evidence.

---

## 3. Objectives
* **Evidence Extraction:** To accurately retrieve the Top-K most relevant factual articles for any given user query.
* **Semantic Search:** To move beyond basic keyword matching (TF-IDF) by utilizing deep learning sentence embeddings to capture the contextual meaning of a claim.
* **Scalability:** To implement FAISS, enabling millisecond search times across massive datasets.
* **Source Credibility:** To strictly gatekeep the Knowledge Base, allowing only high-quality, verified data from trusted domains.

---

## 4. Dataset
The Knowledge Base relies on a fusion of two highly regarded academic datasets, heavily curated and preprocessed:
1. **LIAR Dataset:** Contains over 12,800 short, human-labeled statements spanning six degrees of truthfulness (from *pants-fire* to *true*).
*Note: A custom `KnowledgeBaseManager` actively scrapes, filters, and deduplicates live articles from trusted sources (e.g., Reuters, AP News) to continuously expand the dataset.*

---

## 5. Architecture
The Retrieval Module operates entirely independently of the final classifier. 

```text
=======================================================================
                   RETRIEVAL MODULE ARCHITECTURE
=======================================================================

      [ Classifier / Frontend ]
             │ (JSON Query)
             ▼
+-------------------------+
|   API Gateway (Flask)   | 
+-------------------------+
             │
             ▼
+-------------------------+      +====================================+
| Embedding Generation    |      |  PRE-COMPUTED FAISS DATABASE       |
| (all-MiniLM-L6-v2)      | ---> |  (Cosine Similarity / L2 Distance) |
+-------------------------+      +====================================+
                                                │
                                                ▼
                                 +-------------------------+
                                 |  Retrieve Top-K Matches |
                                 +-------------------------+
             ▲                                  │
             │ (JSON Evidence Payload)          │
             └──────────────────────────────────┘
```

---

## 6. Folder Structure
The workspace is strictly modular to separate raw data, processing scripts, and server infrastructure.

```text
fake_news_detection/
├── api/                # Flask API endpoints (app.py)
├── config/             # Environment variables and API keys
├── data/               
│   ├── raw/            # Untouched datasets (LIAR)
│   └── processed/      # Cleaned master knowledge base
├── docs/               # Architecture and project documentation
├── evaluation/         # Metrics and scoring scripts (evaluator.py)
├── index/              # FAISS vector database (.faiss) and metadata (.pkl)
└── src/                
    ├── embeddings/     # FAISS indexing and retrieval scripts
    ├── preprocessing/  # Text cleaning, deduplication, and KB manager
    └── scraping/       # RSS and web scraping tools
```

---

## 7. Technologies
* **Programming Language:** Python 3.x
* **Web Framework:** Flask / FastAPI
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **Data Engineering:** Pandas, NumPy
* **NLP Processing:** NLTK, spaCy, Regular Expressions (RegEx)
* **Web Scraping:** BeautifulSoup4, Requests

---

## 8. Algorithms
* **Lemmatization:** Reduces words to their base linguistic root (e.g., "running" to "run") using spaCy to normalize search spaces.
* **SHA-256 Hashing:** Generates unique cryptographic fingerprints for article text to achieve $O(1)$ duplicate detection.
* **L2 (Euclidean) Distance:** Calculates the mathematical distance between the 384-dimensional query vector and the database vectors within FAISS to establish similarity rankings.
* **Mean Average Precision (MAP) & MRR:** Used in the evaluation module to mathematically score the quality of the ranked retrieval results.

---

## 9. Workflow
1. **Ingestion:** `kb_manager.py` accepts new scraped articles, enforces a domain whitelist (e.g., `reuters.com`), drops duplicates via SHA-256 hashing, and filters out short/junk text.
2. **Preprocessing:** `clean_data.py` strips HTML, emojis, URLs, and stop words, then lemmatizes the text.
3. **Indexing:** `generate_index.py` converts the cleaned text into 384-dimensional dense vectors and commits them to the FAISS index.
4. **Retrieval:** `app.py` accepts a REST POST request, embeds the query, searches FAISS, and returns the Top-5 most relevant articles alongside their original URLs and titles.

---

## 10. Screenshots
> *(Placeholder for UI Integration)*
> 
> `[Screenshot 1: The Flask API successfully returning a JSON payload of retrieved articles.]`
> 
> `[Screenshot 2: The FAISS indexing process running in the VS Code terminal.]`
>
> `[Screenshot 3: Output of the Evaluation module showing MAP and MRR scores.]`

---

## 11. API Documentation
The module exposes a REST API for Team B (Classification) to fetch evidence.

**Endpoint:** `POST /retrieve`
**Request Body:**
```json
{
  "query": "The COVID vaccine contains microchips.",
  "top_k": 3
}
```
**Response:**
```json
{
  "retrieved_articles": [
    {
      "score": 0.543,
      "title": "Fact Check: No microchips in vaccines",
      "text": "Medical experts have debunked the widespread conspiracy...",
      "url": "http://reuters.com/factcheck1"
    }
  ]
}
```

---

## 12. Results (Evaluation Metrics)
The Retrieval Module is continuously benchmarked using the built-in `evaluation/evaluator.py` script. Current simulated benchmarks indicate:
* **Precision@5:** > 80% (Highly relevant results in the Top 5).
* **Recall@5:** > 85% (Successfully capturing the necessary evidence).
* **Mean Reciprocal Rank (MRR):** Ensuring the absolute best piece of evidence is frequently placed at Rank #1 or #2.

---

## 13. Future Scope
* **Cross-Encoder Re-Ranking:** Implementing a two-stage retrieval pipeline where FAISS retrieves the Top-20, and a heavy Transformer model re-ranks them to the absolute best Top-5.
* **Knowledge Graph Integration:** Extracting specific entities (Politician names, Locations) and tracing their relationships dynamically.
* **Multilingual Retrieval:** Upgrading the embedding model to a multilingual variant (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) to fact-check fake news spreading across borders.

---

## 14. References
* Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.*
* Johnson, J., Douze, M., & Jégou, H. (2017). *Billion-scale similarity search with GPUs (FAISS).*
* Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*

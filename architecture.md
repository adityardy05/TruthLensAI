# Project Architecture: Fake News Detection

## 1. Project Overview
The Fake News Detection project is an AI-driven system designed to analyze news claims, statements, or articles and determine their authenticity. By combining external knowledge retrieval with advanced machine learning classifiers, the system acts as an automated fact-checker capable of mitigating the spread of misinformation.

## 2. Objective
To build a highly accurate, scalable Retrieval-Augmented Generation (RAG) system that can:
1. Accept suspicious claims as input.
2. Rapidly retrieve relevant, factual evidence from a vast database of trusted sources.
3. Pass this contextual evidence to a classification model to output a final truthfulness verdict.

## 3. Dataset Overview
The system relies on the following dataset acting as the evidence library and training ground:
* **LIAR Dataset:** Contains over 12,800 short, human-labeled statements spanning six degrees of truthfulness (from *pants-fire* to *true*). This serves as a rigorous testing ground for processing short-form queries and verifying political claims.

## 4. System Architecture (Retrieval Module)
The Retrieval Module operates entirely independently of the final classifier, acting as the search engine for the AI.

```text
=======================================================================
                   RETRIEVAL MODULE ARCHITECTURE
=======================================================================

      [ Team B / User ]
             │
             ▼
+-------------------------+
|     1. User Query       |  (e.g., A suspicious claim)
+-------------------------+
             │
             ▼
+-------------------------+
|   2. Text Cleaning      |  (Remove noise, lowercase, normalize)
+-------------------------+
             │
             ▼
+-------------------------+
| 3. Embedding Generation |  (Convert text into a dense vector)
+-------------------------+
             │
             ▼
+-------------------------+      +====================================+
|  4. FAISS Vector Search | <--- |  PRE-COMPUTED EVIDENCE DATABASE    |
|   (Cosine Similarity /  |      |  (LIAR Dataset converted to        |
|     L2 Distance)        |      |   Vectors)                         |
+-------------------------+      +====================================+
             │
             ▼
+-------------------------+
|  5. Retrieve Top-K      |  (Fetch the 5 most similar articles)
|   Similar Articles      |
+-------------------------+
             │
             ▼
+-------------------------+
| 6. Return Results to    |  (Send JSON payload with evidence)
|  Classification Module  |
+-------------------------+
             │
             ▼
      [ Team B / AI ]
```

## 5. Folder Structure
The project is modularized to ensure separation of concerns between teams and components.

```text
fake_news_detection/
│
├── api/                # FastAPI/Flask endpoints for the Retrieval Module
├── config/             # Project settings, environment variables, API keys
├── data/               # Raw and processed datasets (LIAR)
├── docs/               # Architecture documents and project documentation
├── evaluation/         # Retrieval metrics (e.g., Recall@5, MRR)
├── index/              # Compiled FAISS vector databases (.faiss files)
├── logs/               # Application and error logging
├── models/             # Downloaded embedding models (e.g., HuggingFace transformers)
├── src/                
│   ├── embeddings/     # Scripts for vectorization
│   ├── preprocessing/  # Data cleaning and formatting scripts
│   └── scraping/       # Scripts for live web-scraping of evidence
│
├── requirements.txt    # Python dependencies
└── .gitignore          # Git exclusion rules
```

## 6. Data Flow
1. **Ingestion:** Raw datasets are cleaned in `src/preprocessing/` and passed to `src/embeddings/`.
2. **Vectorization:** Text is converted into embeddings and stored persistently in the `index/` folder.
3. **Query:** A claim is received via the `api/`.
4. **Search:** The claim is embedded on the fly and compared against the FAISS index.
5. **Retrieval:** The `Top-K` matching documents are extracted from the `data/` folder based on the FAISS indices.
6. **Response:** A JSON package containing the evidence is returned.

## 7. Technologies
* **Language:** Python 3.x
* **Web Framework:** FastAPI or Flask
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embedding Models:** HuggingFace `SentenceTransformers` (e.g., `all-MiniLM-L6-v2`) or OpenAI API
* **Data Manipulation:** Pandas, NumPy
* **NLP Processing:** NLTK, spaCy, or regular expressions

## 8. Future Modules
To scale the system beyond static datasets, future iterations of the Retrieval module will include:
* **Live Web Scraping Pipeline:** Integrating NewsAPI and BeautifulSoup to fetch real-time evidence for claims that are not present in the static LIAR database.
* **Knowledge Graph Integration:** Expanding retrieval to include structured factual graphs, identifying relationships between known malicious domains or highly-correlated fake news spreaders.

## 9. Team Responsibilities
* **Team A (Retrieval / RAG Module):**
  * Responsible for data preprocessing, building the FAISS index, generating embeddings, and ensuring search speed/accuracy.
  * Must deliver a reliable API endpoint that returns highly relevant textual evidence for any given claim.
* **Team B (Classification Module):**
  * Responsible for designing, training, and evaluating the final deep learning model (e.g., BERT, RoBERTa).
  * Must consume the evidence provided by Team A and output a final confidence score and truthfulness label.

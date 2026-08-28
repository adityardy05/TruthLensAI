# Fake News Detection Architecture (TruthLens v2.0 - Team A Specification)

## 1. Overview
TruthLens is an AI-driven automated fact-checking system designed to evaluate suspicious claims, statements, and multimodal inputs (images/text) in real-time.

This document describes **Team A's responsibilities**:
- **Module 0**: OCR & Multilingual Processing
- **Module A**: Data Ingestion, Retrieval & 4-Factor Evidence Reliability Scoring $R(d)$

---

## 2. Team A System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         User Input Layer                            │
│  (Telegram Chat, Image with text, Plain text claim)                 │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│               MODULE 0: OCR & MULTILINGUAL PROCESSING               │
│                                                                      │
│  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────────┐ │
│  │ Image Preprocess │  │ Language Support│  │ Language Detection │ │
│  │ (OpenCV Denoise, │  │ (Hindi, Telugu, │  │ (lingua /          │ │
│  │  Adaptive Thresh)│  │  Tamil, Urdu,   │  │  langdetect)       │ │
│  └────────┬─────────┘  │  Bengali, Mar,  │  └────────────────────┘ │
│           │            │  English)       │                          │
│           ▼            └─────────────────┘  ┌────────────────────┐ │
│  ┌──────────────────┐                       │ Translation        │ │
│  │ Tesseract OCR    │                       │ (deep-translator)  │ │
│  │ (Multi-lang pack)│                       │ -> English Claim   │ │
│  └──────────────────┘                       └────────────────────┘ │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ English-Normalized Claim
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│          MODULE A: DATA INGESTION & RETRIEVAL (Team A)              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Evidence Retrieval                                             │ │
│  │ ├─ Tavily API (Web Search) + Trafilatura (Scraper Fallback)    │ │
│  │ └─ FAISS Vector Store (Historical Dataset Claims)              │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 4-Factor Evidence Scoring: R(d) = Σ λᵢsᵢ  (λᵢ = 0.25)           │ │
│  │                                                                │ │
│  │  s₁ = BM25(claim, evidence)          λ₁ = 0.25 (lexical)     │ │
│  │  s₂ = cosine_sim(claim, evidence)    λ₂ = 0.25 (semantic)    │ │
│  │       sentence-transformers all-MiniLM-L6-v2                  │ │
│  │                                                                │ │
│  │  s₃ = domain_credibility(source)     λ₃ = 0.25               │ │
│  │       3-Layer Hybrid: Hardlist → MBFC cache → Dynamic fallback │ │
│  │                       (TLD, HTTPS, WHOIS heuristics)           │ │
│  │                                                                │ │
│  │  s₄ = recency_score(pub_date)        λ₄ = 0.25 (temporal)    │ │
│  │       <30 days=1.0  <1yr=0.6  >5yr=0.1                       │ │
│  │                                                                │ │
│  │ Filter: Keep top-10 by R(d) combined_reliability score         │ │
│  └────────────────────────┬───────────────────────────────────────┘ │
│                           ▼                                          │
│  OUTPUT: Inter-Team Data Contract JSON (Module A -> Module B)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Evidence Scoring Formula: $R(d)$

Team A owns all evidence scoring and outputs a unified `combined_reliability` score $R(d)$ for every retrieved evidence document:

$$R(d) = 0.25 \cdot s_1 + 0.25 \cdot s_2 + 0.25 \cdot s_3 + 0.25 \cdot s_4$$

1. $s_1$ (**BM25 Score**): Normalized TF-IDF lexical keyword matching $[0, 1]$.
2. $s_2$ (**Semantic Cosine Score**): Cosine similarity using `sentence-transformers/all-MiniLM-L6-v2`.
3. $s_3$ (**Domain Credibility Score**):
   - **Layer 1 (Hardlist)**: Fast lookup for authoritative sites (`who.int`, `reuters.com`, `factcheck.org`, `snopes.com`, `thehindu.com`, etc., score 0.80–0.98) and known misinformation sites ($<0.30$).
   - **Layer 2 (MBFC Cache)**: Pre-computed rating cache for major global news outlets.
   - **Layer 3 (Dynamic Fallback)**: Dynamic heuristic evaluator checking TLDs (`.gov`/`.edu`/`.org`/`.com`), HTTPS, domain hyphens/numbers, and domain age for unknown Indian & global sites.
4. $s_4$ (**Recency Score**):
   - $< 30$ days: `1.0`
   - $< 365$ days (1 year): Interpolated `1.0 -> 0.6`
   - $< 1825$ days (5 years): Interpolated `0.6 -> 0.1`
   - $> 1825$ days: `0.1`

---

## 4. Team Responsibilities
- **Team A (Retrieval & OCR Module)**:
  - Responsible for OCR, language detection, translating non-English inputs, retrieving live web + vector evidence, and scoring evidence using the 4-factor formula $R(d)$.
  - Provides Flask API endpoints (`/retrieve`, `/process_image`, `/health`) serving the standardized Module A $\rightarrow$ Module B contract JSON.
- **Team B (Agentic Reasoning & LangGraph)**:
  - Consumes Team A's output directly using `combined_reliability` scores. Runs multi-persona agentic reasoning rounds and generates truthfulness verdicts.

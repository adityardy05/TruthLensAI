# TruthLens unified architecture

TruthLens is a single Flask application. `VerificationService` normalizes text, image, or URL input; retrieves and scores evidence; then invokes the existing three-persona LangGraph workflow. The graph runs claim decomposition, credibility aggregation, Round 1 Q&A, the existing confidence gate, optional Round 2 Q&A, and final judgment.

The frontend and Telegram bot are clients of the Flask API. Neither imports LangGraph or retrieval code. FAISS artifacts are built offline into `indexes/` and are immutable at runtime.

Round 3, post-verdict conversational mode, additional agents, and dynamic runtime indexing are explicitly out of scope.

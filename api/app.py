"""
================================================================
TEAM A: FLASK RETRIEVAL & OCR API SERVICE (TruthLens v2.0)
================================================================

Serves Team A's endpoints for Team B (Agentic RAG) & Team C (Telegram Bot):

Endpoints:
    GET  /health          — Service health check and model status
    POST /retrieve        — Primary RAG retrieval & 4-factor evidence scoring endpoint
                            Input JSON: {"claim": "..."} or {"query": "..."}
                            Returns Module A -> Module B Inter-Team Data Contract JSON
    POST /process_image   — OCR & translation + retrieval pipeline for image claims
                            Input: Multipart image file upload
                            Returns Module A -> Module B Data Contract + OCR metadata
================================================================
"""

import os
import sys
import time
import requests
try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    Flask = None
    request = None
    jsonify = None
    HAS_FLASK = False

# Include project root directory in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load .env file manually from project root
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()
    print(f"[TeamA API] Loaded environment variables from: {env_path}")

from src.embeddings.retriever import TeamARetrievalPipeline
from src.preprocessing.ocr_processor import process_image_input

if HAS_FLASK:
    app = Flask(__name__)
else:
    app = None
    print("[TeamA API] Warning: Flask package is not installed. Run 'pip install flask' to enable HTTP server.")

# System paths
INDEX_PATH = os.path.join(BASE_DIR, "index", "fake_news.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "index", "metadata.pkl")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
TEAM_B_URL = os.getenv("TEAM_B_URL", "http://127.0.0.1:8000/analyze")
TEAM_B_TIMEOUT = float(os.getenv("TEAM_B_TIMEOUT", "30"))


def forward_to_team_b(result_contract):
    """Send Team A's contract to Team B when a destination is configured."""
    if not TEAM_B_URL:
        return None

    try:
        response = requests.post(
            TEAM_B_URL,
            json=result_contract,
            timeout=TEAM_B_TIMEOUT
        )
        response.raise_for_status()
        return None
    except requests.RequestException as exc:
        return jsonify({
            "error": "Team B forwarding failed.",
            "details": str(exc),
            "team_b_url": TEAM_B_URL
        }), 502

print("================================================================")
print("Starting TruthLens Team A API Server (v2.0 Architecture)...")
print("================================================================")

# Initialize Team A Retrieval Pipeline globally
try:
    pipeline = TeamARetrievalPipeline(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        tavily_api_key=TAVILY_KEY
    )
    print("Team A Pipeline initialized successfully.")
except Exception as e:
    print(f"Warning: Team A Pipeline initialized with fallback: {e}")
    pipeline = TeamARetrievalPipeline(tavily_api_key=TAVILY_KEY)

if app is not None:
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint returning Team A service status."""
        return jsonify({
            "status": "healthy",
            "service": "TruthLens Team A - Data Ingestion & Retrieval API",
            "version": "v2.0",
            "architecture_spec": "TruthLens v2.0",
            "faiss_loaded": pipeline.faiss_index is not None,
            "tavily_configured": bool(pipeline.scraper.api_key),
            "timestamp": time.time()
        }), 200

    @app.route('/retrieve', methods=['POST'])
    def retrieve():
        """
        Primary endpoint serving Team B (Agentic RAG) and Team C (Telegram Bot).
        Accepts text claim and returns evidence scored with R(d) = 0.25·s₁ + 0.25·s₂ + 0.25·s₃ + 0.25·s₄.
        """
        if not request.is_json:
            return jsonify({"error": "Request body must be JSON."}), 400

        data = request.get_json()
        claim = data.get("claim") or data.get("query") or data.get("normalized_claim")

        if not claim or not str(claim).strip():
            return jsonify({"error": "'claim' or 'query' field is required in JSON body."}), 400

        original_lang = data.get("original_language", "en")
        top_k = int(data.get("top_k", 10))

        try:
            result_contract = pipeline.process_claim(
                claim=str(claim).strip(),
                original_language=original_lang,
                top_k=top_k
            )
            forwarding_error = forward_to_team_b(result_contract)
            if forwarding_error:
                return forwarding_error
            return jsonify(result_contract), 200
        except Exception as e:
            return jsonify({
                "error": f"Retrieval pipeline error: {str(e)}",
                "claim": claim
            }), 500

    @app.route('/process_image', methods=['POST'])
    def process_image():
        """
        Image input endpoint for Module 0 (OCR & Multilingual Processing).
        Accepts multipart/form-data with image file ('file' or 'image').
        Performs OCR -> Language Detection -> Translation -> Evidence Retrieval -> 4-factor Scoring.
        """
        image_file = None
        if 'file' in request.files:
            image_file = request.files['file']
        elif 'image' in request.files:
            image_file = request.files['image']

        if not image_file or image_file.filename == '':
            return jsonify({"error": "No image file provided in request. Use 'file' or 'image' field."}), 400

        try:
            image_bytes = image_file.read()
            ocr_result = process_image_input(image_bytes)
            english_claim = ocr_result.get("english_text", "").strip()

            if not english_claim:
                return jsonify({
                    "error": "OCR failed to extract legible text from image.",
                    "ocr_result": ocr_result
                }), 422

            result_contract = pipeline.process_claim(
                claim=english_claim,
                original_language=ocr_result.get("detected_language", "en"),
                top_k=int(request.args.get("top_k", 10))
            )

            result_contract["ocr_metadata"] = {
                "original_text": ocr_result.get("original_text"),
                "detected_language": ocr_result.get("detected_language"),
                "input_type": "image"
            }

            forwarding_error = forward_to_team_b(result_contract)
            if forwarding_error:
                return forwarding_error
            return jsonify(result_contract), 200
        except Exception as e:
            return jsonify({"error": f"Image processing error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

        
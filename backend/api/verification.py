from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request


api = Blueprint("api", __name__, url_prefix="/api/v1")


def _service():
    return current_app.extensions["verification_service"]


def _parse_top_k(payload: dict) -> int:
    """Parse and validate the requested evidence count."""
    try:
        top_k = int(payload.get("top_k", 5))
    except (TypeError, ValueError):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    return top_k


@api.get("/health")
def health():
    retriever = current_app.extensions["retriever"]

    return jsonify({
        "status": "healthy",
        "service": "TruthLens Flask API",
        "faiss_loaded": retriever.faiss_index is not None,
        "tavily_configured": bool(retriever.scraper.api_key),
    })


@api.post("/verify")
def verify_text():
    payload = request.get_json(silent=True) or {}

    try:
        claim = str(payload.get("claim", ""))
        original_language = str(payload.get("original_language", "en"))
        top_k = _parse_top_k(payload)

        result = _service().verify_text(
            claim,
            original_language=original_language,
            top_k=top_k,
        )

        return jsonify(result), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@api.post("/verify/image")
def verify_image():
    image = request.files.get("image") or request.files.get("file")

    if image is None or not image.filename:
        return jsonify({
            "error": "An image file is required in 'image' or 'file'."
        }), 400

    try:
        return jsonify(_service().verify_image(image.read())), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 422


@api.post("/verify/url")
def verify_url():
    payload = request.get_json(silent=True) or {}

    try:
        return jsonify(
            _service().verify_url(
                str(payload.get("url", ""))
            )
        ), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
import json
import queue
import threading

from backend.preprocessing.image_ocr import process_image_input


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
        **retriever.health(),
    })


@api.post("/ocr")
def extract_image_text():
    """Extract text for the image preview without running claim verification."""
    image = request.files.get("image") or request.files.get("file")
    if image is None or not image.filename:
        return jsonify({"error": "An image file is required in 'image' or 'file'."}), 400

    image_bytes = image.read()
    if not image_bytes:
        return jsonify({"error": "The uploaded image is empty."}), 400

    max_upload_bytes = current_app.config.get("MAX_CONTENT_LENGTH")
    if max_upload_bytes and len(image_bytes) > max_upload_bytes:
        return jsonify({"error": "Image exceeds the configured upload limit."}), 413

    result = process_image_input(image_bytes)
    if not result.get("original_text", "").strip():
        return jsonify({
            "error": "OCR could not extract legible text. Check Tesseract installation and language data."
        }), 422

    return jsonify({
        "original_text": result["original_text"],
        "english_text": result["english_text"],
        "detected_language": result["detected_language"],
        "input_type": "image",
    }), 200



_JOBS = {}
_JOBS_LOCK = threading.Lock()


def create_job() -> str:
    import uuid
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "running", "events": [], "result": None, "error": None}
    return job_id


def append_job_event(job_id: str, event: dict) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["events"].append({**event, "sequence": len(job["events"]) + 1})


def finish_job(job_id: str, result=None, error: str | None = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "failed" if error else "completed"
        job["result"] = result
        job["error"] = error


def get_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        return {
            "status": job["status"],
            "events": list(job["events"]),
            "result": job["result"],
            "error": job["error"],
        }


def _run_verification_job(job_id: str, app, service, mode: str, payload: dict, image_bytes=None):
    def progress_callback(event: dict) -> None:
        append_job_event(job_id, event)

    try:
        with app.app_context():
            append_job_event(job_id, {
                "stage": "started", "message": "Verification worker started.", "status": "running"
            })
            if mode == "image":
                result = service.verify_image(image_bytes or b"", progress_callback=progress_callback)
            elif mode == "url":
                result = service.verify_url(str(payload.get("url", "")), progress_callback=progress_callback)
            else:
                result = service.verify_text(
                    str(payload.get("claim", "")),
                    original_language=payload.get("original_language", "en"),
                    top_k=_parse_top_k(payload),
                    progress_callback=progress_callback,
                )
        finish_job(job_id, result=result)
    except Exception as error:
        finish_job(job_id, error=str(error))


@api.post("/verify/start")
def start_verification():
    """Start a verification worker and return its ID for progress polling."""
    payload = request.get_json(silent=True) or {}
    image_bytes = None
    if request.files:
        image = request.files.get("image") or request.files.get("file")
        if image is None or not image.filename:
            return jsonify({"error": "An image file is required."}), 400
        image_bytes = image.read()
        if not image_bytes:
            return jsonify({"error": "The uploaded image is empty."}), 400
        if len(image_bytes) > _service().max_upload_bytes:
            return jsonify({"error": "Image exceeds the configured upload limit."}), 413
        mode = "image"
    else:
        mode = str(payload.get("mode", "text"))
        if mode == "text" and not str(payload.get("claim", "")).strip():
            return jsonify({"error": "A non-empty claim is required."}), 400
        if mode == "url" and not str(payload.get("url", "")).strip():
            return jsonify({"error": "A URL is required."}), 400
        if mode not in ("text", "url"):
            return jsonify({"error": "mode must be text, image, or url."}), 400

    app = current_app._get_current_object()
    service = _service()
    job_id = create_job()
    worker = threading.Thread(
        target=_run_verification_job,
        args=(job_id, app, service, mode, payload, image_bytes),
        daemon=True,
        name="truthlens-" + job_id[:8],
    )
    worker.start()
    return jsonify({"job_id": job_id, "status": "running"}), 202


@api.get("/verify/progress/<job_id>")
def verification_progress(job_id: str):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Verification job not found or expired."}), 404
    return jsonify(job), 200


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


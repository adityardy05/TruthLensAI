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



@api.post("/verify/stream")
def verify_stream():
    """Stream real backend progress events followed by the final verification result."""
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", request.form.get("mode", "text")))
    top_k = _parse_top_k(payload)
    app = current_app._get_current_object()
    service = _service()
    events = queue.Queue()

    if mode == "image":
        image = request.files.get("image") or request.files.get("file")
        if image is None or not image.filename:
            return jsonify({"error": "An image file is required."}), 400
        image_bytes = image.read()
        if not image_bytes:
            return jsonify({"error": "The uploaded image is empty."}), 400
        if len(image_bytes) > service.max_upload_bytes:
            return jsonify({"error": "Image exceeds the configured upload limit."}), 413
        worker_call = lambda: service.verify_image(image_bytes, progress_callback=events.put)
    elif mode == "url":
        url = str(payload.get("url", "")).strip()
        if not url:
            return jsonify({"error": "A URL is required."}), 400
        worker_call = lambda: service.verify_url(url, progress_callback=events.put)
    else:
        claim = str(payload.get("claim", "")).strip()
        if not claim:
            return jsonify({"error": "A non-empty claim is required."}), 400
        language = str(payload.get("original_language", "en"))
        worker_call = lambda: service.verify_text(
            claim, original_language=language, top_k=top_k, progress_callback=events.put
        )

    def run_worker():
        try:
            with app.app_context():
                events.put({"event": "progress", "stage": "started",
                            "message": "Verification worker started.", "status": "running"})
                result = worker_call()
                events.put({"event": "result", "data": result})
        except Exception as error:
            events.put({"event": "error", "message": str(error)})
        finally:
            events.put(None)

    threading.Thread(target=run_worker, daemon=True).start()

    @stream_with_context
    def generate():
        yield "retry: 1500\\n\\n"
        while True:
            event = events.get()
            if event is None:
                yield "event: done\\ndata: {}\\n\\n"
                break
            event_type = event.pop("event", "progress")
            yield "event: " + event_type + "\\ndata: " + json.dumps(event, ensure_ascii=False) + "\\n\\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})




def _run_verification_job(job_id: str, service, mode: str, payload: dict, image_bytes: bytes | None = None):
    def progress_callback(event: dict) -> None:
        append_job_event(job_id, event)

    try:
        if mode == "image":
            result = service.verify_image(image_bytes or b"", progress_callback=progress_callback)
        elif mode == "url":
            result = service.verify_url(str(payload.get("url", "")), progress_callback=progress_callback)
        else:
            result = service.verify_text(
                str(payload.get("claim", "")),
                original_language=payload.get("original_language"),
                top_k=_parse_top_k(payload),
                progress_callback=progress_callback,
            )
        finish_job(job_id, result=result)
    except Exception as error:
        finish_job(job_id, error=str(error))


@api.post("/verify/start")
def start_verification():
    """Start verification in a worker so the UI can poll live progress."""
    mode = request.form.get("mode") if request.form else None
    image_bytes = None

    if request.files:
        image = request.files.get("image") or request.files.get("file")
        if image is None:
            return jsonify({"error": "An image file is required."}), 400
        image_bytes = image.read()
        if not image_bytes:
            return jsonify({"error": "The uploaded image is empty."}), 400
        mode = "image"
        payload = {}
    else:
        payload = request.get_json(silent=True) or {}
        mode = mode or str(payload.get("mode", "text"))

    if mode not in ("text", "image", "url"):
        return jsonify({"error": "mode must be text, image, or url."}), 400
    if mode == "text" and not str(payload.get("claim", "")).strip():
        return jsonify({"error": "A non-empty claim is required."}), 400
    if mode == "url" and not str(payload.get("url", "")).strip():
        return jsonify({"error": "A URL is required."}), 400

    service = _service()
    job_id = create_job()
    worker = Thread(
        target=_run_verification_job,
        args=(job_id, service, mode, payload, image_bytes),
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


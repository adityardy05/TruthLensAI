from __future__ import annotations

from flask import Flask, send_from_directory

from backend.api.verification import api
from backend.core.config import PROJECT_ROOT, load_settings
from backend.rag.retriever import TeamARetrievalPipeline
from backend.services.verification_service import VerificationService


def create_app(test_config: dict | None = None) -> Flask:
    settings = load_settings()
    frontend_dir = PROJECT_ROOT / "frontend"
    app = Flask(__name__, static_folder=str(frontend_dir), static_url_path="")
    app.config.from_mapping(MAX_CONTENT_LENGTH=settings.max_upload_bytes)
    if test_config:
        app.config.update(test_config)

    retriever = TeamARetrievalPipeline(
        index_path=str(settings.faiss_index_path),
        metadata_path=str(settings.faiss_metadata_path),
        tavily_api_key=settings.tavily_api_key,
    )
    app.extensions["retriever"] = retriever
    app.extensions["verification_service"] = VerificationService(retriever, settings.max_upload_bytes)
    app.register_blueprint(api)

    @app.get("/")
    def frontend():
        return send_from_directory(frontend_dir, "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

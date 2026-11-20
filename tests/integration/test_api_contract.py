from flask import Flask

from backend.api.verification import api


class FakeService:
    def verify_text(self, claim, original_language=None, top_k=5):
        return {"original_claim": claim, "verdict": "UNVERIFIABLE", "confidence": 0, "evidence": []}


class FakeRetriever:
    def health(self):
        return {"faiss_loaded": True, "faiss_entries": 3, "embedder_available": True,
                "searxng_configured": True, "searxng_reachable": True}


def test_text_endpoint_contract():
    app = Flask(__name__)
    app.extensions["verification_service"] = FakeService()
    app.extensions["retriever"] = FakeRetriever()
    app.register_blueprint(api)
    response = app.test_client().post("/api/v1/verify", json={"claim": "test claim"})
    assert response.status_code == 200
    assert response.get_json()["verdict"] == "UNVERIFIABLE"


def test_health_endpoint_contract():
    app = Flask(__name__)
    app.extensions["verification_service"] = FakeService()
    app.extensions["retriever"] = FakeRetriever()
    app.register_blueprint(api)
    body = app.test_client().get("/api/v1/health").get_json()
    assert body["status"] == "healthy"
    assert body["searxng_configured"] is True and body["faiss_entries"] == 3
    assert "tavily_configured" not in body


from flask import Flask

from backend.api.verification import api


class FakeService:
    def verify_text(self, claim, original_language=None, top_k=5):
        return {"original_claim": claim, "verdict": "UNVERIFIABLE", "confidence": 0, "evidence": []}


class FakeRetriever:
    faiss_index = None
    class scraper:
        api_key = None


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
    assert app.test_client().get("/api/v1/health").get_json()["status"] == "healthy"

from unittest.mock import patch

from backend.services.verification_service import VerificationService


class FakeRetriever:
    def process_claim(self, claim, original_language, top_k=5):
        return {"sub_claims": [claim], "evidence": [{"source_domain": "example.org", "combined_reliability": 0.8}],
                "metadata": {"num_sources_kept": 1}}


@patch("backend.services.verification_service.run_graph")
def test_text_verification_composes_existing_pipeline(mock_graph):
    mock_graph.return_value = {"verdict": "TRUE", "final_confidence": 85, "justification": "Supported.",
                               "recommendation": "Likely accurate", "persona_insights": {}, "rounds_executed": 1}
    service = VerificationService(FakeRetriever(), 1024)
    result = service.verify_text("A verifiable claim.")
    assert result["verdict"] == "TRUE"
    assert result["evidence"][0]["source_domain"] == "example.org"
    mock_graph.assert_called_once()


def test_empty_claim_is_rejected():
    service = VerificationService(FakeRetriever(), 1024)
    try:
        service.verify_text("   ")
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("empty claims must be rejected")

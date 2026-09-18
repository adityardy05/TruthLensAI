from __future__ import annotations

from typing import Any

from backend.agents.team_b.graph import run_graph
from backend.preprocessing.image_ocr import detect_language, process_image_input, translate_to_english
from backend.rag.retriever import TeamARetrievalPipeline
from backend.services.url_extraction import extract_article_text


class VerificationService:
    """Single in-process pipeline for all web and Telegram inputs."""

    def __init__(self, retriever: TeamARetrievalPipeline, max_upload_bytes: int) -> None:
        self.retriever = retriever
        self.max_upload_bytes = max_upload_bytes

    def verify_text(self, claim: str, input_type: str = "text",original_language: str | None = None,top_k : int = 5) -> dict[str, Any]:
        original_claim = claim.strip()
        if not original_claim:
            raise ValueError("A non-empty claim is required.")
        language = (
             original_language
             if original_language
             else detect_language(original_claim)
         )
        normalized_claim = translate_to_english(original_claim, language).strip()
        retrieval = self.retriever.process_claim(normalized_claim, original_language=language,top_k = top_k)
        result = run_graph(
            claim=normalized_claim,
            evidence=retrieval["evidence"],
            original_claim=original_claim,
            original_language=language,
            sub_claims=retrieval["sub_claims"],
        )
        return self._response(input_type, original_claim, language, normalized_claim, retrieval, result)

    def verify_image(self, image_bytes: bytes) -> dict[str, Any]:
        if not image_bytes:
            raise ValueError("An image file is required.")
        if len(image_bytes) > self.max_upload_bytes:
            raise ValueError("Image exceeds the configured upload limit.")
        ocr = process_image_input(image_bytes)
        if not ocr["english_text"].strip():
            raise ValueError("OCR could not extract legible text from the image.")
        retrieval = self.retriever.process_claim(ocr["english_text"], original_language=ocr["detected_language"])
        result = run_graph(
            claim=ocr["english_text"], evidence=retrieval["evidence"],
            original_claim=ocr["original_text"], original_language=ocr["detected_language"],
            sub_claims=retrieval["sub_claims"],
        )
        response = self._response("image", ocr["original_text"], ocr["detected_language"], ocr["english_text"], retrieval, result)
        response["input_metadata"] = {"ocr_text": ocr["original_text"]}
        return response

    def verify_url(self, url: str) -> dict[str, Any]:
        text = extract_article_text(url)
        response = self.verify_text(text, input_type="url")
        response["input_metadata"] = {"url": url}
        return response

    @staticmethod
    def _response(input_type: str, original_claim: str, language: str, normalized_claim: str,
                  retrieval: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_type": input_type,
            "original_claim": original_claim,
            "detected_language": language,
            "normalized_claim": normalized_claim,
            "sub_claims": result.get("sub_claims", retrieval["sub_claims"]),
            "evidence": retrieval["evidence"],
            "retrieval_metadata": retrieval["metadata"],
            "verdict": result["verdict"],
            "confidence": result["final_confidence"],
            "justification": result["justification"],
            "recommendation": result["recommendation"],
            "persona_insights": result.get("persona_insights", {}),
            "stance_breakdown": result.get("stance_breakdown", {}),
            "coverage_score": result.get("coverage_score", 0),
            "rounds_executed": result.get("rounds_executed", 0),
        }

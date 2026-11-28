from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.agents.team_b.graph import run_graph
from backend.core.config import PROJECT_ROOT
from backend.preprocessing.image_ocr import detect_language, process_image_input, translate_to_english
from backend.rag.retriever import TeamARetrievalPipeline
from backend.services.url_extraction import extract_article_text
from backend.services.verdict_cache import VerdictCache
from backend.services.progress import report_progress


class VerificationService:
    """Single in-process pipeline for all web and Telegram inputs."""

    def __init__(self, retriever: TeamARetrievalPipeline, max_upload_bytes: int,
                 verdict_cache: VerdictCache | None = None) -> None:
        self.retriever = retriever
        self.max_upload_bytes = max_upload_bytes
        cache_path = Path(os.getenv(
            "VERDICT_CACHE_PATH", str(PROJECT_ROOT / "indexes" / "verdict_cache.sqlite3")))
        try:
            cache_ttl_days = int(os.getenv("VERDICT_CACHE_TTL_DAYS", "7"))
        except ValueError:
            cache_ttl_days = 7
        self.verdict_cache = verdict_cache or VerdictCache(cache_path, ttl_days=cache_ttl_days)

    def verify_text(self, claim: str, input_type: str = "text",original_language: str | None = None,top_k : int = 5, progress_callback=None) -> dict[str, Any]:
        original_claim = claim.strip()
        if not original_claim:
            raise ValueError("A non-empty claim is required.")
        language = (
             original_language
             if original_language
             else detect_language(original_claim)
         )
        normalized_claim = translate_to_english(original_claim, language).strip()
        return self._verify_normalized_claim(
            original_claim, language, normalized_claim, input_type, top_k=top_k, progress_callback=progress_callback)

    def _verify_normalized_claim(self, original_claim: str, language: str, normalized_claim: str,
                                 input_type: str, top_k: int = 5, progress_callback=None) -> dict[str, Any]:
        if progress_callback:
            progress_callback({"stage": "started", "message": "Verification request accepted.", "status": "running"})
        cached = self.verdict_cache.get(normalized_claim)
        if cached is not None:
            cached["original_claim"] = original_claim
            cached["detected_language"] = language
            cached["normalized_claim"] = normalized_claim
            cached["input_type"] = input_type
            cached.pop("input_metadata", None)
            cached.setdefault("retrieval_metadata", {})["verdict_cache_hit"] = True
            if progress_callback:
                progress_callback({"stage": "complete", "message": "Cached verification result loaded.", "status": "completed"})
            return cached

        if progress_callback:
            progress_callback({"stage": "retrieval", "message": "Searching sources and ranking evidence...", "status": "running"})
        retrieval = self.retriever.process_claim(
            normalized_claim, original_language=language, top_k=top_k, original_claim=original_claim)
        if progress_callback:
            progress_callback({"stage": "retrieval", "message": "Evidence retrieval completed.", "status": "completed"})
        result = run_graph(
            claim=normalized_claim,
            evidence=retrieval["evidence"],
            original_claim=original_claim,
            original_language=language,
            sub_claims=retrieval["sub_claims"],
            progress_callback=progress_callback,
        )
        result = self._require_verifiable_sources(retrieval, result)
        response = self._response(input_type, original_claim, language, normalized_claim, retrieval, result)
        response["retrieval_metadata"]["verdict_cache_hit"] = False
        if progress_callback:
            progress_callback({"stage": "complete", "message": "Verification completed: " + str(response["verdict"]) + ".", "status": "completed"})
        if response["verdict"] != "UNVERIFIABLE":
            self.verdict_cache.set(normalized_claim, response)
        return response

    def verify_image(self, image_bytes: bytes, progress_callback=None) -> dict[str, Any]:
        if not image_bytes:
            raise ValueError("An image file is required.")
        if len(image_bytes) > self.max_upload_bytes:
            raise ValueError("Image exceeds the configured upload limit.")
        if progress_callback:
            progress_callback({"stage": "ocr", "message": "Extracting text from uploaded image...", "status": "running"})
        ocr = process_image_input(image_bytes)
        if progress_callback:
            progress_callback({"stage": "ocr", "message": "Image text extraction completed.", "status": "completed"})
        if not ocr["english_text"].strip():
            raise ValueError("OCR could not extract legible text from the image.")
        response = self._verify_normalized_claim(
            ocr["original_text"], ocr["detected_language"], ocr["english_text"], "image", progress_callback=progress_callback)
        response["input_metadata"] = {"ocr_text": ocr["original_text"]}
        return response

    def verify_url(self, url: str, progress_callback=None) -> dict[str, Any]:
        text = extract_article_text(url)
        response = self.verify_text(text, input_type="url", progress_callback=progress_callback)
        response["input_metadata"] = {"url": url}
        return response

    @staticmethod
    def _require_verifiable_sources(retrieval: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Never issue a verdict when retrieval produced no evidence at all.

        Cached web evidence counts as evidence (the old rule only accepted fresh "web" items and
        wrongly turned every repeated claim into UNVERIFIABLE).
        """
        metadata = retrieval.get("metadata", {})
        if retrieval.get("evidence") or metadata.get("num_sources_kept", 1) > 0:
            return result

        return {
            **result,
            "verdict": "UNVERIFIABLE",
            "final_confidence": 0.0,
            "justification": (
                "No evidence could be retrieved for this claim (the search service returned nothing "
                "usable). A verdict would be unsupported."
            ),
            "recommendation": "Verify independently using credible, named sources.",
        }

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
            "patterns_detected": result.get("patterns_detected", []),
            "stance_breakdown": result.get("stance_breakdown", {}),
            "coverage_score": result.get("coverage_score", 0),
            "rounds_executed": result.get("rounds_executed", 0),
        }


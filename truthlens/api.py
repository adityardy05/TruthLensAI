"""HTTP API for running the TruthLens Team B verification graph."""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from team_b.graph import run_graph


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    original_claim: Optional[str] = None
    original_language: str = "english"
    sub_claims: List[str] = Field(default_factory=list)
    thread_id: Optional[str] = None


app = FastAPI(title="TruthLens API", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/verify")
def verify(request: VerifyRequest) -> Dict[str, Any]:
    try:
        return run_graph(
            claim=request.claim,
            evidence=request.evidence,
            original_claim=request.original_claim or request.claim,
            original_language=request.original_language,
            sub_claims=request.sub_claims,
            thread_id=request.thread_id,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
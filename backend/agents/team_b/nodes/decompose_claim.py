
"""
decompose_claim.py — Team B · Node 1
Splits the incoming claim into verifiable sub-claims.
Identifies claim type and generates search queries.

LLM calls: 1
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from backend.agents.llm_client import call_deepseek_json
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from backend.agents.llm_client import call_deepseek_json
from backend.agents.team_b.state import VerificationState


DECOMPOSE_PROMPT = """
You are a fact-checking assistant. Your job is to break down a claim into smaller, verifiable parts.

Claim: "{claim}"

Instructions:
1. Identify the claim type:
   - "factual"  → makes specific factual assertions that can be verified
   - "opinion"  → expresses a viewpoint or preference
   - "mixed"    → contains both facts and opinions

2. Break the claim into 2-4 specific sub-claims (individual assertions).

3. Generate 2-3 search queries that would find evidence to verify this claim.
   Make queries specific and focused on finding contradicting OR supporting evidence.

Return ONLY valid JSON, no explanation:
{{
    "claim_type": "factual | opinion | mixed",
    "sub_claims": [
        "sub-claim 1",
        "sub-claim 2"
    ],
    "search_queries": [
        "query 1",
        "query 2"
    ],
    "key_entities": ["entity1", "entity2"]
}}
"""


def decompose_claim_node(state: VerificationState) -> VerificationState:
    """
    Node 1: Decompose claim into sub-claims.

    Reads:  state["claim"]
    Writes: state["claim_type"], state["sub_claims"], state["search_queries"]
    """

    claim = state["claim"]

    try:
        prompt   = DECOMPOSE_PROMPT.format(claim=claim)
        response = call_deepseek_json(prompt)

        sub_claims     = response.get("sub_claims", [claim])
        claim_type     = response.get("claim_type", "factual")
        search_queries = response.get("search_queries", [claim])

        print(f"[decompose_claim] Type: {claim_type} | Sub-claims: {len(sub_claims)}")

        return {
            **state,
            "claim_type":     claim_type,
            "sub_claims":     sub_claims,
            "search_queries": search_queries,
            "total_llm_calls": state.get("total_llm_calls", 0) + 1,
        }

    except Exception as e:
        print(f"[decompose_claim] Failed: {e}. Using original claim as single sub-claim.")
        return {
            **state,
            "claim_type":     "factual",
            "sub_claims":     [claim],
            "search_queries": [claim],
            "total_llm_calls": state.get("total_llm_calls", 0) + 1,
        }

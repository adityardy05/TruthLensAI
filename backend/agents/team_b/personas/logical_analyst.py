"""
logical_analyst.py — Team B · Persona 2
Generates questions focused on logical consistency and reasoning validity.

Round 1:
    Uses generate_answer() → plain text

Round 3:
    Uses generate_structured_answer() → stance + confidence + reasoning
"""

from typing import List, Dict


QUESTION_PROMPT = """You are a Logical Analyst. Your goal is to detect logical fallacies and check if the reasoning in the claim is valid.

Claim being verified: "{claim}"

Retrieved Evidence:
{evidence_text}

Previous Q&A Memory:
{memory_text}

Your task: Ask ONE focused question that has NOT been covered in memory.
Focus on: causal claims (does X really cause Y?), logical fallacies, biological/physical plausibility, correlation vs causation, circular reasoning.

If memory is empty, ask about the most suspicious logical assertion in the claim.
If memory has entries, dig deeper into gaps — don't repeat.

Return ONLY the question, nothing else.
"""


ANSWER_PROMPT = """You are an evidence analyst answering a logical analyst's question.

Claim: "{claim}"

Question asked: "{question}"

Retrieved Evidence:
{evidence_text}

Previous context:
{memory_text}

Instructions:
- Assess the logical validity of the claim using evidence
- Identify any logical fallacies present (false causation, correlation ≠ causation, etc.)
- If evidence explains a mechanism or lack thereof, cite it
- If evidence doesn't cover this, rely on basic reasoning and say so
- Be concise (2-4 sentences)

Answer:
"""


# ── Round 3 structured answer prompt ──────────────────────────

STRUCTURED_ANSWER_PROMPT = """You are a Logical Analyst performing a third-round verification.

Claim:
"{claim}"

Question:
"{question}"

Retrieved Evidence:
{evidence_text}

Previous Q&A Context:
{memory_text}

Analyze the logical validity of the claim using the evidence provided.

Return ONLY valid JSON in exactly this format:

{{
    "stance": "SUPPORT | CONTRADICT | NEUTRAL | UNCERTAIN",
    "confidence": 0.0,
    "reasoning": "2-4 concise sentences explaining the logical analysis"
}}

Rules:
- SUPPORT = the evidence and reasoning support the claim
- CONTRADICT = the evidence or reasoning contradicts the claim
- NEUTRAL = the evidence is relevant but does not clearly support or contradict
- UNCERTAIN = the evidence is insufficient or conflicting
- confidence must be a number between 0.0 and 1.0
- Focus on logical consistency, causality, correlation vs causation, and reasoning validity
- Do not invent evidence or facts
- Do not use unsupported outside knowledge
"""


def generate_question(
    claim: str,
    evidence: List[Dict],
    memory: List[Dict],
    round_num: int
) -> str:
    """Generate a logical-analysis question."""

    try:
        from backend.agents.llm_client import call_deepseek
    except ModuleNotFoundError:
        from backend.agents.llm_client import call_deepseek

    from backend.agents.team_b.personas.fact_checker import (
        _format_evidence,
        _format_memory
    )

    prompt = QUESTION_PROMPT.format(
        claim=claim,
        evidence_text=_format_evidence(evidence),
        memory_text=(
            _format_memory(memory)
            or "No previous questions yet."
        )
    )

    return call_deepseek(prompt)


def generate_answer(
    claim: str,
    question: str,
    evidence: List[Dict],
    memory: List[Dict]
) -> str:
    """
    Existing Round 1 answer function.

    Kept unchanged in behavior so Round 1
    continues to work.
    """

    try:
        from backend.agents.llm_client import call_deepseek
    except ModuleNotFoundError:
        from backend.agents.llm_client import call_deepseek

    from backend.agents.team_b.personas.fact_checker import (
        _format_evidence,
        _format_memory
    )

    prompt = ANSWER_PROMPT.format(
        claim=claim,
        question=question,
        evidence_text=_format_evidence(evidence),
        memory_text=(
            _format_memory(memory)
            or "No previous context."
        )
    )

    return call_deepseek(prompt)


def generate_structured_answer(
    claim: str,
    question: str,
    evidence: List[Dict],
    memory: List[Dict]
) -> Dict:
    """
    Round 3 structured Logical Analyst answer.

    Returns:

        {
            "stance": "SUPPORT | CONTRADICT | NEUTRAL | UNCERTAIN",
            "confidence": 0.0 - 1.0,
            "reasoning": "..."
        }
    """

    try:
        from backend.agents.llm_client import call_deepseek_json
    except ModuleNotFoundError:
        from backend.agents.llm_client import call_deepseek_json

    from backend.agents.team_b.personas.fact_checker import (
        _format_evidence,
        _format_memory
    )

    prompt = STRUCTURED_ANSWER_PROMPT.format(
        claim=claim,
        question=question,
        evidence_text=_format_evidence(evidence),
        memory_text=(
            _format_memory(memory)
            or "No previous context."
        )
    )

    response = call_deepseek_json(prompt)

    # ── Validate stance ────────────────────────────────────────

    allowed_stances = {
        "SUPPORT",
        "CONTRADICT",
        "NEUTRAL",
        "UNCERTAIN"
    }

    stance = str(
        response.get(
            "stance",
            "UNCERTAIN"
        )
    ).upper().strip()

    if stance not in allowed_stances:
        stance = "UNCERTAIN"

    # ── Validate confidence ────────────────────────────────────

    try:
        confidence = float(
            response.get(
                "confidence",
                0.0
            )
        )
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    # ── Reasoning ──────────────────────────────────────────────

    reasoning = str(
        response.get(
            "reasoning",
            ""
        )
    ).strip()

    if not reasoning:
        reasoning = (
            "No structured reasoning was returned."
        )

    return {
        "stance": stance,
        "confidence": round(
            confidence,
            3
        ),
        "reasoning": reasoning,
    }


def extract_insight(
    question: str,
    answer: str
) -> str:
    """Extract a short insight from the answer."""

    first_sentence = answer.split(".")[0].strip()

    return (
        first_sentence[:120]
        if first_sentence
        else answer[:120]
    )
"""
fact_checker.py — Team B · Persona 1
Generates fact-checking questions focused on evidence alignment.

Round 1:
    Uses generate_answer() → plain text

Round 2:
    Uses generate_structured_answer() → stance + confidence + reasoning
"""

from typing import List, Dict


QUESTION_PROMPT = """You are a Fact Checker. Your goal is to verify factual accuracy against retrieved evidence.

Claim being verified: "{claim}"

Retrieved Evidence:
{evidence_text}

Previous Q&A Memory (what has already been discussed):
{memory_text}

Your task: Ask ONE focused question that has NOT been covered in the memory above.
Focus on: factual accuracy, whether evidence supports or contradicts the claim, specific numbers or dates.

If memory is empty, ask about the most important factual assertion in the claim.
If memory has entries, go deeper — don't repeat what's already been asked.

Return ONLY the question, nothing else.
"""


ANSWER_PROMPT = """You are an evidence analyst answering a fact-checker's question.

Claim: "{claim}"

Question asked: "{question}"

Retrieved Evidence:
{evidence_text}

Previous context:
{memory_text}

Instructions:
- Answer using ONLY the evidence provided above
- If evidence directly addresses this question, cite which source
- If evidence does NOT cover this, say "Evidence does not directly address this"
- Be concise (2-4 sentences max)
- Do NOT use outside knowledge or hallucinate facts

Answer:
"""


# ── Round 2 structured answer prompt ──────────────────────────

STRUCTURED_ANSWER_PROMPT = """You are a Fact Checker performing a second-round verification.

Claim:
"{claim}"

Question:
"{question}"

Retrieved Evidence:
{evidence_text}

Previous Q&A Context:
{memory_text}

Analyze the claim using ONLY the evidence provided.

Return ONLY valid JSON in exactly this format:

{{
    "stance": "SUPPORT | CONTRADICT | NEUTRAL | UNCERTAIN",
    "confidence": 0.0,
    "reasoning": "2-4 concise sentences explaining the evidence-based conclusion"
}}

Rules:
- SUPPORT = evidence supports the claim
- CONTRADICT = evidence contradicts the claim
- NEUTRAL = evidence is relevant but does not clearly support or contradict
- UNCERTAIN = evidence is insufficient or conflicting
- confidence must be a number between 0.0 and 1.0
- Do not use outside knowledge
- Do not invent sources or facts
"""


def generate_question(
    claim: str,
    evidence: List[Dict],
    memory: List[Dict],
    round_num: int
) -> str:
    """Generate a fact-checking question based on claim, evidence, and memory."""

    try:
        from shared.ollama_client import call_deepseek
    except ModuleNotFoundError:
        from truthlens.shared.ollama_client import call_deepseek

    evidence_text = _format_evidence(evidence)
    memory_text = _format_memory(memory)

    prompt = QUESTION_PROMPT.format(
        claim=claim,
        evidence_text=evidence_text,
        memory_text=(
            memory_text
            if memory_text
            else "No previous questions yet."
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

    IMPORTANT:
    This remains plain text so Round 1 continues working exactly
    as before.
    """

    try:
        from shared.ollama_client import call_deepseek
    except ModuleNotFoundError:
        from truthlens.shared.ollama_client import call_deepseek

    evidence_text = _format_evidence(evidence)
    memory_text = _format_memory(memory)

    prompt = ANSWER_PROMPT.format(
        claim=claim,
        question=question,
        evidence_text=evidence_text,
        memory_text=(
            memory_text
            if memory_text
            else "No previous context."
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
    Round 2 structured Fact Checker answer.

    Returns:

        {
            "stance": "SUPPORT | CONTRADICT | NEUTRAL | UNCERTAIN",
            "confidence": 0.0 - 1.0,
            "reasoning": "..."
        }

    This function is used by Round 2 only.
    """

    try:
        from shared.ollama_client import call_deepseek_json
    except ModuleNotFoundError:
        from truthlens.shared.ollama_client import call_deepseek_json

    evidence_text = _format_evidence(evidence)
    memory_text = _format_memory(memory)

    prompt = STRUCTURED_ANSWER_PROMPT.format(
        claim=claim,
        question=question,
        evidence_text=evidence_text,
        memory_text=(
            memory_text
            if memory_text
            else "No previous context."
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
        reasoning = "No structured reasoning was returned."

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
    """One-line summary of what was learned from this Q&A pair."""

    first_sentence = answer.split(".")[0].strip()

    return (
        first_sentence[:120]
        if first_sentence
        else answer[:120]
    )


# ── Formatting helpers ────────────────────────────────────────


def _format_evidence(
    evidence: List[Dict]
) -> str:

    if not evidence:
        return "No evidence available."

    lines = []

    for i, ev in enumerate(
        evidence[:5],
        1
    ):

        domain = ev.get(
            "source_domain",
            "unknown"
        )

        content = ev.get(
            "content",
            ""
        )[:300]

        r_score = ev.get(
            "combined_reliability",
            0.5
        )

        lines.append(
            f"[{i}] {domain} "
            f"(reliability={r_score:.2f}):\n"
            f"    {content}"
        )

    return "\n\n".join(lines)


def _format_memory(
    memory: List[Dict]
) -> str:

    if not memory:
        return ""

    lines = []

    for entry in memory:

        persona = entry.get(
            "persona",
            "Unknown"
        )

        q = entry.get(
            "question",
            ""
        )

        a = entry.get(
            "answer",
            ""
        )

        lines.append(
            f"[Round {entry.get('round', 1)} "
            f"· {persona}]\n"
            f"Q: {q}\n"
            f"A: {a}"
        )

    return "\n\n".join(lines)
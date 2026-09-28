"""
bias_detector.py — Team B · Persona 3
Generates questions focused on manipulation tactics, bias, and emotional framing.
"""

from typing import List, Dict


QUESTION_PROMPT = """You are a Bias Detector. Your goal is to identify manipulation tactics, emotional language, and misleading framing in the claim.

Claim being verified: "{claim}"

Retrieved Evidence:
{evidence_text}

Previous Q&A Memory:
{memory_text}

Your task: Ask ONE focused question that has NOT been covered in memory.
Focus on: emotional language (fear, anger, urgency), us-vs-them framing, vague adversaries ("they", "elites"),
appeal to conspiracy, sensationalism, selective use of facts, misleading framing.

If memory is empty, ask about the most suspicious framing or language in the claim.
If memory has entries, investigate deeper — don't repeat.

Return ONLY the question, nothing else.
"""

ANSWER_PROMPT = """You are an evidence analyst answering a bias detector's question.

Claim: "{claim}"

Question asked: "{question}"

Retrieved Evidence:
{evidence_text}

Previous context:
{memory_text}

Instructions:
- Assess if the claim uses manipulation tactics or biased framing
- Check if retrieved evidence shows this claim is part of a known misinformation pattern
- Identify specific tactics: appeal to fear, us-vs-them, conspiracy framing, etc.
- If evidence shows context about who is spreading this claim, cite it
- Be concise (2-4 sentences)

Answer:
"""

# Known manipulation tactics for heuristic pre-check
MANIPULATION_SIGNALS = {
    "fear":       ["danger", "deadly", "kill", "threat", "devastating", "terrifying"],
    "urgency":    ["urgent", "act now", "immediately", "breaking", "alert"],
    "conspiracy": ["they don't want", "hidden", "cover-up", "suppressed", "exposed"],
    "vague_enemy":["they", "elites", "government is hiding", "big pharma"],
    "caps_lock":  [],  # detected by checking uppercase ratio
}


def generate_question(
    claim:      str,
    evidence:   List[Dict],
    memory:     List[Dict],
    round_num:  int
) -> str:
    try:
        from backend.agents.llm_client import call_deepseek
    except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
        from backend.agents.llm_client import call_deepseek
    from backend.agents.team_b.personas.fact_checker import _format_evidence, _format_memory

    prompt = QUESTION_PROMPT.format(
        claim=claim,
        evidence_text=_format_evidence(evidence),
        memory_text=_format_memory(memory) or "No previous questions yet."
    )
    return call_deepseek(prompt)


def generate_answer(
    claim:      str,
    question:   str,
    evidence:   List[Dict],
    memory:     List[Dict]
) -> str:
    try:
        from backend.agents.llm_client import call_deepseek
    except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
        from backend.agents.llm_client import call_deepseek
    from backend.agents.team_b.personas.fact_checker import _format_evidence, _format_memory

    prompt = ANSWER_PROMPT.format(
        claim=claim,
        question=question,
        evidence_text=_format_evidence(evidence),
        memory_text=_format_memory(memory) or "No previous context."
    )
    return call_deepseek(prompt)



ROUND1_STRUCTURED_PROMPT = """You are performing Round 1 evidence-based verification.

Claim:
"{claim}"

Question:
"{question}"

Retrieved Evidence:
{evidence_text}

Previous Q&A Context:
{memory_text}

Return ONLY valid JSON:
{{
    "stance": "SUPPORT | CONTRADICT | NEUTRAL | UNCERTAIN",
    "confidence": 0.0,
    "reasoning": "2-4 concise sentences"
}}

Rules:
- SUPPORT = the evidence supports the claim.
- CONTRADICT = the evidence contradicts the claim.
- NEUTRAL = relevant evidence but no clear support/contradiction.
- UNCERTAIN = insufficient or conflicting evidence.
- confidence must be between 0.0 and 1.0.
- Use ONLY the supplied evidence; do not invent facts.
"""


def generate_round1_structured_answer(
    claim: str,
    question: str,
    evidence: List[Dict],
    memory: List[Dict]
) -> Dict:
    """Return explicit Round 1 stance, confidence and reasoning."""
    try:
        from backend.agents.llm_client import call_deepseek_json
    except ModuleNotFoundError:
        from backend.agents.llm_client import call_deepseek_json

    from backend.agents.team_b.personas.fact_checker import (
        _format_evidence,
        _format_memory
    )

    evidence_text = _format_evidence(evidence)
    memory_text = _format_memory(memory) or "No previous context."
    prompt = ROUND1_STRUCTURED_PROMPT.format(
        claim=claim,
        question=question,
        evidence_text=evidence_text,
        memory_text=memory_text
    )
    response = call_deepseek_json(prompt)

    allowed = {"SUPPORT", "CONTRADICT", "NEUTRAL", "UNCERTAIN"}
    stance = str(response.get("stance", "UNCERTAIN")).upper().strip()
    if stance not in allowed:
        stance = "UNCERTAIN"

    try:
        confidence = float(response.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(response.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = "No structured reasoning was returned."

    return {
        "stance": stance,
        "confidence": round(confidence, 3),
        "reasoning": reasoning
    }

def extract_insight(question: str, answer: str) -> str:
    first_sentence = answer.split(".")[0].strip()
    return first_sentence[:120] if first_sentence else answer[:120]


def quick_manipulation_score(claim: str) -> float:
    """
    Heuristic manipulation score without LLM.
    Used as additional signal in confidence_gate.
    Returns 0.0 (clean) to 1.0 (highly manipulative).
    """
    claim_lower = claim.lower()
    score = 0.0

    for tactic, keywords in MANIPULATION_SIGNALS.items():
        if tactic == "caps_lock":
            caps_ratio = sum(1 for c in claim if c.isupper()) / max(len(claim), 1)
            if caps_ratio > 0.15:
                score += 0.2
        else:
            for kw in keywords:
                if kw in claim_lower:
                    score += 0.1
                    break

    exclamation_count = claim.count("!")
    if exclamation_count >= 2:
        score += 0.1 * exclamation_count

    return min(1.0, round(score, 2))

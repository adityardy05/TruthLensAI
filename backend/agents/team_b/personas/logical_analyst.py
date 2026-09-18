"""
logical_analyst.py — Team B · Persona 2
Generates questions focused on logical consistency and reasoning validity.
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


def extract_insight(question: str, answer: str) -> str:
    first_sentence = answer.split(".")[0].strip()
    return first_sentence[:120] if first_sentence else answer[:120]

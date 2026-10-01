"""
round2_qa.py — Team B · Round 2

Fact Checker only.
Uses HIGH-QUALITY evidence:
    combined_reliability >= 0.70

Round 2 produces:
    stance
    confidence
    reasoning

LLM calls:
    2
    1 → generate question
    1 → structured answer
"""

import sys
import os
import json

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

from backend.agents.team_b.state import VerificationState
from backend.agents.team_b.personas import fact_checker, logical_analyst, bias_detector


ROUND_NUM = 2
HIGH_QUALITY_THRESHOLD = 0.70


def round2_qa_node(
    state: VerificationState
) -> VerificationState:
    """
    Round 2:
    Fact Checker analyzes only high-quality evidence.
    """

    claim = state["claim"]

    all_evidence = state.get(
        "evidence",
        []
    )

    qa_history = state.get(
        "qa_history",
        []
    ).copy()

    qa_memory = state.get(
        "qa_memory",
        []
    ).copy()

    llm_calls = state.get(
        "total_llm_calls",
        0
    )

    # Round 1 sets qa_round = 1
    # Therefore this node becomes Round 2.
    current_round = (
        state.get("qa_round", 1) + 1
    )

    # ── Select high-quality evidence ────────────────────────────

    high_quality = [
        ev
        for ev in all_evidence
        if ev.get(
            "combined_reliability",
            0.0
        ) >= HIGH_QUALITY_THRESHOLD
    ]

    print(
        f"[round2_qa] "
        f"High-quality evidence: "
        f"{len(high_quality)}/{len(all_evidence)}"
    )

    # ── No high-quality evidence ────────────────────────────────

    if not high_quality:

        print(
            "[round2_qa] "
            "No high-quality evidence. "
            "Skipping Fact Checker reasoning."
        )

        return {
            **state,

            "qa_round": current_round,

            "qa_history": qa_history,

            # Round 2 was visited but did NOT
            # produce reasoning.
            "rounds_executed": state.get(
                "rounds_executed",
                1
            ),
        }

    personas = (
        ("Fact Checker", fact_checker),
        ("Logical Analyst", logical_analyst),
        ("Bias Detector", bias_detector),
    )

    print("\n[ROUND 2 - CHALLENGE]")
    for persona_name, persona in personas:
        prior_memory = qa_memory.copy()
        try:
            question = persona.generate_question(
                claim=claim,
                evidence=high_quality,
                memory=prior_memory,
                round_num=ROUND_NUM,
            )
            llm_calls += 1
            structured_answer = getattr(persona, "generate_structured_answer", None)
            if structured_answer is None:
                structured_answer = persona.generate_round1_structured_answer
            result = structured_answer(
                claim=claim,
                question=question,
                evidence=high_quality,
                memory=prior_memory,
            )
            llm_calls += 1
            stance = result.get("stance", "UNCERTAIN")
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
        except Exception as error:
            print(f"[round2_qa] {persona_name} failed: {error}")
            stance = "UNCERTAIN"
            confidence = 0.0
            reasoning = f"Round 2 error: {error}"
            question = f"[FAILED: {persona_name}]"

        qa_memory.append({
            "round": ROUND_NUM,
            "persona": persona_name,
            "question": question,
            "answer": reasoning,
            "insight": persona.extract_insight(question, reasoning),
            "stance": stance,
            "confidence": confidence,
        })
        qa_history.append({
            "round": current_round,
            persona_name.casefold().replace(" ", "_"): {
                "stance": stance,
                "reasoning": reasoning,
                "confidence": confidence,
            },
            "confidence": confidence,
        })

    # ── Return updated state ────────────────────────────────────

    return {
        **state,

        "qa_round": current_round,

        "qa_memory": qa_memory,

        "qa_history": qa_history,

        "rounds_executed": 2,

        "total_llm_calls": llm_calls,
    }
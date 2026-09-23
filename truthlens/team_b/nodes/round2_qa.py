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

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

from team_b.state import VerificationState
from team_b.personas import fact_checker


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

    # Memory available before Round 2
    prior_memory = qa_memory.copy()

    try:

        # ── Step 1: Generate question ───────────────────────────

        print(
            "[round2_qa] "
            "Fact Checker generating question..."
        )

        question = fact_checker.generate_question(
            claim=claim,
            evidence=high_quality,
            memory=prior_memory,
            round_num=ROUND_NUM,
        )

        llm_calls += 1

        # ── Step 2: Generate structured answer ──────────────────

        print(
            "[round2_qa] "
            "Fact Checker generating "
            "structured answer..."
        )

        result = fact_checker.generate_structured_answer(
            claim=claim,
            question=question,
            evidence=high_quality,
            memory=prior_memory,
        )

        llm_calls += 1

        stance = result.get(
            "stance",
            "UNCERTAIN"
        )

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        reasoning = result.get(
            "reasoning",
            ""
        )

        # ── Step 3: Extract insight ─────────────────────────────

        insight = fact_checker.extract_insight(
            question,
            reasoning
        )

        # ── Step 4: Save to Q&A memory ─────────────────────────

        qa_memory.append({
            "round": ROUND_NUM,

            "persona": "Fact Checker",

            "question": question,

            "answer": reasoning,

            "insight": insight,

            "stance": stance,

            "confidence": confidence,
        })

        # ── Step 5: Save structured round history ──────────────

        qa_entry = {
            "round": current_round,

            "fact_checker": {
                "stance": stance,

                "reasoning": reasoning,

                "confidence": confidence,
            },

            "confidence": confidence,
        }

        qa_history.append(
            qa_entry
        )

        print(
            "[round2_qa] "
            f"Fact Checker stance={stance} "
            f"confidence={confidence:.3f}"
        )

        print(
            "[round2_qa] "
            "Round 2 complete."
        )

    except Exception as e:

        print(
            f"[round2_qa] "
            f"Fact Checker failed: {e}"
        )

        # Save failure information so the
        # final node knows Round 2 did not
        # produce a valid result.

        qa_entry = {
            "round": current_round,

            "fact_checker": {
                "stance": "UNCERTAIN",

                "reasoning":
                    f"Round 2 error: {e}",

                "confidence": 0.0,
            },

            "confidence": 0.0,
        }

        qa_history.append(
            qa_entry
        )

    # ── Return updated state ────────────────────────────────────

    return {
        **state,

        "qa_round": current_round,

        "qa_memory": qa_memory,

        "qa_history": qa_history,

        # Round 2 produced reasoning only if
        # the Fact Checker completed successfully.
        "rounds_executed": len([
            entry
            for entry in qa_history
            if (
                "fact_checker" in entry
                or "logical_analyst" in entry
                or "bias_detector" in entry
            )
        ]) + 1,

        "total_llm_calls": llm_calls,
    }
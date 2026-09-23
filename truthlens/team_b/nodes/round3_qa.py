"""
round3_qa.py — Team B · Round 3

Logical Analyst only.
Uses LOW-quality evidence:
    combined_reliability < 0.70

Round 3 produces:
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
from team_b.personas import logical_analyst


ROUND_NUM = 3
HIGH_QUALITY_THRESHOLD = 0.70


def round3_qa_node(
    state: VerificationState
) -> VerificationState:
    """
    Round 3:
    Logical Analyst analyzes only low-quality evidence.
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

    # Round 2 sets qa_round = 2
    # Therefore this node becomes Round 3.
    current_round = (
        state.get("qa_round", 2) + 1
    )

    # ── Select low-quality evidence ─────────────────────────────

    low_quality = [
        ev
        for ev in all_evidence
        if ev.get(
            "combined_reliability",
            1.0
        ) < HIGH_QUALITY_THRESHOLD
    ]

    print(
        f"[round3_qa] "
        f"Low-quality evidence: "
        f"{len(low_quality)}/{len(all_evidence)}"
    )

    # ── No low-quality evidence ─────────────────────────────────

    if not low_quality:

        print(
            "[round3_qa] "
            "No low-quality evidence. "
            "Skipping Logical Analyst reasoning."
        )

        return {
            **state,

            "qa_round": current_round,

            "qa_history": qa_history,

            "rounds_executed": state.get(
                "rounds_executed",
                1
            ),
        }

    # Memory available before Round 3
    prior_memory = qa_memory.copy()

    try:

        # ── Step 1: Generate question ───────────────────────────

        print(
            "[round3_qa] "
            "Logical Analyst generating question..."
        )

        question = logical_analyst.generate_question(
            claim=claim,
            evidence=low_quality,
            memory=prior_memory,
            round_num=ROUND_NUM,
        )

        llm_calls += 1

        # ── Step 2: Generate structured answer ──────────────────

        print(
            "[round3_qa] "
            "Logical Analyst generating "
            "structured answer..."
        )

        result = logical_analyst.generate_structured_answer(
            claim=claim,
            question=question,
            evidence=low_quality,
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

        insight = logical_analyst.extract_insight(
            question,
            reasoning
        )

        # ── Step 4: Save to Q&A memory ─────────────────────────

        qa_memory.append({
            "round": ROUND_NUM,

            "persona": "Logical Analyst",

            "question": question,

            "answer": reasoning,

            "insight": insight,

            "stance": stance,

            "confidence": confidence,
        })

        # ── Step 5: Save structured round history ──────────────

        qa_entry = {
            "round": current_round,

            "logical_analyst": {
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
            "[round3_qa] "
            f"Logical Analyst stance={stance} "
            f"confidence={confidence:.3f}"
        )

        print(
            "[round3_qa] "
            "Round 3 complete."
        )

    except Exception as e:

        print(
            f"[round3_qa] "
            f"Logical Analyst failed: {e}"
        )

        qa_entry = {
            "round": current_round,

            "logical_analyst": {
                "stance": "UNCERTAIN",

                "reasoning":
                    f"Round 3 error: {e}",

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
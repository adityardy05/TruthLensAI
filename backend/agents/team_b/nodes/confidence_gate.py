"""
confidence_gate.py — Team B · Node 4
Calculates confidence after Round 1.
Routes to Round 2 if confidence < 0.80, else goes to final_judgment.

LLM calls: 0  ← pure math, no LLM
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.agents.team_b.state import VerificationState

CONFIDENCE_THRESHOLD = 0.80   # >= this → skip Round 2

# Words in AAns answers that signal uncertainty
UNCERTAIN_PHRASES = [
    "unclear", "uncertain", "insufficient evidence",
    "no evidence", "cannot determine", "mixed",
    "conflicting", "does not directly address",
    "not covered", "unable to confirm", "ambiguous"
]


def confidence_gate_node(state: VerificationState) -> VerificationState:
    """
    Node 4: Calculate Round 1 confidence and decide routing.

    Reads:  state["qa_memory"], state["evidence"], state["avg_source_quality"]
    Writes: state["round1_confidence"], state["stance_consensus"],
            state["answer_certainty"], state["go_to_round2"]

    confidence = 0.50 × stance_consensus
               + 0.30 × avg_source_quality
               + 0.20 × answer_certainty
    """

    # Short-circuit: pipeline already ended
    if state.get("verdict"):
        return state

    memory           = state.get("qa_memory", [])
    evidence         = state.get("evidence", [])
    avg_quality      = state.get("avg_source_quality", 0.5)

    # ── Signal 1: Stance consensus ───────────────────────────────
    # Count stances from evidence (if Team A labeled them)
    # If no stance labels, derive from memory answers
    stances = [ev.get("stance", None) for ev in evidence]
    stances = [s for s in stances if s is not None]

    if stances:
        support    = stances.count("SUPPORT")
        contradict = stances.count("CONTRADICT")
        neutral    = stances.count("NEUTRAL")
        total      = len(stances)
        dominant   = max(support, contradict, neutral)
        stance_consensus = dominant / total if total > 0 else 0.5
    else:
        # No stance labels → derive from memory answers
        stance_consensus = _derive_consensus_from_memory(memory)

    # ── Signal 2: Avg source quality (from Team A R(d)) ──────────
    # Already computed in credibility_check
    # avg_quality already in state

    # ── Signal 3: Answer certainty from memory ───────────────────
    if memory:
        uncertain_count = sum(
            1 for entry in memory
            if any(phrase in entry.get("answer", "").lower()
                   for phrase in UNCERTAIN_PHRASES)
        )
        answer_certainty = 1.0 - (uncertain_count / len(memory))
    else:
        answer_certainty = 0.5  # no memory = uncertain by default

    # ── Combined confidence ───────────────────────────────────────
    confidence = (
        0.50 * stance_consensus +
        0.30 * avg_quality      +
        0.20 * answer_certainty
    )
    confidence = round(confidence, 3)

    go_to_round2 = confidence < CONFIDENCE_THRESHOLD

    print(
        f"[confidence_gate] "
        f"stance={stance_consensus:.2f} | "
        f"quality={avg_quality:.2f} | "
        f"certainty={answer_certainty:.2f} | "
        f"CONFIDENCE={confidence:.3f} | "
        f"round2={'YES' if go_to_round2 else 'NO'}"
    )

    return {
        **state,
        "round1_confidence": confidence,
        "stance_consensus":  round(stance_consensus, 3),
        "answer_certainty":  round(answer_certainty, 3),
        "go_to_round2":      go_to_round2,
        "rounds_executed":   1,   # will be updated to 2 if Round 2 runs
    }


def route_after_gate(state: VerificationState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node.

    Called by: graph.add_conditional_edges()
    """
    if state.get("verdict"):
        # Pipeline already short-circuited (no evidence)
        return "final_judgment"

    if state.get("go_to_round2", False):
        return "round2_qa"

    return "final_judgment"


# ── Helper ────────────────────────────────────────────────────

def _derive_consensus_from_memory(memory: list) -> float:
    """
    When evidence has no stance labels, estimate consensus
    from how definitive AAns answers were.
    """
    if not memory:
        return 0.5

    # Check if answers are mostly definitive
    definitive_phrases = [
        "clearly false", "confirmed false", "no evidence supports",
        "clearly true", "confirmed by", "multiple sources confirm",
        "directly contradicted", "strongly supported"
    ]

    definitive_count = sum(
        1 for entry in memory
        if any(phrase in entry.get("answer", "").lower()
               for phrase in definitive_phrases)
    )

    # More definitive answers = higher consensus
    ratio = definitive_count / len(memory)

    # Scale: 0 definitive = 0.4 consensus, all definitive = 0.9 consensus
    return 0.4 + (0.5 * ratio)

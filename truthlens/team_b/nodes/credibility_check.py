"""
credibility_check.py — Team B · Node 2
Reads Team A's R(d) scores directly. No re-scoring.
Computes aggregate quality metrics for use in confidence_gate.

LLM calls: 0  ← pure rule-based
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from team_b.state import VerificationState

LOW_CREDIBILITY_THRESHOLD  = 0.30   # below this → flagged
HIGH_CREDIBILITY_THRESHOLD = 0.70   # above this → high quality


def credibility_check_node(state: VerificationState) -> VerificationState:
    """
    Node 2: Aggregate Team A's R(d) credibility scores.

    Reads:  state["evidence"]  (Team A output with combined_reliability)
    Writes: state["avg_source_quality"],
            state["flagged_sources"],
            state["high_quality_count"]

    Short-circuits to UNVERIFIABLE if no evidence found.
    """

    evidence = state.get("evidence", [])

    # ── No evidence at all → short-circuit ──────────────────────
    if not evidence:
        print("[credibility_check] No evidence found. Short-circuiting to UNVERIFIABLE.")
        return {
            **state,
            "avg_source_quality": 0.0,
            "flagged_sources":    [],
            "high_quality_count": 0,
            "verdict":            "UNVERIFIABLE",
            "final_confidence":   0.0,
            "justification":      "No evidence could be retrieved to verify this claim.",
            "stance_breakdown":   {"SUPPORT": 0, "CONTRADICT": 0, "NEUTRAL": 0},
            "persona_insights":   {},
            "coverage_score":     0.0,
            "rounds_executed":    0,
            "recommendation":     "Verify independently — no evidence was found.",
            "pipeline_error":     "no_evidence_retrieved",
        }

    # ── Read R(d) scores from Team A ────────────────────────────
    reliability_scores = [
        ev.get("combined_reliability", 0.5)
        for ev in evidence
    ]

    avg_quality        = sum(reliability_scores) / len(reliability_scores)
    flagged_sources    = [
        ev.get("source_domain", "unknown")
        for ev in evidence
        if ev.get("combined_reliability", 0.5) < LOW_CREDIBILITY_THRESHOLD
    ]
    high_quality_count = sum(
        1 for s in reliability_scores
        if s >= HIGH_CREDIBILITY_THRESHOLD
    )

    print(
        f"[credibility_check] "
        f"avg_quality={avg_quality:.2f} | "
        f"high_quality={high_quality_count}/{len(evidence)} | "
        f"flagged={len(flagged_sources)}"
    )

    return {
        **state,
        "avg_source_quality": round(avg_quality, 3),
        "flagged_sources":    flagged_sources,
        "high_quality_count": high_quality_count,
    }
"""
state.py — TruthLens Team B
VerificationState: shared state passed between all LangGraph nodes.
Every node reads from this and writes back to it.
"""

from typing import TypedDict, List, Dict, Optional, Any


class QAMemoryEntry(TypedDict):
    round:      int
    persona:    str
    question:   str
    answer:     str
    insight:    str   # one-line summary of what was learned


class PersonaInsight(TypedDict):
    verdict:    str   # SUPPORT | CONTRADICT | NEUTRAL
    confidence: float
    summary:    str


class VerificationState(TypedDict):
    # ── Set by Module 0 (before Team B) ──────────────────────────
    claim:              str           # English-normalized claim
    original_claim:     str           # original text (any language)
    original_language:  str           # "hindi", "telugu", "english" etc.

    # ── Set by Team A (before Team B) ────────────────────────────
    evidence:           List[Dict]    # ranked evidence list with R(d) scores
    sub_claims:         List[str]     # decomposed sub-claims from Team A

    # ── Set by Node 1: decompose_claim ───────────────────────────
    claim_type:         str           # "factual" | "opinion" | "mixed"
    search_queries:     List[str]     # queries generated per sub-claim

    # ── Set by Node 2: credibility_check ─────────────────────────
    avg_source_quality: float         # mean R(d) across all evidence
    flagged_sources:    List[str]     # domains with R(d) < 0.30
    high_quality_count: int           # evidence pieces with R(d) >= 0.70

    # ── Set by Node 3: round1_qa ──────────────────────────────────
    qa_memory:          List[QAMemoryEntry]  # accumulates all Q&A pairs

    # ── Set by Node 4: confidence_gate ───────────────────────────
    round1_confidence:  float         # 0.0 - 1.0
    stance_consensus:   float         # dominant stance ratio
    answer_certainty:   float         # how certain were AAns answers
    go_to_round2:       bool          # True if confidence < 0.80

    # ── Set by Node 5: round2_qa (conditional) ───────────────────
    rounds_executed:    int           # 1 or 2

    # ── Set by Node 6: final_judgment ────────────────────────────
    verdict:            str           # TRUE | FALSE | PARTIALLY_TRUE | PARTIALLY_FALSE | UNVERIFIABLE
    final_confidence:   float         # 0.0 - 100.0
    justification:      str           # cites memory items — no hallucination
    stance_breakdown:   Dict          # {SUPPORT: int, CONTRADICT: int, NEUTRAL: int}
    persona_insights:   Dict          # {fact_checker: PersonaInsight, ...}
    coverage_score:     float         # angles covered / 3 total angles
    recommendation:     str           # "Do not share" | "Likely accurate" | "Verify independently"

    # ── Pipeline metadata ─────────────────────────────────────────
    total_llm_calls:    int
    pipeline_error:     Optional[str] # set if something failed gracefully
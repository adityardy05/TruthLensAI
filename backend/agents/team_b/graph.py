"""
graph.py — Team B
LangGraph definition. Wires all 6 nodes + conditional edge.
This is the only file that knows about all nodes.

Usage:
    from backend.agents.team_b.graph import build_graph
    app = build_graph()
    result = app.invoke(initial_state)
"""

try:
    from langgraph.graph import StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from backend.agents.team_b.state import VerificationState
from backend.agents.team_b.nodes.decompose_claim import decompose_claim_node
from backend.agents.team_b.nodes.credibility_check import credibility_check_node
from backend.agents.team_b.nodes.round1_qa import round1_qa_node
from backend.agents.team_b.nodes.confidence_gate import confidence_gate_node, route_after_gate
from backend.agents.team_b.nodes.round2_qa import round2_qa_node
from backend.agents.team_b.nodes.final_judgment import final_judgment_node

def build_graph(use_checkpointing: bool = True):
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph not installed. Run: pip install langgraph")

    graph = StateGraph(VerificationState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("decompose_claim",   decompose_claim_node)
    graph.add_node("credibility_check", credibility_check_node)
    graph.add_node("round1_qa",         round1_qa_node)
    graph.add_node("confidence_gate",   confidence_gate_node)
    graph.add_node("round2_qa",         round2_qa_node)
    graph.add_node("final_judgment",    final_judgment_node)

    # ── Fixed edges ───────────────────────────────────────────────
    graph.add_edge("decompose_claim",   "credibility_check")
    graph.add_edge("credibility_check", "round1_qa")
    graph.add_edge("round1_qa",         "confidence_gate")
    graph.add_edge("round2_qa",         "final_judgment")

    # ── Conditional edge — the adaptive routing ───────────────────
    graph.add_conditional_edges(
        "confidence_gate",
        route_after_gate,          # returns "round2_qa" or "final_judgment"
        {
            "round2_qa":      "round2_qa",
            "final_judgment": "final_judgment",
        }
    )

    # ── Entry and finish ──────────────────────────────────────────
    graph.set_entry_point("decompose_claim")
    graph.set_finish_point("final_judgment")

    # ── Compile with optional checkpointing ───────────────────────
    if use_checkpointing:
        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)
        print("[graph] Compiled with MemorySaver checkpointing")
    else:
        app = graph.compile()
        print("[graph] Compiled without checkpointing")
    return app


def run_graph(
    claim:             str,
    evidence:          list,
    original_claim:    str = "",
    original_language: str = "english",
    sub_claims:        list = None,
    thread_id:         str = None,
) -> dict:
    """
    Convenience wrapper — builds graph and runs it.

    Args:
        claim:             English-normalized claim (from Module 0)
        evidence:          Ranked evidence list with R(d) scores (from Team A)
        original_claim:    Original claim in source language
        original_language: Source language name
        sub_claims:        Pre-decomposed sub-claims (optional, Team A may provide)
        thread_id:         Unique ID for checkpointing (use claim hash)

    Returns:
        Final VerificationState dict with verdict, confidence, justification etc.
    """
    import hashlib

    app = build_graph(use_checkpointing=True)

    # Build initial state
    initial_state: VerificationState = {
        # From Module 0
        "claim":              claim,
        "original_claim":     original_claim or claim,
        "original_language":  original_language,

        # From Team A
        "evidence":           evidence,
        "sub_claims":         sub_claims or [],

        # Defaults — will be set by nodes
        "claim_type":         "",
        "search_queries":     [],
        "avg_source_quality": 0.5,
        "flagged_sources":    [],
        "high_quality_count": 0,
        "qa_memory":          [],
        "round1_confidence":  0.0,
        "stance_consensus":   0.0,
        "answer_certainty":   0.5,
        "go_to_round2":       False,
        "rounds_executed":    0,
        "verdict":            "",
        "final_confidence":   0.0,
        "justification":      "",
        "stance_breakdown":   {"SUPPORT": 0, "CONTRADICT": 0, "NEUTRAL": 0},
        "persona_insights":   {},
        "coverage_score":     0.0,
        "recommendation":     "",
        "total_llm_calls":    0,
        "pipeline_error":     None,
    }

    # Use claim hash as thread_id for checkpointing
    if thread_id is None:
        thread_id = hashlib.md5(claim.encode()).hexdigest()

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"[TruthLens] Running pipeline for: {claim[:80]}")
    print(f"{'='*60}")

    final_state = app.invoke(initial_state, config=config)

    print(f"\n[TruthLens] VERDICT: {final_state['verdict']} ({final_state['final_confidence']:.0f}%)")
    print(f"[TruthLens] Rounds: {final_state['rounds_executed']} | LLM calls: {final_state['total_llm_calls']}")

    return final_state

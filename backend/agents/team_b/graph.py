"""
graph.py — Team B
LangGraph definition for the 3-round evidence-stratified QA pipeline.

Flow:
    decompose_claim
        ↓
    credibility_check
        ↓
    round1_qa
        ↓
    confidence_gate
        ├── confidence >= 0.85 → final_judgment
        │
        └── confidence < 0.85 → round2_qa
                                      ↓
                                  round3_qa
                                      ↓
                                final_judgment
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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
from backend.agents.team_b.nodes.confidence_gate import (
    confidence_gate_node,
    route_after_gate,
)
from backend.agents.team_b.nodes.round2_qa import round2_qa_node
from backend.agents.team_b.nodes.round3_qa import round3_qa_node
from backend.agents.team_b.nodes.final_judgment import final_judgment_node


CHECKPOINT_DB = os.getenv(
    "CHECKPOINT_DB",
    "checkpoints/truthlens_checkpoints.db"
)


def build_graph(use_checkpointing: bool = True):
    """
    Build and compile the TruthLens LangGraph.
    """

    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph not installed. Run: pip install langgraph"
        )

    graph = StateGraph(VerificationState)

    # ── Register nodes ─────────────────────────────────────────

    graph.add_node(
        "decompose_claim",
        decompose_claim_node
    )

    graph.add_node(
        "credibility_check",
        credibility_check_node
    )

    graph.add_node(
        "round1_qa",
        round1_qa_node
    )

    graph.add_node(
        "confidence_gate",
        confidence_gate_node
    )

    graph.add_node(
        "round2_qa",
        round2_qa_node
    )

    graph.add_node(
        "round3_qa",
        round3_qa_node
    )

    graph.add_node(
        "final_judgment",
        final_judgment_node
    )

    # ── Fixed edges ────────────────────────────────────────────

    graph.add_edge(
        "decompose_claim",
        "credibility_check"
    )

    graph.add_edge(
        "credibility_check",
        "round1_qa"
    )

    graph.add_edge(
        "round1_qa",
        "confidence_gate"
    )

    # Round 2 → Round 3
    graph.add_edge(
        "round2_qa",
        "round3_qa"
    )

    # Round 3 → Final Judgment
    graph.add_edge(
        "round3_qa",
        "final_judgment"
    )

    # ── Conditional routing after Round 1 ──────────────────────

    graph.add_conditional_edges(
        "confidence_gate",
        route_after_gate,
        {
            "round2_qa": "round2_qa",
            "final_judgment": "final_judgment",
        }
    )

    # ── Entry and finish ──────────────────────────────────────

    graph.set_entry_point(
        "decompose_claim"
    )

    graph.set_finish_point(
        "final_judgment"
    )

    # ── Compile ────────────────────────────────────────────────

    if use_checkpointing:

        checkpointer = MemorySaver()

        app = graph.compile(
            checkpointer=checkpointer
        )

        print(
            "[graph] Compiled with MemorySaver checkpointing"
        )

    else:

        app = graph.compile()

        print(
            "[graph] Compiled without checkpointing"
        )

    return app


def run_graph(
    claim: str,
    evidence: list,
    original_claim: str = "",
    original_language: str = "english",
    sub_claims: list = None,
    thread_id: str = None,
) -> dict:
    """
    Convenience wrapper — builds and runs the TruthLens graph.
    """

    import hashlib

    app = build_graph(
        use_checkpointing=True
    )

    # ── Initial state ──────────────────────────────────────────

    initial_state: VerificationState = {

        # Module 0
        "claim": claim,

        "original_claim": (
            original_claim
            or claim
        ),

        "original_language": (
            original_language
        ),

        # Team A
        "evidence": evidence,

        "sub_claims": (
            sub_claims
            or []
        ),

        # Node 1
        "claim_type": "",

        "search_queries": [],

        # Node 2
        "avg_source_quality": 0.5,

        "flagged_sources": [],

        "high_quality_count": 0,

        # Node 3
        "qa_memory": [],

        # New 3-round architecture
        "qa_round": 0,

        "rounds_executed": 0,

        "qa_history": [],

        # Node 4
        "round1_confidence": 0.0,

        "stance_consensus": 0.0,

        "answer_certainty": 0.5,

        "go_to_round2": False,

        # Node 6
        "verdict": "",

        "final_confidence": 0.0,

        "justification": "",

        "stance_breakdown": {
            "SUPPORT": 0,
            "CONTRADICT": 0,
            "NEUTRAL": 0,
        },

        "persona_insights": {},

        "patterns_detected": [],

        "coverage_score": 0.0,

        "recommendation": "",

        # Pipeline metadata
        "total_llm_calls": 0,

        "pipeline_error": None,
    }

    # ── Thread ID ──────────────────────────────────────────────

    if thread_id is None:

        thread_id = hashlib.md5(
            claim.encode()
        ).hexdigest()

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"[TruthLens] Running pipeline for: "
        f"{claim[:80]}"
    )

    print(
        f"{'=' * 60}"
    )

    # ── Execute graph ──────────────────────────────────────────

    final_state = app.invoke(
        initial_state,
        config=config
    )

    qa_memory = final_state.get("qa_memory", [])
    weighted_support = sum(
        float(item.get("combined_reliability", 0.0) or 0.0)
        for item in final_state.get("evidence", [])
        if str(item.get("stance", "")).upper() == "SUPPORT"
    )
    weighted_refute = sum(
        float(item.get("combined_reliability", 0.0) or 0.0)
        for item in final_state.get("evidence", [])
        if str(item.get("stance", "")).upper() in {"CONTRADICT", "REFUTE"}
    )

    resolved_stances = {}
    for persona in ("Fact Checker", "Logical Analyst", "Bias Detector"):
        persona_entries = [
            entry for entry in qa_memory
            if entry.get("persona") == persona
        ]
        resolved_stances[persona] = (
            persona_entries[-1].get("stance", "UNCERTAIN")
            if persona_entries else "UNCERTAIN"
        )

    print("\n" + "#" * 64)
    print("[FINAL JUDGEMENT]")
    print("#" * 64)
    print(f"\nClaim:\n{claim}")
    print(f"\nEvidence Count:\n{len(final_state.get('evidence', []))}")
    print(f"\nR1 QA Count:\n{sum(1 for item in qa_memory if item.get('round') == 1)}")
    print(f"\nR2 QA Count:\n{sum(1 for item in qa_memory if item.get('round') == 2)}")
    print(f"\nR3 QA Count:\n{sum(1 for item in qa_memory if item.get('round') == 3)}")
    print("\nAgent Stances:")
    for persona in ("Fact Checker", "Logical Analyst", "Bias Detector"):
        print(f"- {persona}: {resolved_stances[persona]}")
    print(f"\nFINAL LABEL:\n{final_state.get('verdict', 'UNVERIFIABLE')}")
    print(f"\nCONFIDENCE:\n{float(final_state.get('final_confidence', 0.0)):.2f}")
    print("\n" + "#" * 64)
    print("[TRUTHLENS PIPELINE COMPLETE]")
    print("#" * 64)

    print(
        f"\n[TruthLens] VERDICT: "
        f"{final_state['verdict']} "
        f"({final_state['final_confidence']:.0f}%)"
    )

    print(
        f"[TruthLens] Rounds: "
        f"{final_state['rounds_executed']} "
        f"| LLM calls: "
        f"{final_state['total_llm_calls']}"
    )

    return final_state
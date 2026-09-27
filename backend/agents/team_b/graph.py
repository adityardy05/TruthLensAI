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
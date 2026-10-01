"""
round1_qa.py — Team B · Node 3
Runs all 3 personas × (question + answer) with shared memory.
Memory M accumulates after each Q&A pair.
Same DeepSeek instance handles both persona Q and AAns A.

LLM calls: 6  (3 questions + 3 structured answers)
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.agents.team_b.state import VerificationState
from backend.agents.team_b.personas import fact_checker, logical_analyst, bias_detector


ROUND_NUM = 1

# Persona registry — order matters for memory building
PERSONAS = [
    {
        "name":   "Fact Checker",
        "module": fact_checker,
        "goal":   "verify factual accuracy against evidence",
    },
    {
        "name":   "Logical Analyst",
        "module": logical_analyst,
        "goal":   "detect logical fallacies and reasoning validity",
    },
    {
        "name":   "Bias Detector",
        "module": bias_detector,
        "goal":   "identify manipulation tactics and emotional framing",
    },
]


def round1_qa_node(state: VerificationState) -> VerificationState:
    """
    Node 3: Run Round 1 Q&A loop across all 3 personas.

    Reads:  state["claim"], state["evidence"]
    Writes: state["qa_memory"], state["total_llm_calls"]

    Memory grows after EACH Q&A pair so each persona sees
    what the previous persona already asked and answered.
    """

    # Short-circuit: if pipeline already ended in credibility_check
    if state.get("verdict"):
        return state

    claim    = state["claim"]
    evidence = state.get("evidence", [])
    memory   = []     # M₀ = empty
    llm_calls = state.get("total_llm_calls", 0)

    print(f"[round1_qa] Starting Round 1 — {len(PERSONAS)} personas")
    print("\n" + "#" * 64)
    print("[TRUTHLENS MULTI-AGENT DELIBERATION]")
    print("#" * 64)
    print("\n[ROUND 1 - ANALYZE]")

    for persona in PERSONAS:
        name   = persona["name"]
        module = persona["module"]

        try:
            print("\n[LANGGRAPH STATE]")
            print(f"Round: {ROUND_NUM}")
            print(f"Agent: {name}")
            print(f"Previous QA Count: {len(memory)}")
            print("Current Memory/State Available: YES")

            # ── Step 1: Persona generates question ──────────────
            print(f"[round1_qa] {name} generating question...")
            question = module.generate_question(
                claim=claim,
                evidence=evidence,
                memory=memory,
                round_num=ROUND_NUM
            )
            llm_calls += 1

            # ── Step 2: Generate structured Round 1 stance ─────
            print(f"[round1_qa] {name} generating structured stance...")
            structured = module.generate_round1_structured_answer(
                claim=claim,
                question=question,
                evidence=evidence,
                memory=memory
            )
            llm_calls += 1

            stance = structured["stance"]
            persona_confidence = structured["confidence"]
            reasoning = structured["reasoning"]
            answer = reasoning

            print("\n" + "-" * 60)
            print(f"AGENT: {name.upper()}")
            print("-" * 60)
            print("\nQUESTION:\n" + str(question))
            print("\nANSWER:\n" + str(answer))
            print("\nAGENT OUTPUT:\n" + json.dumps(structured, indent=2, default=str))
            print(f"\nSTANCE:\n{stance}")

            # ── Step 3: Extract one-line insight ────────────────
            insight = module.extract_insight(question, answer)

            # ── Step 4: Update memory ────────────────────────────
            # Next persona sees this Q&A before asking its question
            memory.append({
                "round":    ROUND_NUM,
                "persona":  name,
                "question": question,
                "answer":      answer,
                "stance":      stance,
                "confidence":  persona_confidence,
                "reasoning":   reasoning,
                "insight":     insight,
            })

            print(f"[round1_qa] {name} done. Insight: {insight[:60]}...")

        except Exception as e:
            print(f"[round1_qa] {name} failed: {e}. Skipping.")
            print(f"\nAGENT OUTPUT:\n[FAILED: {name}] {e}")
            memory.append({
                "round":    ROUND_NUM,
                "persona":  name,
                "question": f"[FAILED: {name}]",
                "answer":   f"Error occurred: {e}",
                "insight":  "persona_failed",
            })

    print(f"[round1_qa] Round 1 complete. Memory entries: {len(memory)}")

    return {
        **state,
        "qa_memory":       memory,
        "total_llm_calls": llm_calls,
    }

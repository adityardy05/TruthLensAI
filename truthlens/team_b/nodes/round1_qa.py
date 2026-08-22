"""
round1_qa.py — Team B · Node 3
Runs all 3 personas × (question + answer) with shared memory.
Memory M accumulates after each Q&A pair.
Same DeepSeek instance handles both persona Q and AAns A.

LLM calls: 6  (3 questions + 3 answers)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from team_b.state import VerificationState
from team_b.personas import fact_checker, logical_analyst, bias_detector


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

    for persona in PERSONAS:
        name   = persona["name"]
        module = persona["module"]

        try:
            # ── Step 1: Persona generates question ──────────────
            print(f"[round1_qa] {name} generating question...")
            question = module.generate_question(
                claim=claim,
                evidence=evidence,
                memory=memory,
                round_num=ROUND_NUM
            )
            llm_calls += 1

            # ── Step 2: Same DeepSeek answers as AAns ───────────
            print(f"[round1_qa] AAns answering for {name}...")
            answer = module.generate_answer(
                claim=claim,
                question=question,
                evidence=evidence,
                memory=memory
            )
            llm_calls += 1

            # ── Step 3: Extract one-line insight ────────────────
            insight = module.extract_insight(question, answer)

            # ── Step 4: Update memory ────────────────────────────
            # Next persona sees this Q&A before asking its question
            memory.append({
                "round":    ROUND_NUM,
                "persona":  name,
                "question": question,
                "answer":   answer,
                "insight":  insight,
            })

            print(f"[round1_qa] {name} done. Insight: {insight[:60]}...")

        except Exception as e:
            print(f"[round1_qa] {name} failed: {e}. Skipping.")
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
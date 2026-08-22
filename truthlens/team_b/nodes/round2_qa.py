"""
round2_qa.py — Team B · Node 5
Conditional node — only runs when Round 1 confidence < 0.80.
Personas read Round 1 memory and ask DEEPER follow-up questions.
Memory prevents any question already asked in Round 1 from repeating.

LLM calls: 6  (3 questions + 3 answers)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from team_b.state import VerificationState
from team_b.personas import fact_checker, logical_analyst, bias_detector

ROUND_NUM = 2

PERSONAS = [
    {"name": "Fact Checker",     "module": fact_checker},
    {"name": "Logical Analyst",  "module": logical_analyst},
    {"name": "Bias Detector",    "module": bias_detector},
]


def round2_qa_node(state: VerificationState) -> VerificationState:
    """
    Node 5: Round 2 Q&A — deeper questioning using Round 1 memory.

    Reads:  state["qa_memory"]  (Round 1 entries already in here)
    Writes: state["qa_memory"]  (appends Round 2 entries)
            state["rounds_executed"] = 2

    The Round 1 memory passed to each persona prevents:
    - Repeating questions already asked
    - Generating redundant insights
    """

    claim    = state["claim"]
    evidence = state.get("evidence", [])
    memory   = state.get("qa_memory", []).copy()   # includes Round 1
    llm_calls = state.get("total_llm_calls", 0)

    print(
        f"[round2_qa] Starting Round 2 "
        f"(Round 1 confidence was {state.get('round1_confidence', '?'):.3f})"
    )

    for persona in PERSONAS:
        name   = persona["name"]
        module = persona["module"]

        try:
            # Persona sees ALL of Round 1 memory before asking
            # → will naturally ask something not covered in Round 1
            print(f"[round2_qa] {name} generating deeper question...")
            question = module.generate_question(
                claim=claim,
                evidence=evidence,
                memory=memory,         # ← includes Round 1
                round_num=ROUND_NUM
            )
            llm_calls += 1

            print(f"[round2_qa] AAns answering for {name}...")
            answer = module.generate_answer(
                claim=claim,
                question=question,
                evidence=evidence,
                memory=memory          # ← context-aware answer
            )
            llm_calls += 1

            insight = module.extract_insight(question, answer)

            # Append Round 2 entry to same memory list
            memory.append({
                "round":    ROUND_NUM,
                "persona":  name,
                "question": question,
                "answer":   answer,
                "insight":  insight,
            })

            print(f"[round2_qa] {name} done. Insight: {insight[:60]}...")

        except Exception as e:
            print(f"[round2_qa] {name} failed: {e}. Skipping.")
            memory.append({
                "round":    ROUND_NUM,
                "persona":  name,
                "question": f"[FAILED: {name}]",
                "answer":   f"Error: {e}",
                "insight":  "persona_failed",
            })

    print(f"[round2_qa] Round 2 complete. Total memory entries: {len(memory)}")

    return {
        **state,
        "qa_memory":       memory,
        "rounds_executed": 2,
        "total_llm_calls": llm_calls,
    }
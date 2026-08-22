"""
final_judgment.py — Team B · Node 6
Supervisor logic + verdict generation.
Reads full memory → aggregates signals → generates verdict via DeepSeek.
Justification must cite memory items — no hallucination.

LLM calls: 1
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from team_b.state import VerificationState
try:
    from shared.ollama_client import call_deepseek_json
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from truthlens.shared.ollama_client import call_deepseek_json

JUDGMENT_PROMPT = """You are a senior fact-checking supervisor making the final verdict on a claim.

Claim: "{claim}"

Retrieved Evidence Summary:
{evidence_summary}

Complete Q&A Analysis Memory:
{memory_text}

Coverage Assessment:
- Factual accuracy covered: {factual_covered}
- Logical validity covered: {logical_covered}
- Bias/manipulation covered: {bias_covered}

Evidence Stance Distribution:
- Supporting: {support_count} sources
- Contradicting: {contradict_count} sources
- Neutral: {neutral_count} sources

Instructions:
1. Base verdict ONLY on the memory and evidence above
2. Do NOT use outside knowledge
3. Cite specific memory insights in your justification
4. Choose verdict:
   - TRUE: strong credible support, no credible contradiction
   - FALSE: strong credible contradiction, little/no support
   - PARTIALLY_TRUE: mixed evidence or claim is partially correct
   - PARTIALLY_FALSE: claim is mostly wrong but has some truth
   - UNVERIFIABLE: insufficient or too conflicting evidence

Return ONLY valid JSON:
{{
    "verdict": "TRUE | FALSE | PARTIALLY_TRUE | PARTIALLY_FALSE | UNVERIFIABLE",
    "final_confidence": <0-100 as integer>,
    "justification": "<2-4 sentences citing specific memory insights and evidence>",
    "fact_checker_summary": "<one line from Fact Checker insights>",
    "logical_analyst_summary": "<one line from Logical Analyst insights>",
    "bias_detector_summary": "<one line from Bias Detector insights>",
    "recommendation": "Do not share | Likely accurate | Verify independently | Partially accurate"
}}
"""

# Expected angles — used for coverage scoring
EXPECTED_ANGLES = ["Fact Checker", "Logical Analyst", "Bias Detector"]


def final_judgment_node(state: VerificationState) -> VerificationState:
    """
    Node 6: Supervisor check + final verdict generation.

    Reads:  state["qa_memory"], state["evidence"], state["claim"]
    Writes: state["verdict"], state["final_confidence"],
            state["justification"], state["stance_breakdown"],
            state["persona_insights"], state["coverage_score"],
            state["recommendation"]
    """

    # Short-circuit: pipeline already ended (no evidence case)
    if state.get("verdict"):
        return state

    claim    = state["claim"]
    memory   = state.get("qa_memory", [])
    evidence = state.get("evidence", [])
    llm_calls = state.get("total_llm_calls", 0)

    # ── Supervisor check: coverage ────────────────────────────────
    personas_in_memory = {entry["persona"] for entry in memory}
    coverage_score     = len(
        [a for a in EXPECTED_ANGLES if a in personas_in_memory]
    ) / len(EXPECTED_ANGLES)

    print(f"[final_judgment] Coverage: {coverage_score:.0%} | Rounds: {state.get('rounds_executed', 1)}")

    # ── Stance breakdown from evidence ───────────────────────────
    stances = [ev.get("stance", "NEUTRAL") for ev in evidence]
    stance_breakdown = {
        "SUPPORT":    stances.count("SUPPORT"),
        "CONTRADICT": stances.count("CONTRADICT"),
        "NEUTRAL":    stances.count("NEUTRAL"),
    }

    # ── Evidence summary for prompt ──────────────────────────────
    evidence_summary = _build_evidence_summary(evidence)
    memory_text      = _build_memory_text(memory)

    # ── Build prompt ─────────────────────────────────────────────
    prompt = JUDGMENT_PROMPT.format(
        claim=claim,
        evidence_summary=evidence_summary,
        memory_text=memory_text,
        factual_covered="Yes" if "Fact Checker"    in personas_in_memory else "No",
        logical_covered="Yes" if "Logical Analyst" in personas_in_memory else "No",
        bias_covered=   "Yes" if "Bias Detector"   in personas_in_memory else "No",
        support_count=    stance_breakdown["SUPPORT"],
        contradict_count= stance_breakdown["CONTRADICT"],
        neutral_count=    stance_breakdown["NEUTRAL"],
    )

    # ── DeepSeek generates final verdict ─────────────────────────
    try:
        response  = call_deepseek_json(prompt)
        llm_calls += 1

        verdict     = response.get("verdict",          "UNVERIFIABLE")
        confidence  = float(response.get("final_confidence", 50.0))
        justification = response.get("justification",  "Unable to generate justification.")
        recommendation = response.get("recommendation","Verify independently")

        persona_insights = {
            "fact_checker":    response.get("fact_checker_summary",    ""),
            "logical_analyst": response.get("logical_analyst_summary", ""),
            "bias_detector":   response.get("bias_detector_summary",   ""),
        }

        print(f"[final_judgment] Verdict: {verdict} ({confidence:.0f}%)")

    except Exception as e:
        print(f"[final_judgment] DeepSeek failed: {e}. Using fallback verdict.")

        # Fallback: rule-based verdict from stance counts
        verdict, confidence, justification, recommendation = \
            _fallback_verdict(stance_breakdown, state.get("avg_source_quality", 0.5))
        persona_insights = {}
        llm_calls += 1

    return {
        **state,
        "verdict":          verdict,
        "final_confidence": confidence,
        "justification":    justification,
        "stance_breakdown": stance_breakdown,
        "persona_insights": persona_insights,
        "coverage_score":   round(coverage_score, 2),
        "recommendation":   recommendation,
        "total_llm_calls":  llm_calls,
        "pipeline_error":   None,
    }


# ── Helpers ───────────────────────────────────────────────────

def _build_evidence_summary(evidence: list) -> str:
    if not evidence:
        return "No evidence retrieved."

    lines = []
    for ev in evidence[:5]:
        domain  = ev.get("source_domain", "unknown")
        r_score = ev.get("combined_reliability", 0.5)
        stance  = ev.get("stance", "NEUTRAL")
        content = ev.get("content", "")[:150]
        lines.append(f"- {domain} (R={r_score:.2f}, stance={stance}): {content}")

    return "\n".join(lines)


def _build_memory_text(memory: list) -> str:
    if not memory:
        return "No Q&A memory available."

    lines = []
    for entry in memory:
        lines.append(
            f"[Round {entry['round']} · {entry['persona']}]\n"
            f"Q: {entry['question']}\n"
            f"A: {entry['answer']}\n"
            f"Insight: {entry['insight']}"
        )
    return "\n\n".join(lines)


def _fallback_verdict(
    stance_breakdown: dict,
    avg_quality: float
) -> tuple:
    """
    Rule-based fallback if DeepSeek fails.
    Returns (verdict, confidence, justification, recommendation).
    """
    support    = stance_breakdown.get("SUPPORT", 0)
    contradict = stance_breakdown.get("CONTRADICT", 0)
    total      = support + contradict + stance_breakdown.get("NEUTRAL", 0)

    if total == 0:
        return (
            "UNVERIFIABLE", 20.0,
            "No evidence was available to verify this claim.",
            "Verify independently"
        )

    contradict_ratio = contradict / total
    support_ratio    = support    / total

    if contradict_ratio >= 0.7:
        return (
            "FALSE",
            round(60 + contradict_ratio * 35, 1),
            f"{contradict}/{total} sources contradict the claim.",
            "Do not share"
        )
    elif support_ratio >= 0.7:
        return (
            "TRUE",
            round(60 + support_ratio * 35, 1),
            f"{support}/{total} sources support the claim.",
            "Likely accurate"
        )
    else:
        return (
            "PARTIALLY_TRUE",
            50.0,
            "Evidence is mixed — some sources support, others contradict.",
            "Verify independently"
        )
"""
final_judgment.py — Team B · Final Supervisor

Uses the latest available persona result:

Fact Checker:
    Round 2 if available, otherwise Round 1

Logical Analyst:
    Round 3 if available, otherwise Round 1

Bias Detector:
    Round 1 only

LLM calls: 1
"""

import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..'
    )
)

from team_b.state import VerificationState

try:
    from shared.ollama_client import call_deepseek_json
except ModuleNotFoundError:
    from truthlens.shared.ollama_client import call_deepseek_json


JUDGMENT_PROMPT = """You are a senior fact-checking supervisor making the final verdict on a claim.

Claim:
"{claim}"

Retrieved Evidence Summary:
{evidence_summary}

Latest Persona Analysis:
{persona_analysis}

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

Rounds executed:
{rounds_executed}

Instructions:
1. Base the verdict ONLY on the evidence and persona analysis provided.
2. Do NOT use outside knowledge.
3. Give priority to the latest available result for each persona.
4. Fact Checker: use Round 2 if available, otherwise Round 1.
5. Logical Analyst: use Round 3 if available, otherwise Round 1.
6. Bias Detector: use Round 1.
7. Cite specific persona insights or evidence in the justification.
8. Do not invent evidence, sources, or persona findings.
9. Choose exactly one verdict:
   - TRUE
   - FALSE
   - PARTIALLY_TRUE
   - PARTIALLY_FALSE
   - UNVERIFIABLE

Return ONLY valid JSON:

{{
    "verdict": "TRUE | FALSE | PARTIALLY_TRUE | PARTIALLY_FALSE | UNVERIFIABLE",
    "final_confidence": <0-100 as integer>,
    "justification": "<2-4 sentences citing specific persona insights and evidence>",
    "fact_checker_summary": "<one line from the selected Fact Checker result>",
    "logical_analyst_summary": "<one line from the selected Logical Analyst result>",
    "bias_detector_summary": "<one line from the selected Bias Detector result>",
    "recommendation": "Do not share | Likely accurate | Verify independently | Partially accurate"
}}
"""


EXPECTED_ANGLES = [
    "Fact Checker",
    "Logical Analyst",
    "Bias Detector"
]


def final_judgment_node(state: VerificationState) -> VerificationState:
    """
    Final supervisor node.

    Resolves the latest available result for each persona,
    builds the final judgment prompt, and generates the verdict.
    """

    # Pipeline already ended
    if state.get("verdict"):
        return state

    claim = state["claim"]
    memory = state.get("qa_memory", [])
    evidence = state.get("evidence", [])
    qa_history = state.get("qa_history", [])
    rounds_executed = state.get("rounds_executed", 1)

    llm_calls = state.get("total_llm_calls", 0)

    # ── Resolve latest persona results ───────────────────────────

    resolved_personas = _resolve_latest_personas(
        memory=memory,
        qa_history=qa_history
    )

    print(
        "[final_judgment] "
        f"Fact Checker → R{resolved_personas['Fact Checker']['round']} | "
        f"Logical Analyst → R{resolved_personas['Logical Analyst']['round']} | "
        f"Bias Detector → R{resolved_personas['Bias Detector']['round']}"
    )

    # ── Coverage ─────────────────────────────────────────────────

    covered_personas = [
        persona
        for persona, result in resolved_personas.items()
        if result.get("available", False)
    ]

    coverage_score = (
        len(covered_personas) /
        len(EXPECTED_ANGLES)
    )

    print(
        f"[final_judgment] "
        f"Coverage: {coverage_score:.0%} | "
        f"Rounds: {rounds_executed}"
    )

    # ── Stance breakdown ─────────────────────────────────────────

    stances = [
        ev.get("stance", "NEUTRAL")
        for ev in evidence
    ]

    stance_breakdown = {
        "SUPPORT": stances.count("SUPPORT"),
        "CONTRADICT": stances.count("CONTRADICT"),
        "NEUTRAL": stances.count("NEUTRAL"),
    }

    # ── Build prompt information ─────────────────────────────────

    evidence_summary = _build_evidence_summary(evidence)

    memory_text = _build_memory_text(memory)

    persona_analysis = _build_persona_analysis(
        resolved_personas
    )

    # ── Final DeepSeek prompt ────────────────────────────────────

    prompt = JUDGMENT_PROMPT.format(
        claim=claim,

        evidence_summary=evidence_summary,

        persona_analysis=persona_analysis,

        memory_text=memory_text,

        factual_covered=(
            "Yes"
            if resolved_personas["Fact Checker"]["available"]
            else "No"
        ),

        logical_covered=(
            "Yes"
            if resolved_personas["Logical Analyst"]["available"]
            else "No"
        ),

        bias_covered=(
            "Yes"
            if resolved_personas["Bias Detector"]["available"]
            else "No"
        ),

        support_count=stance_breakdown["SUPPORT"],

        contradict_count=stance_breakdown["CONTRADICT"],

        neutral_count=stance_breakdown["NEUTRAL"],

        rounds_executed=rounds_executed,
    )

    # ── Generate final verdict ───────────────────────────────────

    try:

        response = call_deepseek_json(prompt)

        llm_calls += 1

        verdict = response.get(
            "verdict",
            "UNVERIFIABLE"
        )

        confidence = float(
            response.get(
                "final_confidence",
                50.0
            )
        )

        justification = response.get(
            "justification",
            "Unable to generate justification."
        )

        recommendation = response.get(
            "recommendation",
            "Verify independently"
        )

        persona_insights = {
            "fact_checker":
                response.get(
                    "fact_checker_summary",
                    ""
                ),

            "logical_analyst":
                response.get(
                    "logical_analyst_summary",
                    ""
                ),

            "bias_detector":
                response.get(
                    "bias_detector_summary",
                    ""
                ),
        }

        print(
            f"[final_judgment] "
            f"Verdict: {verdict} "
            f"({confidence:.0f}%)"
        )

    except Exception as e:

        print(
            f"[final_judgment] "
            f"DeepSeek failed: {e}. "
            f"Using fallback verdict."
        )

        (
            verdict,
            confidence,
            justification,
            recommendation
        ) = _fallback_verdict(
            stance_breakdown,
            state.get(
                "avg_source_quality",
                0.5
            )
        )

        persona_insights = {
            "fact_checker":
                resolved_personas[
                    "Fact Checker"
                ].get("insight", ""),

            "logical_analyst":
                resolved_personas[
                    "Logical Analyst"
                ].get("insight", ""),

            "bias_detector":
                resolved_personas[
                    "Bias Detector"
                ].get("insight", ""),
        }

        llm_calls += 1

    return {
        **state,

        "verdict": verdict,

        "final_confidence": confidence,

        "justification": justification,

        "stance_breakdown": stance_breakdown,

        "persona_insights": persona_insights,

        "coverage_score": round(
            coverage_score,
            2
        ),

        "recommendation": recommendation,

        "total_llm_calls": llm_calls,

        "pipeline_error": None,
    }


# ──────────────────────────────────────────────────────────────
# Persona resolution
# ──────────────────────────────────────────────────────────────


def _resolve_latest_personas(
    memory: list,
    qa_history: list
) -> dict:
    """
    Resolve the latest available result for each persona.

    Resolution rules:

    Fact Checker:
        R2 > R1

    Logical Analyst:
        R3 > R1

    Bias Detector:
        R1 only
    """

    resolved = {
        "Fact Checker": {
            "available": False,
            "round": 0,
            "question": "",
            "answer": "",
            "insight": "",
            "stance": "UNCERTAIN",
            "confidence": 0.0,
        },

        "Logical Analyst": {
            "available": False,
            "round": 0,
            "question": "",
            "answer": "",
            "insight": "",
            "stance": "UNCERTAIN",
            "confidence": 0.0,
        },

        "Bias Detector": {
            "available": False,
            "round": 0,
            "question": "",
            "answer": "",
            "insight": "",
            "stance": "UNCERTAIN",
            "confidence": 0.0,
        },
    }

    # ── First read normal Q&A memory ─────────────────────────────

    for entry in memory:

        persona = entry.get(
            "persona",
            ""
        )

        if persona not in resolved:
            continue

        round_num = int(
            entry.get(
                "round",
                1
            )
        )

        current_round = resolved[
            persona
        ]["round"]

        # Only replace when this is a newer round
        if round_num >= current_round:

            resolved[persona] = {
                "available": True,

                "round": round_num,

                "question": entry.get(
                    "question",
                    ""
                ),

                "answer": entry.get(
                    "answer",
                    ""
                ),

                "insight": entry.get(
                    "insight",
                    ""
                ),

                "stance": entry.get(
                    "stance",
                    "UNCERTAIN"
                ),

                "confidence": float(
                    entry.get(
                        "confidence",
                        0.0
                    )
                ),
            }

    # ── Then read structured QA history ──────────────────────────
    #
    # qa_history takes priority because Round 2/3 nodes
    # store their structured persona results here.

    for round_entry in qa_history:

        round_num = int(
            round_entry.get(
                "round",
                0
            )
        )

        for persona_key, persona_name in [
            ("fact_checker", "Fact Checker"),
            ("logical_analyst", "Logical Analyst"),
            ("bias_detector", "Bias Detector"),
        ]:

            if persona_key not in round_entry:
                continue

            data = round_entry[
                persona_key
            ]

            if not isinstance(data, dict):
                continue

            current_round = resolved[
                persona_name
            ]["round"]

            if round_num < current_round:
                continue

            resolved[persona_name].update({
                "available": True,

                "round": round_num,

                "stance": data.get(
                    "stance",
                    resolved[
                        persona_name
                    ].get(
                        "stance",
                        "UNCERTAIN"
                    )
                ),

                "confidence": float(
                    data.get(
                        "confidence",
                        resolved[
                            persona_name
                        ].get(
                            "confidence",
                            0.0
                        )
                    )
                ),

                "reasoning": data.get(
                    "reasoning",
                    ""
                ),
            })

            # If structured reasoning exists,
            # use it as the answer/insight source.
            if data.get("reasoning"):
                resolved[persona_name][
                    "answer"
                ] = data["reasoning"]

                resolved[persona_name][
                    "insight"
                ] = data["reasoning"]

    return resolved


# ──────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────


def _build_persona_analysis(
    resolved_personas: dict
) -> str:
    """
    Build a concise representation of the selected
    latest persona results.
    """

    lines = []

    for persona in EXPECTED_ANGLES:

        result = resolved_personas[
            persona
        ]

        if not result.get("available"):
            lines.append(
                f"[{persona}] No result available."
            )
            continue

        round_num = result.get(
            "round",
            1
        )

        stance = result.get(
            "stance",
            "UNCERTAIN"
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        insight = result.get(
            "insight",
            ""
        )

        answer = result.get(
            "answer",
            ""
        )

        if not insight:
            insight = answer

        lines.append(
            f"[{persona} · Round {round_num}]\n"
            f"Stance: {stance}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Insight: {insight}"
        )

    return "\n\n".join(lines)


def _build_evidence_summary(
    evidence: list
) -> str:

    if not evidence:
        return "No evidence retrieved."

    lines = []

    for ev in evidence[:5]:

        domain = ev.get(
            "source_domain",
            "unknown"
        )

        r_score = ev.get(
            "combined_reliability",
            0.5
        )

        stance = ev.get(
            "stance",
            "NEUTRAL"
        )

        content = ev.get(
            "content",
            ""
        )[:150]

        lines.append(
            f"- {domain} "
            f"(R={r_score:.2f}, "
            f"stance={stance}): "
            f"{content}"
        )

    return "\n".join(lines)


def _build_memory_text(
    memory: list
) -> str:

    if not memory:
        return "No Q&A memory available."

    lines = []

    for entry in memory:

        lines.append(
            f"[Round {entry.get('round', 1)} "
            f"· {entry.get('persona', 'Unknown')}]\n"
            f"Q: {entry.get('question', '')}\n"
            f"A: {entry.get('answer', '')}\n"
            f"Insight: {entry.get('insight', '')}"
        )

    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Fallback verdict
# ──────────────────────────────────────────────────────────────


def _fallback_verdict(
    stance_breakdown: dict,
    avg_quality: float
) -> tuple:
    """
    Rule-based fallback if DeepSeek fails.

    Returns:
        verdict,
        confidence,
        justification,
        recommendation
    """

    support = stance_breakdown.get(
        "SUPPORT",
        0
    )

    contradict = stance_breakdown.get(
        "CONTRADICT",
        0
    )

    neutral = stance_breakdown.get(
        "NEUTRAL",
        0
    )

    total = (
        support +
        contradict +
        neutral
    )

    if total == 0:

        return (
            "UNVERIFIABLE",
            20.0,
            "No evidence was available "
            "to verify this claim.",
            "Verify independently"
        )

    contradict_ratio = (
        contradict / total
    )

    support_ratio = (
        support / total
    )

    if contradict_ratio >= 0.7:

        return (
            "FALSE",
            round(
                60 +
                contradict_ratio * 35,
                1
            ),
            f"{contradict}/{total} "
            "sources contradict the claim.",
            "Do not share"
        )

    elif support_ratio >= 0.7:

        return (
            "TRUE",
            round(
                60 +
                support_ratio * 35,
                1
            ),
            f"{support}/{total} "
            "sources support the claim.",
            "Likely accurate"
        )

    else:

        return (
            "PARTIALLY_TRUE",
            50.0,
            "Evidence is mixed — some "
            "sources support, others "
            "contradict.",
            "Verify independently"
        )
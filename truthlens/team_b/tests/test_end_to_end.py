import sys
from pathlib import Path
from unittest.mock import patch


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRUTHLENS_DIR = PROJECT_ROOT / "truthlens"

if str(TRUTHLENS_DIR) not in sys.path:
    sys.path.insert(0, str(TRUTHLENS_DIR))


from team_b.graph import run_graph


# ============================================================
# TEST CLAIM
# ============================================================

FAKE_CLAIM = "Drinking whiskey cures COVID-19"


# ============================================================
# MOCK DECOMPOSITION
# ============================================================

FAKE_DECOMPOSE_RESPONSE = {
    "sub_claims": [
        "Whiskey can cure COVID-19"
    ],
    "claim_type": "medical",
    "search_queries": [
        "whiskey cures COVID-19"
    ]
}


# ============================================================
# NORMAL TEST EVIDENCE
#
# 2 high-quality
# 1 low-quality
#
# This deliberately causes the 3-round path.
# ============================================================

ALL_EVIDENCE = [
    {
        "title": "Medical Evidence 1",
        "content": "Medical evidence contradicts the claim.",
        "url": "https://example.com/medical1",
        "combined_reliability": 0.94,
        "stance": "CONTRADICT"
    },
    {
        "title": "Medical Evidence 2",
        "content": "Clinical evidence does not support the claim.",
        "url": "https://example.com/medical2",
        "combined_reliability": 0.87,
        "stance": "CONTRADICT"
    },
    {
        "title": "Low Quality Evidence",
        "content": "An unreliable source makes the claim.",
        "url": "https://example.com/low",
        "combined_reliability": 0.44,
        "stance": "NEUTRAL"
    }
]


# ============================================================
# HIGH-CONFIDENCE TEST EVIDENCE
#
# ALL THREE evidence items CONTRADICT the claim.
#
# This gives the confidence gate a consensus of 1.0.
# ============================================================

HIGH_CONFIDENCE_EVIDENCE = [
    {
        "title": "Medical Evidence 1",
        "content": "Medical evidence contradicts the claim.",
        "url": "https://example.com/medical1",
        "combined_reliability": 0.94,
        "stance": "CONTRADICT"
    },
    {
        "title": "Medical Evidence 2",
        "content": "Clinical evidence does not support the claim.",
        "url": "https://example.com/medical2",
        "combined_reliability": 0.87,
        "stance": "CONTRADICT"
    },
    {
        "title": "Medical Evidence 3",
        "content": "Additional evidence contradicts the claim.",
        "url": "https://example.com/medical3",
        "combined_reliability": 0.90,
        "stance": "CONTRADICT"
    }
]


# ============================================================
# ROUND 1 ANSWERS
# ============================================================

CERTAIN_ANSWER = (
    "The available evidence directly contradicts the claim. "
    "The evidence clearly indicates that the claim is not supported."
)


UNCERTAIN_ANSWER = (
    "The evidence is unclear and does not directly address this claim."
)


# ============================================================
# ROUND 2 MOCK RESPONSE
# ============================================================

STRUCTURED_FACT_CHECKER = {
    "stance": "CONTRADICT",
    "confidence": 0.90,
    "reasoning": (
        "The high-quality evidence contradicts the claim. "
        "There is no reliable evidence that whiskey cures COVID-19."
    )
}


# ============================================================
# ROUND 3 MOCK RESPONSE
# ============================================================

STRUCTURED_LOGICAL_ANALYST = {
    "stance": "CONTRADICT",
    "confidence": 0.88,
    "reasoning": (
        "The low-quality evidence does not establish the claim. "
        "The claim is not logically supported by the available evidence."
    )
}


# ============================================================
# FINAL JUDGMENT MOCK RESPONSE
# ============================================================

FINAL_JUDGMENT = {
    "verdict": "FALSE",
    "confidence": 95,
    "justification": (
        "The available evidence contradicts the claim. "
        "Reliable medical evidence does not support whiskey as a cure."
    ),
    "recommendation": (
        "Do not treat whiskey as a cure for COVID-19."
    )
}


# ============================================================
# TEST 1
#
# LOW CONFIDENCE
#
# Expected:
#
# Round 1
#     ↓
# Confidence < 0.85
#     ↓
# Round 2 Fact Checker
#     ↓
# Round 3 Logical Analyst
#     ↓
# Final Judgment
# ============================================================

def test_full_three_round_pipeline_cpu_safe():

    with patch(
        "team_b.nodes.decompose_claim.call_deepseek_json",
        return_value=FAKE_DECOMPOSE_RESPONSE
    ), \
    patch(
        "team_b.personas.fact_checker.generate_question",
        return_value="Does reliable evidence support this claim?"
    ), \
    patch(
        "team_b.personas.fact_checker.generate_answer",
        return_value=UNCERTAIN_ANSWER
    ), \
    patch(
        "team_b.personas.logical_analyst.generate_question",
        return_value="Is the claim logically supported by the evidence?"
    ), \
    patch(
        "team_b.personas.logical_analyst.generate_answer",
        return_value=UNCERTAIN_ANSWER
    ), \
    patch(
        "team_b.personas.bias_detector.generate_question",
        return_value="Could bias affect the interpretation?"
    ), \
    patch(
        "team_b.personas.bias_detector.generate_answer",
        return_value=UNCERTAIN_ANSWER
    ), \
    patch(
        "team_b.personas.fact_checker.generate_structured_answer",
        return_value=STRUCTURED_FACT_CHECKER
    ), \
    patch(
        "team_b.personas.logical_analyst.generate_structured_answer",
        return_value=STRUCTURED_LOGICAL_ANALYST
    ), \
    patch(
        "team_b.nodes.final_judgment.call_deepseek_json",
        return_value=FINAL_JUDGMENT
    ):

        result = run_graph(
            claim=FAKE_CLAIM,
            evidence=ALL_EVIDENCE,
            original_claim="क्या व्हिस्की से COVID ठीक होता है?",
            original_language="hindi",
            sub_claims=[],
            thread_id="cpu-test-three-rounds"
        )


    print()
    print("=" * 55)
    print("THREE ROUND TEST")
    print("=" * 55)

    print("QA Round:", result.get("qa_round"))
    print("Rounds Executed:", result.get("rounds_executed"))
    print("QA History:", result.get("qa_history"))
    print("QA Memory Entries:", len(result.get("qa_memory", [])))
    print("Verdict:", result.get("verdict"))
    print("Final Confidence:", result.get("final_confidence"))
    print("Total LLM Calls:", result.get("total_llm_calls"))
    print("Pipeline Error:", result.get("pipeline_error"))

    print("=" * 55)
    print()


    # Three rounds must execute.

    assert result["qa_round"] == 3

    assert result["rounds_executed"] == 3


    # R1:
    # Fact Checker
    # Logical Analyst
    # Bias Detector
    #
    # R2:
    # Fact Checker
    #
    # R3:
    # Logical Analyst

    assert len(result["qa_memory"]) == 5


    # R2 + R3 history

    assert len(result["qa_history"]) == 2


    # Final judgment

    assert result["verdict"] == "FALSE"


    # We only require a valid final confidence.
    #
    # The final judgment node currently produces its own
    # final confidence value.

    assert isinstance(
        result["final_confidence"],
        (int, float)
    )

    assert 0 <= result["final_confidence"] <= 100


    # Expected calls:
    #
    # Decompose = 1
    # R1 = 6
    # R2 = 2
    # R3 = 2
    # Final = 1
    #
    # Total = 12

    assert result["total_llm_calls"] == 12

    assert result["pipeline_error"] is None


# ============================================================
# TEST 2
#
# HIGH CONFIDENCE
#
# Expected:
#
# Round 1
#     ↓
# Confidence >= 0.85
#     ↓
# Final Judgment
#
# Round 2 and Round 3 must NOT execute.
# ============================================================

def test_high_confidence_skips_round2_and_round3():

    with patch(
        "team_b.nodes.decompose_claim.call_deepseek_json",
        return_value=FAKE_DECOMPOSE_RESPONSE
    ), \
    patch(
        "team_b.personas.fact_checker.generate_question",
        return_value="Does reliable evidence support this claim?"
    ), \
    patch(
        "team_b.personas.fact_checker.generate_answer",
        return_value=CERTAIN_ANSWER
    ), \
    patch(
        "team_b.personas.logical_analyst.generate_question",
        return_value="Is the claim logically supported?"
    ), \
    patch(
        "team_b.personas.logical_analyst.generate_answer",
        return_value=CERTAIN_ANSWER
    ), \
    patch(
        "team_b.personas.bias_detector.generate_question",
        return_value="Could bias affect the interpretation?"
    ), \
    patch(
        "team_b.personas.bias_detector.generate_answer",
        return_value=CERTAIN_ANSWER
    ), \
    patch(
        "team_b.personas.fact_checker.generate_structured_answer",
        side_effect=AssertionError(
            "Round 2 should NOT execute"
        )
    ), \
    patch(
        "team_b.personas.logical_analyst.generate_structured_answer",
        side_effect=AssertionError(
            "Round 3 should NOT execute"
        )
    ), \
    patch(
        "team_b.nodes.final_judgment.call_deepseek_json",
        return_value=FINAL_JUDGMENT
    ):

        result = run_graph(
            claim=FAKE_CLAIM,
            evidence=HIGH_CONFIDENCE_EVIDENCE,
            original_claim="क्या व्हिस्की से COVID ठीक होता है?",
            original_language="hindi",
            sub_claims=[],
            thread_id="cpu-test-early-exit"
        )


    print()
    print("=" * 55)
    print("EARLY EXIT TEST")
    print("=" * 55)

    print("QA Round:", result.get("qa_round"))
    print("Rounds Executed:", result.get("rounds_executed"))
    print("QA Memory Entries:", len(result.get("qa_memory", [])))
    print("QA History:", result.get("qa_history"))
    print("Verdict:", result.get("verdict"))
    print("Final Confidence:", result.get("final_confidence"))
    print("Total LLM Calls:", result.get("total_llm_calls"))
    print("Pipeline Error:", result.get("pipeline_error"))

    print("=" * 55)
    print()


    # Only Round 1 should execute.

    assert result["qa_round"] == 1

    assert result["rounds_executed"] == 1


    # Only 3 Round-1 personas.

    assert len(result["qa_memory"]) == 3


    # No Round 2 / Round 3 history.

    assert len(result["qa_history"]) == 0


    # Final judgment still runs.

    assert result["verdict"] == "FALSE"


    # Expected calls:
    #
    # Decompose = 1
    # R1 = 6
    # Final = 1
    #
    # Total = 8

    assert result["total_llm_calls"] == 8

    assert result["pipeline_error"] is None
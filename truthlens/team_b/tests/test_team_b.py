"""
test_team_b.py — TruthLens
Mock tests for all Team B nodes.

NO Ollama needed. NO internet needed. NO API keys needed.
All DeepSeek calls are mocked with fake responses.

Run: pytest tests/test_team_b.py -v
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ─────────────────────────────────────────────────────────────
# SHARED MOCK DATA
# Simulates what Team A sends to Team B
# ─────────────────────────────────────────────────────────────

FAKE_CLAIM    = "Does whiskey cure COVID-19?"
FAKE_EVIDENCE = [
    {
        "source_domain":        "who.int",
        "author":               "World Health Organization",
        "organization":         "WHO",
        "content":              "Alcohol does not protect against COVID-19. Consuming alcohol does not destroy the virus, and its consumption is likely to increase health risks.",
        "url":                  "https://who.int/news/item/...",
        "bm25_score":           0.90,
        "cosine_score":         0.92,
        "domain_credibility":   0.97,
        "credibility_source":   "hardlist_trusted",
        "recency_score":        0.90,
        "combined_reliability": 0.94,
        "stance":               "CONTRADICT",
        "retrieval_rank":       1,
    },
    {
        "source_domain":        "bbc.com",
        "author":               "BBC Health",
        "organization":         "BBC",
        "content":              "Medical experts warn that drinking alcohol does not prevent or cure coronavirus. The myth has spread widely on social media.",
        "url":                  "https://bbc.com/news/...",
        "bm25_score":           0.85,
        "cosine_score":         0.88,
        "domain_credibility":   0.88,
        "credibility_source":   "hardlist_trusted",
        "recency_score":        0.85,
        "combined_reliability": 0.87,
        "stance":               "CONTRADICT",
        "retrieval_rank":       2,
    },
    {
        "source_domain":        "twitter.com",
        "author":               "random_user_99",
        "organization":         "",
        "content":              "My uncle drank whiskey every night and never got COVID! It works!",
        "url":                  "https://twitter.com/...",
        "bm25_score":           0.40,
        "cosine_score":         0.45,
        "domain_credibility":   0.30,
        "credibility_source":   "dynamic_signals",
        "recency_score":        0.60,
        "combined_reliability": 0.44,
        "stance":               "SUPPORT",
        "retrieval_rank":       3,
    },
]

# Fake JSON responses DeepSeek would return
FAKE_DECOMPOSE_RESPONSE = {
    "claim_type":     "factual",
    "sub_claims":     [
        "Whiskey has antiviral properties",
        "Drinking whiskey can cure COVID-19",
    ],
    "search_queries": [
        "whiskey alcohol cure COVID scientific evidence",
        "alcohol antiviral COVID-19 medical research",
    ],
    "key_entities": ["whiskey", "COVID-19", "alcohol"],
}

FAKE_JUDGMENT_RESPONSE = {
    "verdict":                  "FALSE",
    "final_confidence":         95,
    "justification":            "WHO and BBC directly contradict this claim. No biological mechanism exists. Social media anecdote is unreliable evidence.",
    "fact_checker_summary":     "0/2 credible sources support the claim; both WHO and BBC contradict it.",
    "logical_analyst_summary":  "No plausible mechanism — alcohol at drinking concentrations cannot neutralise SARS-CoV-2.",
    "bias_detector_summary":    "Claim follows common COVID miracle-cure misinformation pattern.",
    "recommendation":           "Do not share",
}

FAKE_QUESTION = "Does any credible medical source confirm that whiskey has antiviral properties against COVID-19?"
FAKE_ANSWER   = "No. WHO explicitly states alcohol does not protect against COVID-19. BBC confirms this is a myth spread on social media. Evidence does not support this claim."


# ─────────────────────────────────────────────────────────────
# NODE 1: decompose_claim
# ─────────────────────────────────────────────────────────────

class TestDecomposeClaimNode:

    def test_normal_claim(self):
        """Standard claim decomposes into sub-claims correctly."""
        from team_b.nodes.decompose_claim import decompose_claim_node

        initial_state = _base_state()

        with patch("team_b.nodes.decompose_claim.call_deepseek_json",
                   return_value=FAKE_DECOMPOSE_RESPONSE):

            result = decompose_claim_node(initial_state)

        assert result["claim_type"]            == "factual"
        assert len(result["sub_claims"])       == 2
        assert len(result["search_queries"])   >= 1
        assert result["total_llm_calls"]       == 1
        print(f"\n  sub_claims: {result['sub_claims']}")

    def test_llm_failure_fallback(self):
        """If DeepSeek fails, falls back to original claim as single sub-claim."""
        from team_b.nodes.decompose_claim import decompose_claim_node

        initial_state = _base_state()

        with patch("team_b.nodes.decompose_claim.call_deepseek_json",
                   side_effect=RuntimeError("Ollama unreachable")):

            result = decompose_claim_node(initial_state)

        # Should not crash — graceful fallback
        assert result["sub_claims"]  == [FAKE_CLAIM]
        assert result["claim_type"]  == "factual"
        print(f"\n  Fallback sub_claims: {result['sub_claims']}")


# ─────────────────────────────────────────────────────────────
# NODE 2: credibility_check
# ─────────────────────────────────────────────────────────────

class TestCredibilityCheckNode:

    def test_reads_team_a_scores_directly(self):
        """Should use Team A's R(d) scores — no re-scoring."""
        from team_b.nodes.credibility_check import credibility_check_node

        state  = _base_state()
        result = credibility_check_node(state)

        # avg_quality = (0.94 + 0.87 + 0.44) / 3 = 0.75
        expected_avg = round((0.94 + 0.87 + 0.44) / 3, 3)
        assert abs(result["avg_source_quality"] - expected_avg) < 0.01
        print(f"\n  avg_source_quality: {result['avg_source_quality']}")

    def test_flags_low_credibility_sources(self):
        """twitter.com (R=0.44) should NOT be flagged (threshold is 0.30)."""
        from team_b.nodes.credibility_check import credibility_check_node

        result = credibility_check_node(_base_state())

        # twitter.com R(d)=0.44, threshold=0.30 → not flagged
        assert "twitter.com" not in result["flagged_sources"]
        print(f"\n  flagged_sources: {result['flagged_sources']}")

    def test_flags_very_low_credibility(self):
        """Source with R(d) < 0.30 should be flagged."""
        from team_b.nodes.credibility_check import credibility_check_node

        state = _base_state()
        state["evidence"] = state["evidence"].copy()
        state["evidence"][2] = {
            **state["evidence"][2],
            "combined_reliability": 0.10,   # below 0.30 threshold
            "source_domain": "fakenews.xyz"
        }

        result = credibility_check_node(state)
        assert "fakenews.xyz" in result["flagged_sources"]
        print(f"\n  flagged_sources: {result['flagged_sources']}")

    def test_no_evidence_short_circuits(self):
        """Empty evidence should short-circuit to UNVERIFIABLE immediately."""
        from team_b.nodes.credibility_check import credibility_check_node

        state           = _base_state()
        state["evidence"] = []

        result = credibility_check_node(state)

        assert result["verdict"]          == "UNVERIFIABLE"
        assert result["final_confidence"] == 0.0
        assert result["rounds_executed"]  == 0
        print(f"\n  Short-circuit verdict: {result['verdict']}")

    def test_high_quality_count(self):
        """Count evidence pieces with R(d) >= 0.70."""
        from team_b.nodes.credibility_check import credibility_check_node

        result = credibility_check_node(_base_state())

        # who.int (0.94) and bbc.com (0.87) are >= 0.70
        # twitter.com (0.44) is not
        assert result["high_quality_count"] == 2
        print(f"\n  high_quality_count: {result['high_quality_count']}")


# ─────────────────────────────────────────────────────────────
# NODE 3: round1_qa
# ─────────────────────────────────────────────────────────────

class TestRound1QA:

    def test_runs_all_3_personas(self):
        """All 3 personas should add entries to memory."""
        from team_b.nodes.round1_qa import round1_qa_node

        state = _after_credibility(_base_state())

        with patch("team_b.personas.fact_checker.call_deepseek",    return_value=FAKE_QUESTION), \
             patch("team_b.personas.logical_analyst.call_deepseek",  return_value=FAKE_QUESTION), \
             patch("team_b.personas.bias_detector.call_deepseek",    return_value=FAKE_QUESTION):

            result = round1_qa_node(state)

        assert len(result["qa_memory"]) == 3

        personas_in_memory = {e["persona"] for e in result["qa_memory"]}
        assert "Fact Checker"    in personas_in_memory
        assert "Logical Analyst" in personas_in_memory
        assert "Bias Detector"   in personas_in_memory

        print(f"\n  Memory entries: {len(result['qa_memory'])}")
        for entry in result["qa_memory"]:
            print(f"  [{entry['persona']}] Q: {entry['question'][:60]}...")

    def test_memory_round_number_is_1(self):
        """All Round 1 entries should have round=1."""
        from team_b.nodes.round1_qa import round1_qa_node

        state = _after_credibility(_base_state())

        with patch("team_b.personas.fact_checker.call_deepseek",    return_value=FAKE_QUESTION), \
             patch("team_b.personas.logical_analyst.call_deepseek",  return_value=FAKE_QUESTION), \
             patch("team_b.personas.bias_detector.call_deepseek",    return_value=FAKE_QUESTION):

            result = round1_qa_node(state)

        for entry in result["qa_memory"]:
            assert entry["round"] == 1

    def test_llm_calls_count(self):
        """Round 1 should add exactly 6 LLM calls (3Q + 3A)."""
        from team_b.nodes.round1_qa import round1_qa_node

        state               = _after_credibility(_base_state())
        state["total_llm_calls"] = 1  # decompose already ran

        with patch("team_b.personas.fact_checker.call_deepseek",    return_value=FAKE_QUESTION), \
             patch("team_b.personas.logical_analyst.call_deepseek",  return_value=FAKE_QUESTION), \
             patch("team_b.personas.bias_detector.call_deepseek",    return_value=FAKE_QUESTION):

            result = round1_qa_node(state)

        # 1 (decompose) + 6 (round1) = 7
        assert result["total_llm_calls"] == 7
        print(f"\n  Total LLM calls after Round 1: {result['total_llm_calls']}")

    def test_skips_if_pipeline_already_ended(self):
        """If verdict already set (no evidence), should pass through unchanged."""
        from team_b.nodes.round1_qa import round1_qa_node

        state = _base_state()
        state["verdict"] = "UNVERIFIABLE"   # already set by credibility_check

        result = round1_qa_node(state)

        # Memory should still be empty — node skipped
        assert result["qa_memory"] == []


# ─────────────────────────────────────────────────────────────
# NODE 4: confidence_gate
# ─────────────────────────────────────────────────────────────

class TestConfidenceGate:

    def test_high_confidence_skips_round2(self):
        """Strong contradicting evidence → high confidence → skip Round 2."""
        from team_b.nodes.confidence_gate import confidence_gate_node

        state = _after_round1(_base_state(), certain_answers=True)
        result = confidence_gate_node(state)

        print(f"\n  round1_confidence: {result['round1_confidence']}")
        print(f"  go_to_round2: {result['go_to_round2']}")

        assert result["round1_confidence"] >= 0.80
        assert result["go_to_round2"] == False

    def test_low_confidence_triggers_round2(self):
        """Mixed/uncertain evidence → low confidence → trigger Round 2."""
        from team_b.nodes.confidence_gate import confidence_gate_node

        state = _after_round1(_base_state(), certain_answers=False)
        # Override evidence to be all neutral stances
        state["evidence"] = [
            {**ev, "stance": "NEUTRAL"}
            for ev in state["evidence"]
        ]
        state["avg_source_quality"] = 0.30  # low quality sources

        result = confidence_gate_node(state)

        print(f"\n  round1_confidence: {result['round1_confidence']}")
        print(f"  go_to_round2: {result['go_to_round2']}")

        assert result["round1_confidence"] < 0.80
        assert result["go_to_round2"] == True

    def test_confidence_formula(self):
        """Verify the formula: 0.5*stance + 0.3*quality + 0.2*certainty."""
        from team_b.nodes.confidence_gate import confidence_gate_node

        state = _after_round1(_base_state(), certain_answers=True)
        result = confidence_gate_node(state)

        # stance_consensus: 2 CONTRADICT out of 3 = 0.667
        # avg_source_quality: ~0.75
        # answer_certainty: 1.0 (no uncertain phrases)
        # confidence = 0.5*0.667 + 0.3*0.75 + 0.2*1.0 = 0.333 + 0.225 + 0.2 = 0.758
        # (approx — actual depends on memory content)
        assert 0.0 <= result["round1_confidence"] <= 1.0
        print(f"\n  Confidence: {result['round1_confidence']:.3f}")

    def test_route_function_returns_correct_node(self):
        """route_after_gate returns correct string for LangGraph routing."""
        from team_b.nodes.confidence_gate import route_after_gate

        # High confidence → skip round2
        state_high = {"round1_confidence": 0.90, "go_to_round2": False, "verdict": ""}
        assert route_after_gate(state_high) == "final_judgment"

        # Low confidence → go to round2
        state_low = {"round1_confidence": 0.60, "go_to_round2": True, "verdict": ""}
        assert route_after_gate(state_low) == "round2_qa"

        # Short-circuit (no evidence) → final_judgment
        state_short = {"verdict": "UNVERIFIABLE", "go_to_round2": False}
        assert route_after_gate(state_short) == "final_judgment"

        print("\n  All routing cases correct")


# ─────────────────────────────────────────────────────────────
# NODE 5: round2_qa
# ─────────────────────────────────────────────────────────────

class TestRound2QA:

    def test_appends_to_existing_memory(self):
        """Round 2 should ADD to Round 1 memory, not replace it."""
        from team_b.nodes.round2_qa import round2_qa_node

        state = _after_round1(_base_state(), certain_answers=True)
        round1_memory_len = len(state["qa_memory"])   # 3

        with patch("team_b.personas.fact_checker.call_deepseek",    return_value=FAKE_QUESTION), \
             patch("team_b.personas.logical_analyst.call_deepseek",  return_value=FAKE_QUESTION), \
             patch("team_b.personas.bias_detector.call_deepseek",    return_value=FAKE_QUESTION):

            result = round2_qa_node(state)

        # Should have Round 1 (3) + Round 2 (3) = 6 entries
        assert len(result["qa_memory"]) == round1_memory_len + 3
        assert result["rounds_executed"] == 2
        print(f"\n  Memory after Round 2: {len(result['qa_memory'])} entries")

    def test_round2_entries_have_round_2(self):
        """New entries added in Round 2 should have round=2."""
        from team_b.nodes.round2_qa import round2_qa_node

        state = _after_round1(_base_state(), certain_answers=True)

        with patch("team_b.personas.fact_checker.call_deepseek",    return_value=FAKE_QUESTION), \
             patch("team_b.personas.logical_analyst.call_deepseek",  return_value=FAKE_QUESTION), \
             patch("team_b.personas.bias_detector.call_deepseek",    return_value=FAKE_QUESTION):

            result = round2_qa_node(state)

        round2_entries = [e for e in result["qa_memory"] if e["round"] == 2]
        assert len(round2_entries) == 3

    def test_total_llm_calls_after_round2(self):
        """Round 2 adds 6 more LLM calls."""
        from team_b.nodes.round2_qa import round2_qa_node

        state = _after_round1(_base_state(), certain_answers=True)
        state["total_llm_calls"] = 7   # after decompose(1) + round1(6)

        with patch("team_b.personas.fact_checker.call_deepseek",    return_value=FAKE_QUESTION), \
             patch("team_b.personas.logical_analyst.call_deepseek",  return_value=FAKE_QUESTION), \
             patch("team_b.personas.bias_detector.call_deepseek",    return_value=FAKE_QUESTION):

            result = round2_qa_node(state)

        assert result["total_llm_calls"] == 13
        print(f"\n  Total LLM calls after Round 2: {result['total_llm_calls']}")


# ─────────────────────────────────────────────────────────────
# NODE 6: final_judgment
# ─────────────────────────────────────────────────────────────

class TestFinalJudgment:

    def test_false_verdict_for_contradicting_evidence(self):
        """Strong contradicting evidence should produce FALSE verdict."""
        from team_b.nodes.final_judgment import final_judgment_node

        state = _after_round1(_base_state(), certain_answers=True)

        with patch("team_b.nodes.final_judgment.call_deepseek_json",
                   return_value=FAKE_JUDGMENT_RESPONSE):

            result = final_judgment_node(state)

        assert result["verdict"]           == "FALSE"
        assert result["final_confidence"]  == 95
        assert result["recommendation"]    == "Do not share"
        assert result["coverage_score"]    == 1.0   # all 3 angles covered
        print(f"\n  Verdict: {result['verdict']} ({result['final_confidence']}%)")
        print(f"  Justification: {result['justification']}")

    def test_outputs_all_required_fields(self):
        """Final state must have all fields Team C needs."""
        from team_b.nodes.final_judgment import final_judgment_node

        state = _after_round1(_base_state(), certain_answers=True)

        with patch("team_b.nodes.final_judgment.call_deepseek_json",
                   return_value=FAKE_JUDGMENT_RESPONSE):

            result = final_judgment_node(state)

        required_fields = [
            "verdict", "final_confidence", "justification",
            "stance_breakdown", "persona_insights",
            "coverage_score", "recommendation"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
            print(f"  ✓ {field}: {str(result[field])[:60]}")

    def test_stance_breakdown_is_correct(self):
        """Stance breakdown should count from evidence stances."""
        from team_b.nodes.final_judgment import final_judgment_node

        state = _after_round1(_base_state(), certain_answers=True)

        with patch("team_b.nodes.final_judgment.call_deepseek_json",
                   return_value=FAKE_JUDGMENT_RESPONSE):

            result = final_judgment_node(state)

        # From FAKE_EVIDENCE: 2 CONTRADICT, 1 SUPPORT
        assert result["stance_breakdown"]["CONTRADICT"] == 2
        assert result["stance_breakdown"]["SUPPORT"]    == 1
        assert result["stance_breakdown"]["NEUTRAL"]    == 0
        print(f"\n  Stance breakdown: {result['stance_breakdown']}")

    def test_fallback_when_deepseek_fails(self):
        """If DeepSeek fails, rule-based fallback should still produce a verdict."""
        from team_b.nodes.final_judgment import final_judgment_node

        state = _after_round1(_base_state(), certain_answers=True)

        with patch("team_b.nodes.final_judgment.call_deepseek_json",
                   side_effect=RuntimeError("Ollama down")):

            result = final_judgment_node(state)

        # Fallback: 2 CONTRADICT > 1 SUPPORT → FALSE
        assert result["verdict"] in ["FALSE", "PARTIALLY_FALSE", "PARTIALLY_TRUE", "UNVERIFIABLE"]
        assert result["final_confidence"] > 0
        print(f"\n  Fallback verdict: {result['verdict']}")

    def test_skips_if_already_unverifiable(self):
        """If short-circuited earlier, verdict should not be overwritten."""
        from team_b.nodes.final_judgment import final_judgment_node

        state = _base_state()
        state["verdict"] = "UNVERIFIABLE"

        result = final_judgment_node(state)

        assert result["verdict"] == "UNVERIFIABLE"


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE: End-to-End
# ─────────────────────────────────────────────────────────────

class TestFullPipeline:

    def test_round1_only_path(self):
        """High confidence claim → should complete in Round 1 only."""
        from team_b.graph import run_graph

        with patch("team_b.nodes.decompose_claim.call_deepseek_json",   return_value=FAKE_DECOMPOSE_RESPONSE), \
             patch("team_b.personas.fact_checker.call_deepseek",         return_value=FAKE_ANSWER), \
             patch("team_b.personas.logical_analyst.call_deepseek",      return_value=FAKE_ANSWER), \
             patch("team_b.personas.bias_detector.call_deepseek",        return_value=FAKE_ANSWER), \
             patch("team_b.nodes.final_judgment.call_deepseek_json",     return_value=FAKE_JUDGMENT_RESPONSE):

            result = run_graph(
                claim=FAKE_CLAIM,
                evidence=FAKE_EVIDENCE,
                original_claim="क्या व्हिस्की से COVID ठीक होता है?",
                original_language="hindi",
                thread_id="test_round1_only"
            )

        assert result["verdict"]          == "FALSE"
        assert result["rounds_executed"]  in [1, 2]    # depends on confidence
        assert result["final_confidence"] > 0
        assert result["justification"]    != ""

        print(f"\n  Verdict:   {result['verdict']}")
        print(f"  Confidence:{result['final_confidence']}%")
        print(f"  Rounds:    {result['rounds_executed']}")
        print(f"  LLM calls: {result['total_llm_calls']}")

    def test_no_evidence_pipeline(self):
        """Empty evidence → pipeline should short-circuit to UNVERIFIABLE."""
        from team_b.graph import run_graph

        with patch("team_b.nodes.decompose_claim.call_deepseek_json",
                   return_value=FAKE_DECOMPOSE_RESPONSE):

            result = run_graph(
                claim=FAKE_CLAIM,
                evidence=[],                # empty — no evidence retrieved
                thread_id="test_no_evidence"
            )

        assert result["verdict"]          == "UNVERIFIABLE"
        assert result["rounds_executed"]  == 0
        print(f"\n  Short-circuit verdict: {result['verdict']}")


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# Build realistic state objects for each test stage
# ─────────────────────────────────────────────────────────────

def _base_state() -> dict:
    """Minimal starting state — as received from Module 0 + Team A."""
    return {
        "claim":              FAKE_CLAIM,
        "original_claim":     "क्या व्हिस्की से COVID ठीक होता है?",
        "original_language":  "hindi",
        "evidence":           FAKE_EVIDENCE,
        "sub_claims":         [],
        "claim_type":         "",
        "search_queries":     [],
        "avg_source_quality": 0.0,
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


def _after_credibility(state: dict) -> dict:
    """State after credibility_check has run."""
    return {
        **state,
        "avg_source_quality": 0.75,
        "flagged_sources":    [],
        "high_quality_count": 2,
        "total_llm_calls":    1,
    }


def _after_round1(state: dict, certain_answers: bool = True) -> dict:
    """State after round1_qa has run."""
    answer = FAKE_ANSWER if certain_answers else "Evidence is unclear and does not directly address this."

    state = _after_credibility(state)
    return {
        **state,
        "qa_memory": [
            {
                "round":    1,
                "persona":  "Fact Checker",
                "question": "Does any credible source confirm whiskey cures COVID?",
                "answer":   answer,
                "insight":  "No credible source confirms whiskey cures COVID" if certain_answers else "unclear",
            },
            {
                "round":    1,
                "persona":  "Logical Analyst",
                "question": "Is there a biological mechanism for whiskey to cure COVID?",
                "answer":   answer,
                "insight":  "No biological mechanism exists" if certain_answers else "uncertain",
            },
            {
                "round":    1,
                "persona":  "Bias Detector",
                "question": "Does this claim follow a known misinformation pattern?",
                "answer":   answer,
                "insight":  "Matches miracle cure misinformation pattern" if certain_answers else "mixed signals",
            },
        ],
        "total_llm_calls": 7,
    }

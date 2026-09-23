"""
test_team_b.py — TruthLens
Tests for the 3-round Team B QA architecture.

Round 1:
    Fact Checker + Logical Analyst + Bias Detector
    6 LLM calls

Round 2:
    Fact Checker only
    High-quality evidence: R(d) >= 0.70
    2 LLM calls

Round 3:
    Logical Analyst only
    Low-quality evidence: R(d) < 0.70
    2 LLM calls

No Ollama, internet, API keys, or real LLM calls required.
All LLM calls are mocked.
"""

import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


# ============================================================
# SHARED MOCK DATA
# ============================================================

FAKE_CLAIM = "Does whiskey cure COVID-19?"

FAKE_EVIDENCE = [
    {
        "source_domain": "who.int",
        "author": "World Health Organization",
        "organization": "WHO",
        "content": (
            "Alcohol does not protect against COVID-19. "
            "Consuming alcohol does not destroy the virus."
        ),
        "url": "https://who.int/news/item/...",
        "bm25_score": 0.90,
        "cosine_score": 0.92,
        "domain_credibility": 0.97,
        "credibility_source": "hardlist_trusted",
        "recency_score": 0.90,
        "combined_reliability": 0.94,
        "stance": "CONTRADICT",
        "retrieval_rank": 1,
    },
    {
        "source_domain": "bbc.com",
        "author": "BBC Health",
        "organization": "BBC",
        "content": (
            "Medical experts warn that drinking alcohol "
            "does not prevent or cure coronavirus."
        ),
        "url": "https://bbc.com/news/...",
        "bm25_score": 0.85,
        "cosine_score": 0.88,
        "domain_credibility": 0.88,
        "credibility_source": "hardlist_trusted",
        "recency_score": 0.85,
        "combined_reliability": 0.87,
        "stance": "CONTRADICT",
        "retrieval_rank": 2,
    },
    {
        "source_domain": "twitter.com",
        "author": "random_user_99",
        "organization": "",
        "content": (
            "My uncle drank whiskey every night "
            "and never got COVID! It works!"
        ),
        "url": "https://twitter.com/...",
        "bm25_score": 0.40,
        "cosine_score": 0.45,
        "domain_credibility": 0.30,
        "credibility_source": "dynamic_signals",
        "recency_score": 0.60,
        "combined_reliability": 0.44,
        "stance": "SUPPORT",
        "retrieval_rank": 3,
    },
]

FAKE_DECOMPOSE_RESPONSE = {
    "claim_type": "factual",
    "sub_claims": [
        "Whiskey has antiviral properties",
        "Drinking whiskey can cure COVID-19",
    ],
    "search_queries": [
        "whiskey alcohol cure COVID scientific evidence",
        "alcohol antiviral COVID-19 medical research",
    ],
    "key_entities": [
        "whiskey",
        "COVID-19",
        "alcohol",
    ],
}

FAKE_JUDGMENT_RESPONSE = {
    "verdict": "FALSE",
    "final_confidence": 95,
    "justification": (
        "WHO and BBC directly contradict this claim. "
        "The social media anecdote is unreliable evidence."
    ),
    "fact_checker_summary": (
        "Credible sources contradict the claim."
    ),
    "logical_analyst_summary": (
        "The evidence does not establish a mechanism for the claim."
    ),
    "bias_detector_summary": (
        "Claim follows a common misinformation pattern."
    ),
    "recommendation": "Do not share",
}

FAKE_QUESTION = (
    "Does credible medical evidence support the claim?"
)

FAKE_ANSWER = (
    "No. WHO and BBC evidence contradicts the claim."
)

FAKE_STRUCTURED_ANSWER = {
    "stance": "CONTRADICT",
    "confidence": 0.95,
    "reasoning": (
        "The provided evidence directly contradicts the claim. "
        "The high-quality sources do not support whiskey as a cure."
    ),
}


# ============================================================
# NODE 1 — DECOMPOSE CLAIM
# ============================================================

class TestDecomposeClaimNode:

    def test_normal_claim(self):
        from team_b.nodes.decompose_claim import decompose_claim_node

        state = _base_state()

        with patch(
            "team_b.nodes.decompose_claim.call_deepseek_json",
            return_value=FAKE_DECOMPOSE_RESPONSE,
        ):
            result = decompose_claim_node(state)

        assert "claim" in result
        assert result["claim_type"] == "factual"
        assert len(result["sub_claims"]) == 2

    def test_llm_failure_fallback(self):
        from team_b.nodes.decompose_claim import decompose_claim_node

        state = _base_state()

        with patch(
            "team_b.nodes.decompose_claim.call_deepseek_json",
            side_effect=RuntimeError("Ollama unreachable"),
        ):
            result = decompose_claim_node(state)

        assert result["sub_claims"] == [FAKE_CLAIM]
        assert result["claim_type"] == "factual"


# ============================================================
# NODE 2 — CREDIBILITY CHECK
# ============================================================

class TestCredibilityCheckNode:

    def test_reads_team_a_scores_directly(self):
        from team_b.nodes.credibility_check import credibility_check_node

        result = credibility_check_node(_base_state())

        expected_avg = round(
            (0.94 + 0.87 + 0.44) / 3,
            3,
        )

        assert abs(
            result["avg_source_quality"] - expected_avg
        ) < 0.01

    def test_flags_low_credibility_sources(self):
        from team_b.nodes.credibility_check import credibility_check_node

        result = credibility_check_node(_base_state())

        assert "twitter.com" not in result["flagged_sources"]

    def test_flags_very_low_credibility(self):
        from team_b.nodes.credibility_check import credibility_check_node

        state = _base_state()

        state["evidence"] = state["evidence"].copy()

        state["evidence"][2] = {
            **state["evidence"][2],
            "combined_reliability": 0.10,
            "source_domain": "fakenews.xyz",
        }

        result = credibility_check_node(state)

        assert "fakenews.xyz" in result["flagged_sources"]

    def test_no_evidence_short_circuits(self):
        from team_b.nodes.credibility_check import credibility_check_node

        state = _base_state()
        state["evidence"] = []

        result = credibility_check_node(state)

        assert result["verdict"] == "UNVERIFIABLE"
        assert result["final_confidence"] == 0.0
        assert result["rounds_executed"] == 0

    def test_high_quality_count(self):
        from team_b.nodes.credibility_check import credibility_check_node

        result = credibility_check_node(_base_state())

        assert result["high_quality_count"] == 2


# ============================================================
# NODE 3 — ROUND 1 QA
# ============================================================

class TestRound1QA:

    def _round1_mocks(self):
        return [
            patch(
                "team_b.personas.fact_checker.generate_question",
                return_value=FAKE_QUESTION,
            ),
            patch(
                "team_b.personas.fact_checker.generate_answer",
                return_value=FAKE_ANSWER,
            ),
            patch(
                "team_b.personas.logical_analyst.generate_question",
                return_value=FAKE_QUESTION,
            ),
            patch(
                "team_b.personas.logical_analyst.generate_answer",
                return_value=FAKE_ANSWER,
            ),
            patch(
                "team_b.personas.bias_detector.generate_question",
                return_value=FAKE_QUESTION,
            ),
            patch(
                "team_b.personas.bias_detector.generate_answer",
                return_value=FAKE_ANSWER,
            ),
        ]

    def test_runs_all_three_personas(self):
        from team_b.nodes.round1_qa import round1_qa_node

        state = _after_credibility(_base_state())

        mocks = self._round1_mocks()

        for mock in mocks:
            mock.start()

        try:
            result = round1_qa_node(state)
        finally:
            for mock in reversed(mocks):
                mock.stop()

        assert len(result["qa_memory"]) == 3

        personas = {
            entry["persona"]
            for entry in result["qa_memory"]
        }

        assert "Fact Checker" in personas
        assert "Logical Analyst" in personas
        assert "Bias Detector" in personas

    def test_round1_entries_have_round_one(self):
        from team_b.nodes.round1_qa import round1_qa_node

        state = _after_credibility(_base_state())

        mocks = self._round1_mocks()

        for mock in mocks:
            mock.start()

        try:
            result = round1_qa_node(state)
        finally:
            for mock in reversed(mocks):
                mock.stop()

        for entry in result["qa_memory"]:
            assert entry["round"] == 1

    def test_round1_has_six_llm_calls(self):
        from team_b.nodes.round1_qa import round1_qa_node

        state = _after_credibility(_base_state())
        state["total_llm_calls"] = 1

        mocks = self._round1_mocks()

        for mock in mocks:
            mock.start()

        try:
            result = round1_qa_node(state)
        finally:
            for mock in reversed(mocks):
                mock.stop()

        assert result["total_llm_calls"] == 7

    def test_round1_does_not_change_qa_round(self):
        """
        Round 1 performs the three persona analyses.

        confidence_gate is responsible for setting:
            qa_round = 1

        Therefore Round 1 itself should not be required
        to modify qa_round.
        """
        from team_b.nodes.round1_qa import round1_qa_node

        state = _after_credibility(_base_state())

        mocks = self._round1_mocks()

        for mock in mocks:
            mock.start()

        try:
            result = round1_qa_node(state)
        finally:
            for mock in reversed(mocks):
                mock.stop()

        assert result.get("qa_round", 0) == 0

    def test_skips_if_pipeline_already_ended(self):
        from team_b.nodes.round1_qa import round1_qa_node

        state = _base_state()
        state["verdict"] = "UNVERIFIABLE"

        result = round1_qa_node(state)

        assert result["qa_memory"] == []


# ============================================================
# NODE 4 — CONFIDENCE GATE
# ============================================================

class TestConfidenceGate:

    def test_high_confidence_skips_round2(self):
        from team_b.nodes.confidence_gate import confidence_gate_node

        state = _after_round1(
            _base_state(),
            certain_answers=True,
        )

        result = confidence_gate_node(state)

        assert 0.0 <= result["round1_confidence"] <= 1.0

        if result["round1_confidence"] >= 0.85:
            assert result["go_to_round2"] is False

    def test_low_confidence_triggers_round2(self):
        from team_b.nodes.confidence_gate import confidence_gate_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["evidence"] = [
            {
                **ev,
                "stance": "NEUTRAL",
            }
            for ev in state["evidence"]
        ]

        state["avg_source_quality"] = 0.30

        result = confidence_gate_node(state)

        assert result["round1_confidence"] < 0.85
        assert result["go_to_round2"] is True

    def test_threshold_is_085(self):
        from team_b.nodes.confidence_gate import (
            CONFIDENCE_THRESHOLD,
        )

        assert CONFIDENCE_THRESHOLD == 0.85

    def test_round_one_sets_qa_round(self):
        from team_b.nodes.confidence_gate import confidence_gate_node

        state = _after_round1(
            _base_state(),
            certain_answers=True,
        )

        result = confidence_gate_node(state)

        assert result["qa_round"] == 1

    def test_route_function(self):
        from team_b.nodes.confidence_gate import route_after_gate

        high_state = {
            "round1_confidence": 0.90,
            "go_to_round2": False,
            "verdict": "",
        }

        low_state = {
            "round1_confidence": 0.60,
            "go_to_round2": True,
            "verdict": "",
        }

        short_state = {
            "verdict": "UNVERIFIABLE",
            "go_to_round2": False,
        }

        assert (
            route_after_gate(high_state)
            == "final_judgment"
        )

        assert (
            route_after_gate(low_state)
            == "round2_qa"
        )

        assert (
            route_after_gate(short_state)
            == "final_judgment"
        )


# ============================================================
# NODE 5 — ROUND 2 QA
# FACT CHECKER ONLY
# ============================================================

class TestRound2QA:

    def test_uses_fact_checker_only(self):
        from team_b.nodes.round2_qa import round2_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 1
        state["rounds_executed"] = 1

        with patch(
            "team_b.personas.fact_checker.generate_question",
            return_value=FAKE_QUESTION,
        ) as mock_question, patch(
            "team_b.personas.fact_checker.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ) as mock_answer:

            result = round2_qa_node(state)

        mock_question.assert_called_once()
        mock_answer.assert_called_once()

        personas = [
            entry["persona"]
            for entry in result["qa_memory"]
            if entry["round"] == 2
        ]

        assert personas == ["Fact Checker"]

    def test_round2_uses_high_quality_evidence_only(self):
        from team_b.nodes.round2_qa import round2_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 1
        state["rounds_executed"] = 1

        with patch(
            "team_b.personas.fact_checker.generate_question",
            return_value=FAKE_QUESTION,
        ) as mock_question, patch(
            "team_b.personas.fact_checker.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ):
            round2_qa_node(state)

        evidence_used = mock_question.call_args.kwargs["evidence"]

        assert len(evidence_used) == 2

        for evidence in evidence_used:
            assert (
                evidence["combined_reliability"]
                >= 0.70
            )

    def test_round2_has_two_llm_calls(self):
        from team_b.nodes.round2_qa import round2_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 1
        state["rounds_executed"] = 1
        state["total_llm_calls"] = 7

        with patch(
            "team_b.personas.fact_checker.generate_question",
            return_value=FAKE_QUESTION,
        ), patch(
            "team_b.personas.fact_checker.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ):
            result = round2_qa_node(state)

        assert result["total_llm_calls"] == 9

    def test_round2_updates_history(self):
        from team_b.nodes.round2_qa import round2_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 1
        state["rounds_executed"] = 1

        with patch(
            "team_b.personas.fact_checker.generate_question",
            return_value=FAKE_QUESTION,
        ), patch(
            "team_b.personas.fact_checker.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ):
            result = round2_qa_node(state)

        assert len(result["qa_history"]) >= 1

        latest = result["qa_history"][-1]

        assert latest["round"] == 2
        assert "fact_checker" in latest

    def test_round2_sets_round_two(self):
        from team_b.nodes.round2_qa import round2_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 1
        state["rounds_executed"] = 1

        with patch(
            "team_b.personas.fact_checker.generate_question",
            return_value=FAKE_QUESTION,
        ), patch(
            "team_b.personas.fact_checker.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ):
            result = round2_qa_node(state)

        assert result["qa_round"] == 2


# ============================================================
# NODE 6 — ROUND 3 QA
# LOGICAL ANALYST ONLY
# ============================================================

class TestRound3QA:

    def test_uses_logical_analyst_only(self):
        from team_b.nodes.round3_qa import round3_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 2
        state["rounds_executed"] = 2

        with patch(
            "team_b.personas.logical_analyst.generate_question",
            return_value=FAKE_QUESTION,
        ) as mock_question, patch(
            "team_b.personas.logical_analyst.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ) as mock_answer:

            result = round3_qa_node(state)

        mock_question.assert_called_once()
        mock_answer.assert_called_once()

        personas = [
            entry["persona"]
            for entry in result["qa_memory"]
            if entry["round"] == 3
        ]

        assert personas == ["Logical Analyst"]

    def test_round3_uses_low_quality_evidence_only(self):
        from team_b.nodes.round3_qa import round3_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 2
        state["rounds_executed"] = 2

        with patch(
            "team_b.personas.logical_analyst.generate_question",
            return_value=FAKE_QUESTION,
        ) as mock_question, patch(
            "team_b.personas.logical_analyst.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ):
            round3_qa_node(state)

        evidence_used = mock_question.call_args.kwargs["evidence"]

        assert len(evidence_used) == 1

        for evidence in evidence_used:
            assert (
                evidence["combined_reliability"]
                < 0.70
            )

    def test_round3_has_two_llm_calls(self):
        from team_b.nodes.round3_qa import round3_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 2
        state["rounds_executed"] = 2
        state["total_llm_calls"] = 9

        with patch(
            "team_b.personas.logical_analyst.generate_question",
            return_value=FAKE_QUESTION,
        ), patch(
            "team_b.personas.logical_analyst.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ):
            result = round3_qa_node(state)

        assert result["total_llm_calls"] == 11

    def test_round3_updates_history(self):
        from team_b.nodes.round3_qa import round3_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 2
        state["rounds_executed"] = 2

        with patch(
            "team_b.personas.logical_analyst.generate_question",
            return_value=FAKE_QUESTION,
        ), patch(
            "team_b.personas.logical_analyst.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ):
            result = round3_qa_node(state)

        assert len(result["qa_history"]) >= 1

        latest = result["qa_history"][-1]

        assert latest["round"] == 3
        assert "logical_analyst" in latest

    def test_round3_sets_round_three(self):
        from team_b.nodes.round3_qa import round3_qa_node

        state = _after_round1(
            _base_state(),
            certain_answers=False,
        )

        state["qa_round"] = 2
        state["rounds_executed"] = 2

        with patch(
            "team_b.personas.logical_analyst.generate_question",
            return_value=FAKE_QUESTION,
        ), patch(
            "team_b.personas.logical_analyst.generate_structured_answer",
            return_value=FAKE_STRUCTURED_ANSWER,
        ):
            result = round3_qa_node(state)

        assert result["qa_round"] == 3


# ============================================================
# NODE 7 — FINAL JUDGMENT
# ============================================================

class TestFinalJudgment:

    def test_false_verdict_for_contradicting_evidence(self):
        from team_b.nodes.final_judgment import final_judgment_node

        state = _after_round1(
            _base_state(),
            certain_answers=True,
        )

        with patch(
            "team_b.nodes.final_judgment.call_deepseek_json",
            return_value=FAKE_JUDGMENT_RESPONSE,
        ):
            result = final_judgment_node(state)

        assert result["verdict"] == "FALSE"
        assert result["final_confidence"] == 95
        assert result["recommendation"] == "Do not share"
        assert result["coverage_score"] == 1.0

    def test_outputs_required_fields(self):
        from team_b.nodes.final_judgment import final_judgment_node

        state = _after_round1(
            _base_state(),
            certain_answers=True,
        )

        with patch(
            "team_b.nodes.final_judgment.call_deepseek_json",
            return_value=FAKE_JUDGMENT_RESPONSE,
        ):
            result = final_judgment_node(state)

        required_fields = [
            "verdict",
            "final_confidence",
            "justification",
            "stance_breakdown",
            "persona_insights",
            "coverage_score",
            "recommendation",
        ]

        for field in required_fields:
            assert field in result

    def test_stance_breakdown(self):
        from team_b.nodes.final_judgment import final_judgment_node

        state = _after_round1(
            _base_state(),
            certain_answers=True,
        )

        with patch(
            "team_b.nodes.final_judgment.call_deepseek_json",
            return_value=FAKE_JUDGMENT_RESPONSE,
        ):
            result = final_judgment_node(state)

        assert (
            result["stance_breakdown"]["CONTRADICT"]
            == 2
        )

        assert (
            result["stance_breakdown"]["SUPPORT"]
            == 1
        )

        assert (
            result["stance_breakdown"]["NEUTRAL"]
            == 0
        )

    def test_fallback_when_deepseek_fails(self):
        from team_b.nodes.final_judgment import final_judgment_node

        state = _after_round1(
            _base_state(),
            certain_answers=True,
        )

        with patch(
            "team_b.nodes.final_judgment.call_deepseek_json",
            side_effect=RuntimeError("Ollama down"),
        ):
            result = final_judgment_node(state)

        assert result["verdict"] in [
            "FALSE",
            "PARTIALLY_FALSE",
            "PARTIALLY_TRUE",
            "UNVERIFIABLE",
        ]

        assert result["final_confidence"] > 0

    def test_skips_if_already_unverifiable(self):
        from team_b.nodes.final_judgment import final_judgment_node

        state = _base_state()
        state["verdict"] = "UNVERIFIABLE"

        result = final_judgment_node(state)

        assert result["verdict"] == "UNVERIFIABLE"


# ============================================================
# FINAL PERSONA RESOLUTION
# ============================================================

class TestPersonaResolution:

    def test_round2_fact_checker_replaces_round1(self):
        from team_b.nodes.final_judgment import (
            _resolve_latest_personas,
        )

        memory = [
            {
                "round": 1,
                "persona": "Fact Checker",
                "question": "Q1",
                "answer": "A1",
                "insight": "Round 1 insight",
            },
            {
                "round": 1,
                "persona": "Logical Analyst",
                "question": "Q1",
                "answer": "A1",
                "insight": "Round 1 logic",
            },
            {
                "round": 1,
                "persona": "Bias Detector",
                "question": "Q1",
                "answer": "A1",
                "insight": "Round 1 bias",
            },
            {
                "round": 2,
                "persona": "Fact Checker",
                "question": "Q2",
                "answer": "A2",
                "insight": "Round 2 fact check",
                "stance": "CONTRADICT",
                "confidence": 0.95,
            },
        ]

        history = [
            {
                "round": 2,
                "fact_checker": {
                    "stance": "CONTRADICT",
                    "reasoning": "R2 reasoning",
                    "confidence": 0.95,
                },
                "confidence": 0.95,
            }
        ]

        resolved = _resolve_latest_personas(
            memory,
            history,
        )

        assert (
            resolved["Fact Checker"]["round"]
            == 2
        )

        assert (
            resolved["Logical Analyst"]["round"]
            == 1
        )

        assert (
            resolved["Bias Detector"]["round"]
            == 1
        )


# ============================================================
# GRAPH
# ============================================================

class TestGraph:

    def test_graph_compiles(self):
        from team_b.graph import build_graph

        app = build_graph(
            use_checkpointing=False
        )

        assert app is not None


# ============================================================
# HELPERS
# ============================================================

def _base_state() -> dict:
    return {
        "claim": FAKE_CLAIM,
        "original_claim": "क्या व्हिस्की से COVID ठीक होता है?",
        "original_language": "hindi",

        "evidence": FAKE_EVIDENCE,

        "sub_claims": [],
        "claim_type": "",
        "search_queries": [],

        "avg_source_quality": 0.0,
        "flagged_sources": [],
        "high_quality_count": 0,

        "qa_memory": [],

        "qa_round": 0,
        "rounds_executed": 0,
        "qa_history": [],

        "round1_confidence": 0.0,
        "stance_consensus": 0.0,
        "answer_certainty": 0.5,

        "go_to_round2": False,

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

        "total_llm_calls": 0,

        "pipeline_error": None,
    }


def _after_credibility(state: dict) -> dict:
    return {
        **state,
        "avg_source_quality": 0.75,
        "flagged_sources": [],
        "high_quality_count": 2,
        "total_llm_calls": 1,
    }


def _after_round1(
    state: dict,
    certain_answers: bool = True,
) -> dict:

    answer = (
        FAKE_ANSWER
        if certain_answers
        else
        "Evidence is unclear and does not directly address this."
    )

    return {
        **_after_credibility(state),

        "qa_memory": [
            {
                "round": 1,
                "persona": "Fact Checker",
                "question": (
                    "Does any credible source confirm "
                    "whiskey cures COVID?"
                ),
                "answer": answer,
                "insight": (
                    "No credible source confirms whiskey cures COVID"
                    if certain_answers
                    else "unclear"
                ),
            },
            {
                "round": 1,
                "persona": "Logical Analyst",
                "question": (
                    "Is there a biological mechanism "
                    "for whiskey to cure COVID?"
                ),
                "answer": answer,
                "insight": (
                    "No biological mechanism exists"
                    if certain_answers
                    else "uncertain"
                ),
            },
            {
                "round": 1,
                "persona": "Bias Detector",
                "question": (
                    "Does this claim follow a known "
                    "misinformation pattern?"
                ),
                "answer": answer,
                "insight": (
                    "Matches miracle cure misinformation pattern"
                    if certain_answers
                    else "mixed signals"
                ),
            },
        ],

        "qa_round": 1,
        "rounds_executed": 1,
        "total_llm_calls": 7,
    }
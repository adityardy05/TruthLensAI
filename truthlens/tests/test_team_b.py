import sys, os, pytest
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

FAKE_CLAIM = "Does whiskey cure COVID-19?"
FAKE_EVIDENCE = [
    {"source_domain": "who.int", "author": "WHO", "organization": "WHO",
     "content": "Alcohol does not protect against COVID-19.", "url": "https://who.int/",
     "bm25_score": 0.90, "cosine_score": 0.92, "domain_credibility": 0.97,
     "credibility_source": "hardlist_trusted", "recency_score": 0.90,
     "combined_reliability": 0.94, "stance": "CONTRADICT", "retrieval_rank": 1},
    {"source_domain": "bbc.com", "author": "BBC Health", "organization": "BBC",
     "content": "Drinking alcohol does not cure coronavirus.", "url": "https://bbc.com/",
     "bm25_score": 0.85, "cosine_score": 0.88, "domain_credibility": 0.88,
     "credibility_source": "hardlist_trusted", "recency_score": 0.85,
     "combined_reliability": 0.87, "stance": "CONTRADICT", "retrieval_rank": 2},
    {"source_domain": "twitter.com", "author": "random_user", "organization": "",
     "content": "My uncle drank whiskey and never got COVID!", "url": "https://twitter.com/",
     "bm25_score": 0.40, "cosine_score": 0.45, "domain_credibility": 0.30,
     "credibility_source": "dynamic_signals", "recency_score": 0.60,
     "combined_reliability": 0.44, "stance": "SUPPORT", "retrieval_rank": 3},
]
FAKE_DECOMPOSE = {"claim_type": "factual",
    "sub_claims": ["Whiskey has antiviral properties", "Whiskey cures COVID-19"],
    "search_queries": ["whiskey cure COVID scientific evidence"], "key_entities": ["whiskey"]}
FAKE_JUDGMENT = {"verdict": "FALSE", "final_confidence": 95,
    "justification": "WHO and BBC contradict this claim.",
    "fact_checker_summary": "No credible sources support.",
    "logical_analyst_summary": "No biological mechanism.",
    "bias_detector_summary": "Matches miracle-cure pattern.",
    "recommendation": "Do not share"}
FAKE_Q = "Does any credible source confirm whiskey cures COVID?"
FAKE_A = "No. WHO states alcohol does not protect against COVID-19."


# ── NODE 1 ────────────────────────────────────────────────────
class TestDecomposeClaimNode:
    def test_normal_claim(self):
        from team_b.nodes.decompose_claim import decompose_claim_node
        with patch("team_b.nodes.decompose_claim.call_deepseek_json", return_value=FAKE_DECOMPOSE):
            r = decompose_claim_node(_base())
        assert r["claim_type"] == "factual"
        assert len(r["sub_claims"]) == 2
        assert r["total_llm_calls"] == 1
        print(f"\n  sub_claims: {r['sub_claims']}")

    def test_llm_failure_fallback(self):
        from team_b.nodes.decompose_claim import decompose_claim_node
        with patch("team_b.nodes.decompose_claim.call_deepseek_json", side_effect=RuntimeError("down")):
            r = decompose_claim_node(_base())
        assert r["sub_claims"] == [FAKE_CLAIM]
        print(f"\n  Fallback: {r['sub_claims']}")


# ── NODE 2 ────────────────────────────────────────────────────
class TestCredibilityCheckNode:
    def test_reads_team_a_scores(self):
        from team_b.nodes.credibility_check import credibility_check_node
        r = credibility_check_node(_base())
        expected = round((0.94 + 0.87 + 0.44) / 3, 3)
        assert abs(r["avg_source_quality"] - expected) < 0.01
        print(f"\n  avg_quality: {r['avg_source_quality']}")

    def test_does_not_flag_medium(self):
        from team_b.nodes.credibility_check import credibility_check_node
        r = credibility_check_node(_base())
        assert "twitter.com" not in r["flagged_sources"]
        print(f"\n  flagged: {r['flagged_sources']}")

    def test_flags_very_low(self):
        from team_b.nodes.credibility_check import credibility_check_node
        s = _base()
        s["evidence"][2] = {**s["evidence"][2], "combined_reliability": 0.10, "source_domain": "fake.xyz"}
        r = credibility_check_node(s)
        assert "fake.xyz" in r["flagged_sources"]
        print(f"\n  flagged: {r['flagged_sources']}")

    def test_no_evidence_short_circuits(self):
        from team_b.nodes.credibility_check import credibility_check_node
        s = _base(); s["evidence"] = []
        r = credibility_check_node(s)
        assert r["verdict"] == "UNVERIFIABLE"
        assert r["rounds_executed"] == 0
        print(f"\n  Short-circuit: {r['verdict']}")

    def test_high_quality_count(self):
        from team_b.nodes.credibility_check import credibility_check_node
        r = credibility_check_node(_base())
        assert r["high_quality_count"] == 2
        print(f"\n  high_quality_count: {r['high_quality_count']}")


# ── NODE 3 ────────────────────────────────────────────────────
class TestRound1QA:
    def test_runs_all_3_personas(self):
        from team_b.nodes.round1_qa import round1_qa_node
        with patch("shared.ollama_client.call_deepseek", return_value=FAKE_Q):
            r = round1_qa_node(_cred(_base()))
        assert len(r["qa_memory"]) == 3
        personas = {e["persona"] for e in r["qa_memory"]}
        assert {"Fact Checker", "Logical Analyst", "Bias Detector"} == personas
        print(f"\n  personas: {personas}")

    def test_memory_round_is_1(self):
        from team_b.nodes.round1_qa import round1_qa_node
        with patch("shared.ollama_client.call_deepseek", return_value=FAKE_Q):
            r = round1_qa_node(_cred(_base()))
        assert all(e["round"] == 1 for e in r["qa_memory"])
        print(f"\n  All entries round=1 ✓")

    def test_llm_calls_count(self):
        from team_b.nodes.round1_qa import round1_qa_node
        s = _cred(_base()); s["total_llm_calls"] = 1
        with patch("shared.ollama_client.call_deepseek", return_value=FAKE_Q):
            r = round1_qa_node(s)
        assert r["total_llm_calls"] == 7
        print(f"\n  LLM calls: {r['total_llm_calls']}")

    def test_skips_if_verdict_set(self):
        from team_b.nodes.round1_qa import round1_qa_node
        s = _base(); s["verdict"] = "UNVERIFIABLE"
        r = round1_qa_node(s)
        assert r["qa_memory"] == []
        print(f"\n  Skipped correctly ✓")


# ── NODE 4 ────────────────────────────────────────────────────
class TestConfidenceGate:
    def test_758_is_correct_triggers_round2(self):
        """2/3 CONTRADICT → stance=0.667 → conf=0.758 < 0.80 → Round 2. Correct."""
        from team_b.nodes.confidence_gate import confidence_gate_node
        r = confidence_gate_node(_r1(_base(), certain=True))
        print(f"\n  confidence={r['round1_confidence']} go_round2={r['go_to_round2']}")
        assert 0.70 <= r["round1_confidence"] <= 0.85
        assert r["go_to_round2"] == True

    def test_low_conf_triggers_round2(self):
        from team_b.nodes.confidence_gate import confidence_gate_node
        s = _r1(_base(), certain=False)
        s["evidence"] = [{**e, "stance": "NEUTRAL"} for e in s["evidence"]]
        s["avg_source_quality"] = 0.30
        r = confidence_gate_node(s)
        assert r["round1_confidence"] < 0.80
        assert r["go_to_round2"] == True
        print(f"\n  Low conf: {r['round1_confidence']:.3f} ✓")

    def test_high_conf_skips_round2(self):
        from team_b.nodes.confidence_gate import confidence_gate_node
        s = _r1(_base(), certain=True)
        s["evidence"] = [{"source_domain": f"gov{i}.gov", "combined_reliability": 0.95, "stance": "CONTRADICT"} for i in range(5)]
        s["avg_source_quality"] = 0.95
        r = confidence_gate_node(s)
        assert r["round1_confidence"] >= 0.80
        assert r["go_to_round2"] == False
        print(f"\n  High conf: {r['round1_confidence']:.3f} → skip ✓")

    def test_route_function(self):
        from team_b.nodes.confidence_gate import route_after_gate
        assert route_after_gate({"round1_confidence": 0.90, "go_to_round2": False, "verdict": ""}) == "final_judgment"
        assert route_after_gate({"round1_confidence": 0.60, "go_to_round2": True,  "verdict": ""}) == "round2_qa"
        assert route_after_gate({"verdict": "UNVERIFIABLE", "go_to_round2": False})                == "final_judgment"
        print("\n  All routes correct ✓")

    def test_formula_components_in_range(self):
        from team_b.nodes.confidence_gate import confidence_gate_node
        r = confidence_gate_node(_r1(_base(), certain=True))
        assert 0.0 <= r["stance_consensus"]  <= 1.0
        assert 0.0 <= r["answer_certainty"]  <= 1.0
        assert 0.0 <= r["round1_confidence"] <= 1.0
        print(f"\n  stance={r['stance_consensus']:.3f} certainty={r['answer_certainty']:.3f} conf={r['round1_confidence']:.3f}")


# ── NODE 5 ────────────────────────────────────────────────────
class TestRound2QA:
    def test_appends_to_memory(self):
        from team_b.nodes.round2_qa import round2_qa_node
        s = _r1(_base(), certain=True); before = len(s["qa_memory"])
        with patch("shared.ollama_client.call_deepseek", return_value=FAKE_Q):
            r = round2_qa_node(s)
        assert len(r["qa_memory"]) == before + 3
        assert r["rounds_executed"] == 2
        print(f"\n  {before} + 3 = {len(r['qa_memory'])} ✓")

    def test_round2_entries_labelled(self):
        from team_b.nodes.round2_qa import round2_qa_node
        s = _r1(_base(), certain=True)
        with patch("shared.ollama_client.call_deepseek", return_value=FAKE_Q):
            r = round2_qa_node(s)
        assert len([e for e in r["qa_memory"] if e["round"] == 2]) == 3
        print(f"\n  Round 2 entries = 3 ✓")

    def test_total_calls_13(self):
        from team_b.nodes.round2_qa import round2_qa_node
        s = _r1(_base(), certain=True); s["total_llm_calls"] = 7
        with patch("shared.ollama_client.call_deepseek", return_value=FAKE_Q):
            r = round2_qa_node(s)
        assert r["total_llm_calls"] == 13
        print(f"\n  Total calls: {r['total_llm_calls']} ✓")


# ── NODE 6 ────────────────────────────────────────────────────
class TestFinalJudgment:
    def test_false_verdict(self):
        from team_b.nodes.final_judgment import final_judgment_node
        with patch("team_b.nodes.final_judgment.call_deepseek_json", return_value=FAKE_JUDGMENT):
            r = final_judgment_node(_r1(_base(), certain=True))
        assert r["verdict"] == "FALSE"
        assert r["final_confidence"] == 95
        assert r["recommendation"] == "Do not share"
        assert r["coverage_score"] == 1.0
        print(f"\n  {r['verdict']} ({r['final_confidence']}%)")

    def test_all_fields_present(self):
        from team_b.nodes.final_judgment import final_judgment_node
        with patch("team_b.nodes.final_judgment.call_deepseek_json", return_value=FAKE_JUDGMENT):
            r = final_judgment_node(_r1(_base(), certain=True))
        for f in ["verdict","final_confidence","justification","stance_breakdown","persona_insights","coverage_score","recommendation"]:
            assert f in r; print(f"  ✓ {f}")

    def test_stance_breakdown(self):
        from team_b.nodes.final_judgment import final_judgment_node
        with patch("team_b.nodes.final_judgment.call_deepseek_json", return_value=FAKE_JUDGMENT):
            r = final_judgment_node(_r1(_base(), certain=True))
        assert r["stance_breakdown"]["CONTRADICT"] == 2
        assert r["stance_breakdown"]["SUPPORT"]    == 1
        assert r["stance_breakdown"]["NEUTRAL"]    == 0
        print(f"\n  {r['stance_breakdown']}")

    def test_fallback_on_failure(self):
        from team_b.nodes.final_judgment import final_judgment_node
        with patch("team_b.nodes.final_judgment.call_deepseek_json", side_effect=RuntimeError("down")):
            r = final_judgment_node(_r1(_base(), certain=True))
        assert r["verdict"] in ["FALSE","PARTIALLY_FALSE","PARTIALLY_TRUE","UNVERIFIABLE"]
        assert r["final_confidence"] > 0
        print(f"\n  Fallback: {r['verdict']}")

    def test_preserves_unverifiable(self):
        from team_b.nodes.final_judgment import final_judgment_node
        s = _base(); s["verdict"] = "UNVERIFIABLE"
        r = final_judgment_node(s)
        assert r["verdict"] == "UNVERIFIABLE"
        print(f"\n  Preserved ✓")


# ── FULL PIPELINE ─────────────────────────────────────────────
def _has_langgraph():
    try:
        from langgraph.graph import StateGraph
        from team_b.graph import run_graph
        return True
    except Exception:
        return False
skip_lg = pytest.mark.skipif(
    not _has_langgraph(),
    reason="langgraph not compatible — skipped"
)

class TestFullPipeline:
    @skip_lg
    def test_full_run(self):
        from team_b.graph import run_graph
        with patch("team_b.nodes.decompose_claim.call_deepseek_json", return_value=FAKE_DECOMPOSE), \
             patch("shared.ollama_client.call_deepseek", return_value=FAKE_A), \
             patch("team_b.nodes.final_judgment.call_deepseek_json", return_value=FAKE_JUDGMENT):
            r = run_graph(claim=FAKE_CLAIM, evidence=FAKE_EVIDENCE, thread_id="e2e")
        assert r["verdict"] == "FALSE"
        print(f"\n  {r['verdict']} rounds={r['rounds_executed']}")

    @skip_lg
    def test_no_evidence(self):
        from team_b.graph import run_graph
        with patch("team_b.nodes.decompose_claim.call_deepseek_json", return_value=FAKE_DECOMPOSE):
            r = run_graph(claim=FAKE_CLAIM, evidence=[], thread_id="no_ev")
        assert r["verdict"] == "UNVERIFIABLE"
        print(f"\n  {r['verdict']} ✓")

# ── HELPERS ───────────────────────────────────────────────────
def _base():
    return {
        "claim": FAKE_CLAIM, "original_claim": "original", "original_language": "hindi",
        "evidence": FAKE_EVIDENCE, "sub_claims": [], "claim_type": "", "search_queries": [],
        "avg_source_quality": 0.0, "flagged_sources": [], "high_quality_count": 0,
        "qa_memory": [], "round1_confidence": 0.0, "stance_consensus": 0.0,
        "answer_certainty": 0.5, "go_to_round2": False, "rounds_executed": 0,
        "verdict": "", "final_confidence": 0.0, "justification": "",
        "stance_breakdown": {"SUPPORT":0,"CONTRADICT":0,"NEUTRAL":0},
        "persona_insights": {}, "coverage_score": 0.0, "recommendation": "",
        "total_llm_calls": 0, "pipeline_error": None,
    }

def _cred(s):
    return {**s, "avg_source_quality": 0.75, "flagged_sources": [], "high_quality_count": 2, "total_llm_calls": 1}

def _r1(s, certain=True):
    a = FAKE_A if certain else "Evidence is unclear and does not directly address this."
    return {**_cred(s), "total_llm_calls": 7, "qa_memory": [
        {"round":1,"persona":"Fact Checker",   "question":"Q1","answer":a,"insight":"No source confirms" if certain else "unclear"},
        {"round":1,"persona":"Logical Analyst","question":"Q2","answer":a,"insight":"No mechanism"       if certain else "uncertain"},
        {"round":1,"persona":"Bias Detector",  "question":"Q3","answer":a,"insight":"Matches pattern"    if certain else "mixed"},
    ]}
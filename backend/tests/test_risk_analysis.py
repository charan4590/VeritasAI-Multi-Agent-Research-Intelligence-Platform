"""
Phase 3 Milestone 4: regression tests for the Risk Analysis Agent.

Covers the six scenarios called out in the milestone:
  - contradictory sources
  - weak evidence
  - missing evidence
  - single-source dominance
  - cache hits
  - graceful fallback when the LLM fails

Style matches test_fact_verification.py: unit-level tests against the
agent directly (no graph needed), with an autouse fixture forcing a
fresh in-memory cache backend per test for full isolation (same
test-isolation bug this file's sibling already hit once, avoided here
from the start).
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent.graph import initial_state
import agent.agents.risk_analysis as ra_mod
from agent.agents.risk_analysis import (
    RiskAnalysisAgent,
    _compute_risk_signals,
    _single_source_dominance,
    _citation_frequency,
)


@pytest.fixture(autouse=True)
def _isolated_risk_cache(monkeypatch):
    """See test_fact_verification.py's identical fixture for the full
    rationale — forces a throwaway in-memory backend so cache state never
    leaks between tests via the shared on-disk diskcache directory."""
    import agent.cache as cache_mod

    monkeypatch.setattr(cache_mod, "_backend_singleton", cache_mod.InMemoryCacheBackend())
    monkeypatch.setattr(cache_mod, "_instances", {})
    yield


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, questions):
        self.questions = questions
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return FakeResponse(json.dumps(self.questions))


def _source(id, url, title, snippet):
    return {"id": id, "url": url, "title": title, "snippet": snippet, "source_type": "web"}


def _make_state(
    question, sources, report, citations_used=None, citation_verification=None, retrieved_chunks=None
):
    state = initial_state(question)
    state["sources"] = sources
    state["report"] = report
    state["citations_used"] = citations_used or list(sources.keys())
    state["citation_verification"] = citation_verification or []
    state["retrieved_chunks"] = (
        retrieved_chunks or [{"text": "x"}] * 5
    )  # enough to avoid the "thin evidence" signal by default
    return state


class TestContradictorySources:
    def test_contradiction_flagged_in_conflicting_claims(self):
        sources = {
            1: _source(
                1,
                "https://siteone.example.com/a",
                "Study A",
                "The treatment is proven safe and effective for most patients.",
            ),
            2: _source(
                2,
                "https://sitetwo.example.org/b",
                "Study B",
                "The treatment is dangerous and ineffective according to this analysis.",
            ),
            3: _source(
                3, "https://sitethree.example.net/c", "Study C", "Further research supports these findings."
            ),
        }
        report = "Findings are mixed [1][2][3]."
        state = _make_state("is this treatment safe", sources, report)

        signals = _compute_risk_signals(state)

        assert len(signals["conflicting_claims"]) >= 1
        assert signals["risk_score"] > 0

    def test_no_contradiction_no_flag(self):
        sources = {
            1: _source(
                1,
                "https://siteone.example.com/a",
                "Study A",
                "The treatment shows consistent positive results.",
            ),
            2: _source(
                2,
                "https://sitetwo.example.org/b",
                "Study B",
                "Follow-up analysis confirms the positive results.",
            ),
            3: _source(
                3, "https://sitethree.example.net/c", "Study C", "A third study also confirms these results."
            ),
        }
        report = "Findings are consistent [1][2][3]."
        state = _make_state("is this treatment safe", sources, report)

        signals = _compute_risk_signals(state)

        assert signals["conflicting_claims"] == []


class TestFabricatedTableFigures:
    """Regression tests for a real bug found in production: risk_score
    stayed at 0 (falsely "Low Risk") for a report whose Experimental
    Results table was entirely fabricated, because citation_verification
    (Milestone 3) only ever evaluates prose sentences -- table rows are
    explicitly skipped by FactVerificationAgent's claim extraction -- and
    RevisionAgent's more precise per-cell grounding check runs *after*
    risk_analyze in the pipeline, too late to inform this score."""

    def test_fabricated_table_raises_risk_score(self):
        sources = {
            4: _source(
                4,
                "https://arxiv.org/abs/x",
                "Deep Learning Techniques for Lung Cancer Diagnosis",
                "A systematic review of deep learning techniques for lung cancer diagnosis with CT imaging.",
            ),
            5: _source(
                5,
                "https://frontiersin.org/y",
                "Evaluation of lightweight architectures",
                "We evaluate lightweight architectures for lung cancer CT classification.",
            ),
        }
        report = (
            "## 6. Experimental Results\n\n"
            "We evaluate our proposed model [4] and [5].\n\n"
            "| Method | Dataset | Accuracy | AUC | Sensitivity | Specificity |\n"
            "|-|-|-|-|-|-|\n"
            "| Proposed method | LIDC-IDRI | 95% | 0.95 | 90% | 95% |\n"
            "| Proposed method | ELCAP | 92% | 0.92 | 85% | 92% |\n"
        )
        state = _make_state("lung cancer detection", sources, report, citations_used=[4, 5])
        # citation_verification deliberately empty -- table rows were
        # never covered, which is exactly the scenario that produced a
        # falsely-0 risk score before this fix.
        state["citation_verification"] = []

        signals = _compute_risk_signals(state)

        assert signals["risk_score"] > 0
        assert any("table row" in r.lower() for r in signals["identified_risks"])

    def test_grounded_table_does_not_raise_risk_score(self):
        sources = {
            1: _source(
                1, "https://x.com/a", "Paper A", "Our method achieves 94.2% accuracy on the benchmark."
            )
        }
        report = "## Results\n\n" "| Method | Accuracy |\n" "|-|-|\n" "| Ours | 94.2% |\n"
        state = _make_state("q", sources, report, citations_used=[1])
        state["citation_verification"] = []

        signals = _compute_risk_signals(state)

        assert not any("table row" in r.lower() for r in signals["identified_risks"])

    def test_no_report_no_crash(self):
        signals = _compute_risk_signals(_make_state("q", {}, "", citations_used=[]))
        assert signals["risk_score"] == 0


class TestWeakEvidence:
    def test_few_rag_chunks_flagged_as_evidence_gap(self):
        sources = {
            1: _source(1, "https://siteone.example.com/a", "Study A", "Some finding."),
            2: _source(2, "https://sitetwo.example.org/b", "Study B", "Another finding."),
            3: _source(3, "https://sitethree.example.net/c", "Study C", "A third finding."),
        }
        state = _make_state(
            "test question",
            sources,
            "A claim [1].",
            retrieved_chunks=[{"text": "only one chunk"}],
        )

        signals = _compute_risk_signals(state)

        assert any("retrieved" in g.lower() for g in signals["evidence_gaps"])

    def test_low_credibility_ratio_flagged_as_risk(self):
        sources = {
            i: _source(i, f"https://randomblog{i}.example.com/post", f"Blog {i}", "content")
            for i in range(1, 6)
        }
        state = _make_state("test question", sources, "Claims [1][2][3][4][5].")

        signals = _compute_risk_signals(state)

        assert any("credibility" in r.lower() for r in signals["identified_risks"])
        assert signals["risk_score"] > 0


class TestMissingEvidence:
    def test_no_sources_short_circuits_gracefully(self, monkeypatch):
        fake = FakeLLM([])
        monkeypatch.setattr(ra_mod, "get_llm", lambda temperature=0.0: fake)

        state = _make_state("test question", {}, "", citations_used=[])
        result = RiskAnalysisAgent().run(state)

        assert result["risk_score"] is None
        assert result["risk_level"] is None
        assert result["identified_risks"] == []
        assert result["evidence_gaps"] == []
        assert result["conflicting_claims"] == []
        assert result["recommended_follow_up_questions"] == []
        assert fake.call_count == 0

    def test_no_report_short_circuits_gracefully(self):
        sources = {1: _source(1, "https://arxiv.org/abs/x", "Paper X", "content")}
        state = _make_state("test question", sources, "", citations_used=[1])
        result = RiskAnalysisAgent().run(state)
        assert result["risk_score"] is None


class TestSingleSourceDominance:
    def test_dominant_citation_flagged(self):
        counts = {1: 8, 2: 1, 3: 1}
        dominant, dominant_id, share = _single_source_dominance(counts)
        assert dominant is True
        assert dominant_id == 1
        assert share == 0.8

    def test_balanced_citations_not_flagged(self):
        counts = {1: 3, 2: 3, 3: 4}
        dominant, _, _ = _single_source_dominance(counts)
        assert dominant is False

    def test_single_citation_overall_not_dominance(self):
        """A report that only ever cites ONE source isn't 'dominance' --
        that's just a single-source report (caught separately by the
        domain-diversity check)."""
        counts = {1: 5}
        dominant, _, _ = _single_source_dominance(counts)
        assert dominant is False

    def test_full_signal_flags_dominance_via_citation_verification(self):
        sources = {
            1: _source(1, "https://arxiv.org/abs/x", "Paper X", "Detailed findings."),
            2: _source(2, "https://arxiv.org/abs/y", "Paper Y", "Brief mention."),
        }
        verification = [
            {"citation_id": 1, "verdict": "supported", "confidence": 90, "reasoning": "ok", "sentence": "s"}
        ] * 4 + [
            {"citation_id": 2, "verdict": "supported", "confidence": 90, "reasoning": "ok", "sentence": "s"}
        ]
        state = _make_state(
            "test question",
            sources,
            "Claims [1][2].",
            citation_verification=verification,
        )
        signals = _compute_risk_signals(state)
        assert any("leans heavily on a single source" in r for r in signals["identified_risks"])

    def test_citation_frequency_falls_back_to_report_regex_without_verification(self):
        state = initial_state("q")
        state["report"] = "Claim [1]. Another claim [1]. A third claim [2]."
        state["citation_verification"] = []
        counts = _citation_frequency(state)
        assert counts == {1: 2, 2: 1}


class TestCacheHits:
    def test_repeat_signals_hit_cache_not_llm(self, monkeypatch):
        fake = FakeLLM(["What does peer review say?", "Are there more recent studies?"])
        monkeypatch.setattr(ra_mod, "get_llm", lambda temperature=0.3: fake)

        sources = {
            i: _source(i, f"https://randomblog{i}.example.com/post", f"Blog {i}", "content")
            for i in range(1, 6)
        }
        state1 = _make_state("test question", sources, "Claims [1][2][3][4][5].")
        result1 = RiskAnalysisAgent().run(state1)
        assert fake.call_count == 1
        assert result1["recommended_follow_up_questions"]

        state2 = _make_state("test question", sources, "Claims [1][2][3][4][5].")
        result2 = RiskAnalysisAgent().run(state2)
        assert fake.call_count == 1  # unchanged -- cache hit
        assert result2["recommended_follow_up_questions"] == result1["recommended_follow_up_questions"]

    def test_cache_stats_show_hit(self, monkeypatch):
        import agent.cache as cache_mod

        fake = FakeLLM(["Q1?", "Q2?"])
        monkeypatch.setattr(ra_mod, "get_llm", lambda temperature=0.3: fake)

        sources = {
            i: _source(i, f"https://randomblog{i}.example.com/post", f"Blog {i}", "content")
            for i in range(1, 6)
        }
        state = _make_state("test question", sources, "Claims [1][2][3][4][5].")
        RiskAnalysisAgent().run(state)
        RiskAnalysisAgent().run(state)

        stats = cache_mod.get_risk_cache().stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    def test_no_concerns_never_calls_llm(self, monkeypatch):
        """A clean report with no risk signals shouldn't spend an LLM call
        confirming there's nothing to follow up on."""
        fake = FakeLLM([])
        monkeypatch.setattr(ra_mod, "get_llm", lambda temperature=0.3: fake)

        sources = {
            1: _source(1, "https://arxiv.org/abs/a", "Paper A", "Solid finding with 2024 publication date."),
            2: _source(2, "https://ieee.org/paper/b", "Paper B", "Confirms finding, published 2023."),
            3: _source(3, "https://pubmed.ncbi.nlm.nih.gov/c", "Paper C", "Third confirmation, 2024."),
        }
        verification = [
            {"citation_id": 1, "verdict": "supported", "confidence": 90, "reasoning": "ok", "sentence": "s"},
            {"citation_id": 2, "verdict": "supported", "confidence": 90, "reasoning": "ok", "sentence": "s"},
            {"citation_id": 3, "verdict": "supported", "confidence": 90, "reasoning": "ok", "sentence": "s"},
        ]
        state = _make_state(
            "test question",
            sources,
            "Claims [1][2][3].",
            citation_verification=verification,
            retrieved_chunks=[{"text": "x"}] * 5,
        )
        result = RiskAnalysisAgent().run(state)
        assert fake.call_count == 0
        assert result["recommended_follow_up_questions"] == []


class TestFallbackOnLLMError:
    def test_llm_exception_falls_back_to_templated_followups(self, monkeypatch):
        def broken_get_llm(temperature=0.3):
            raise RuntimeError("simulated LLM outage")

        monkeypatch.setattr(ra_mod, "get_llm", broken_get_llm)

        sources = {
            i: _source(i, f"https://randomblog{i}.example.com/post", f"Blog {i}", "content")
            for i in range(1, 6)
        }
        state = _make_state("test question", sources, "Claims [1][2][3][4][5].")

        result = RiskAnalysisAgent().run(state)

        # Core signals must NOT be affected by the LLM failure -- they
        # have no LLM dependency at all.
        assert result["risk_score"] is not None
        assert result["risk_score"] > 0
        assert result["identified_risks"]
        # Follow-ups degrade to the templated fallback, not an empty list
        # or a crash.
        assert isinstance(result["recommended_follow_up_questions"], list)

    def test_malformed_json_response_falls_back(self, monkeypatch):
        class GarbageLLM:
            def invoke(self, messages):
                return FakeResponse("not valid json")

        monkeypatch.setattr(ra_mod, "get_llm", lambda temperature=0.3: GarbageLLM())

        sources = {
            i: _source(i, f"https://randomblog{i}.example.com/post", f"Blog {i}", "content")
            for i in range(1, 6)
        }
        state = _make_state("test question", sources, "Claims [1][2][3][4][5].")

        result = RiskAnalysisAgent().run(state)

        assert result["risk_score"] is not None  # signals still computed
        assert isinstance(result["recommended_follow_up_questions"], list)

    def test_signal_computation_failure_falls_back_gracefully(self, monkeypatch):
        def broken_signals(state):
            raise RuntimeError("simulated bug in signal computation")

        monkeypatch.setattr(ra_mod, "_compute_risk_signals", broken_signals)

        sources = {1: _source(1, "https://arxiv.org/abs/x", "Paper X", "content")}
        state = _make_state("test question", sources, "Claim [1].")

        result = RiskAnalysisAgent().run(state)

        assert result["risk_score"] is None
        assert result["risk_level"] is None
        assert result["recommended_follow_up_questions"] == []
        assert any("unavailable" in entry.lower() for entry in result["log"])


class TestObservabilityIntegration:
    def test_is_agent_subclass_with_name(self):
        from agent.agents.base import Agent

        agent = RiskAnalysisAgent()
        assert isinstance(agent, Agent)
        assert agent.name == "risk_analyze"

    def test_registered_in_graph(self):
        from agent.graph import build_graph

        g = build_graph()
        assert "risk_analyze" in g.nodes


class TestRiskLevelThresholds:
    def test_level_labels_match_score_ranges(self):
        from agent.agents.risk_analysis import _risk_level

        assert _risk_level(0) == "Low"
        assert _risk_level(33) == "Low"
        assert _risk_level(34) == "Medium"
        assert _risk_level(66) == "Medium"
        assert _risk_level(67) == "High"
        assert _risk_level(100) == "High"

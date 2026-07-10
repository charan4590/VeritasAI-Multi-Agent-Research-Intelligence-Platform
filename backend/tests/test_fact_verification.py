"""
Phase 3 Milestone 3: regression tests for the Fact Verification Agent.

Covers the six scenarios called out in the milestone:
  - supported claims
  - unsupported claims
  - mixed citations (some supported, some not, in the same report)
  - missing source text
  - verification cache hits
  - fallback behavior when the verifier errors

Style matches test_pdf_agent.py: unit-level tests against the agent
directly (no LLM/network/graph needed), using monkeypatch to swap in a
fake LLM response.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent.graph import initial_state
import agent.agents.fact_verification as fv_mod
from agent.agents.fact_verification import FactVerificationAgent, _extract_claims


@pytest.fixture(autouse=True)
def _isolated_verification_cache(monkeypatch):
    """
    Forces every test in this file onto a fresh, throwaway in-memory
    cache backend. Without this, the verification cache's default
    diskcache backend persists to the same on-disk directory across
    every test (and across pytest runs), so a (sentence, source_text)
    pair cached by one test would silently produce a cache HIT in a
    later test that expects a cache MISS (e.g. the fallback-on-error
    tests, which need the LLM to actually be called to prove the
    fallback path works) — a test-isolation bug, not a bug in the agent.
    monkeypatch automatically reverts this after each test.
    """
    import agent.cache as cache_mod

    monkeypatch.setattr(cache_mod, "_backend_singleton", cache_mod.InMemoryCacheBackend())
    monkeypatch.setattr(cache_mod, "_instances", {})
    yield


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns a fixed JSON array response regardless of prompt content —
    tests control behavior via what they pass in, not by parsing the
    prompt."""

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        return FakeResponse(json.dumps(self.verdicts))


def _make_state_with_report(report: str, sources: dict, citations_used: list):
    state = initial_state("test question")
    state["report"] = report
    state["sources"] = sources
    state["citations_used"] = citations_used
    return state


SOURCE_1 = {
    "id": 1,
    "url": "https://arxiv.org/abs/x",
    "title": "Paper X",
    "snippet": "The model achieves 94% accuracy on the benchmark dataset.",
    "source_type": "web",
}
SOURCE_2 = {
    "id": 2,
    "url": "https://arxiv.org/abs/y",
    "title": "Paper Y",
    "snippet": "No accuracy figures are reported in this preliminary study.",
    "source_type": "web",
}


class TestSupportedClaims:
    def test_supported_verdict_recorded(self, monkeypatch):
        fake = FakeLLM([{"verdict": "supported", "confidence": 90, "reasoning": "Matches exactly."}])
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        state = _make_state_with_report(
            "The model achieves 94% accuracy on the benchmark dataset [1].",
            {1: SOURCE_1},
            [1],
        )
        result = FactVerificationAgent().run(state)

        assert len(result["citation_verification"]) == 1
        assert result["citation_verification"][0]["verdict"] == "supported"
        assert result["citation_verification"][0]["citation_id"] == 1
        assert result["citation_confidence"] is not None
        assert result["citation_confidence"] > 70  # high for a clean "supported"

    def test_report_and_citations_unchanged(self, monkeypatch):
        """Verification must never alter the report text or citation ids
        that CitationAgent already produced."""
        fake = FakeLLM([{"verdict": "supported", "confidence": 90, "reasoning": "ok"}])
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        original_report = "The model achieves 94% accuracy [1].\n\n---\n\n**Sources**\n\n[1] Paper X — https://arxiv.org/abs/x"
        state = _make_state_with_report(original_report, {1: SOURCE_1}, [1])
        result = FactVerificationAgent().run(state)

        assert "report" not in result  # agent doesn't touch report at all
        assert "citations_used" not in result  # or citations_used


class TestUnsupportedClaims:
    def test_unsupported_verdict_recorded(self, monkeypatch):
        fake = FakeLLM(
            [{"verdict": "unsupported", "confidence": 85, "reasoning": "Source says no figures reported."}]
        )
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        state = _make_state_with_report(
            "The model achieves 99% accuracy according to preliminary results [2].",
            {2: SOURCE_2},
            [2],
        )
        result = FactVerificationAgent().run(state)

        assert result["citation_verification"][0]["verdict"] == "unsupported"
        assert result["citation_confidence"] < 50  # low for an "unsupported" claim


class TestMixedCitations:
    def test_mixed_verdicts_both_recorded(self, monkeypatch):
        fake = FakeLLM(
            [
                {"verdict": "supported", "confidence": 90, "reasoning": "Matches."},
                {"verdict": "unsupported", "confidence": 80, "reasoning": "Contradicted."},
            ]
        )
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        report = (
            "The model achieves 94% accuracy on the benchmark [1]. "
            "It also reportedly reaches 99% in preliminary trials [2]."
        )
        state = _make_state_with_report(report, {1: SOURCE_1, 2: SOURCE_2}, [1, 2])
        result = FactVerificationAgent().run(state)

        verdicts = {v["citation_id"]: v["verdict"] for v in result["citation_verification"]}
        assert verdicts[1] == "supported"
        assert verdicts[2] == "unsupported"
        # overall confidence should sit between a pure-supported and pure-unsupported run
        assert 0 < result["citation_confidence"] < 100


class TestMissingSourceText:
    def test_claim_with_no_source_text_marked_cannot_determine(self, monkeypatch):
        fake = FakeLLM([])  # should never be called -- nothing verifiable to send
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        empty_source = {
            "id": 3,
            "url": "https://arxiv.org/abs/z",
            "title": "Paper Z",
            "snippet": "",
            "source_type": "web",
        }
        state = _make_state_with_report(
            "This claim cites a source with no retrievable text [3].",
            {3: empty_source},
            [3],
        )
        result = FactVerificationAgent().run(state)

        assert len(result["citation_verification"]) == 1
        assert result["citation_verification"][0]["verdict"] == "cannot_determine"
        assert result["citation_verification"][0]["confidence"] == 0
        assert fake.call_count == 0  # never called the LLM for an unverifiable claim

    def test_extract_claims_skips_sources_footer(self):
        report = (
            "The model performs well [1].\n\n---\n\n**Sources**\n\n"
            "[1] Paper X — https://arxiv.org/abs/x\n[99] Fake — https://example.com"
        )
        claims = _extract_claims(report, {1, 99})
        # Only the body sentence should be picked up -- not the reference list lines
        assert len(claims) == 1
        assert claims[0]["citation_id"] == 1


class TestVerificationCacheHits:
    def test_repeat_claim_hits_cache_not_llm(self, monkeypatch):
        fake = FakeLLM([{"verdict": "supported", "confidence": 90, "reasoning": "ok"}])
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        state1 = _make_state_with_report(
            "The model achieves 94% accuracy on the benchmark dataset [1].",
            {1: SOURCE_1},
            [1],
        )
        result1 = FactVerificationAgent().run(state1)
        assert fake.call_count == 1

        # Identical sentence + identical source text -> second run should
        # hit the cache and NOT call the LLM again.
        state2 = _make_state_with_report(
            "The model achieves 94% accuracy on the benchmark dataset [1].",
            {1: SOURCE_1},
            [1],
        )
        result2 = FactVerificationAgent().run(state2)
        assert fake.call_count == 1  # unchanged -- second run was a cache hit
        assert (
            result2["citation_verification"][0]["verdict"] == result1["citation_verification"][0]["verdict"]
        )

    def test_cache_stats_show_hit(self, monkeypatch):
        import agent.cache as cache_mod

        fake = FakeLLM([{"verdict": "supported", "confidence": 90, "reasoning": "ok"}])
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        state = _make_state_with_report(
            "The model achieves 94% accuracy on the benchmark dataset [1].",
            {1: SOURCE_1},
            [1],
        )
        FactVerificationAgent().run(state)
        FactVerificationAgent().run(state)

        stats = cache_mod.get_verification_cache().stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestFallbackOnVerifierError:
    def test_llm_exception_falls_back_gracefully(self, monkeypatch):
        def broken_get_llm(temperature=0.0):
            raise RuntimeError("simulated LLM outage")

        monkeypatch.setattr(fv_mod, "get_llm", broken_get_llm)

        original_report = "The model achieves 94% accuracy [1].\n\n---\n\n**Sources**\n\n[1] Paper X — https://arxiv.org/abs/x"
        state = _make_state_with_report(original_report, {1: SOURCE_1}, [1])

        result = FactVerificationAgent().run(state)

        # Must not raise (already implicit -- we got here), and must
        # gracefully degrade with the documented empty-result contract.
        assert result["citation_verification"] == []
        assert result["citation_confidence"] is None
        assert any("unavailable" in entry.lower() for entry in result["log"])

    def test_malformed_json_response_falls_back_gracefully(self, monkeypatch):
        class GarbageLLM:
            def invoke(self, messages):
                return FakeResponse("this is not json at all")

        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: GarbageLLM())

        state = _make_state_with_report(
            "The model achieves 94% accuracy [1].",
            {1: SOURCE_1},
            [1],
        )
        result = FactVerificationAgent().run(state)

        assert result["citation_verification"] == []
        assert result["citation_confidence"] is None

    def test_wrong_length_response_falls_back_gracefully(self, monkeypatch):
        # Two claims sent, but LLM only returns one verdict -- length
        # mismatch must be treated as a failure, not silently zipped short.
        fake = FakeLLM([{"verdict": "supported", "confidence": 90, "reasoning": "ok"}])
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        report = (
            "The model achieves 94% accuracy on the benchmark [1]. "
            "It also reportedly reaches 99% in preliminary trials [2]."
        )
        state = _make_state_with_report(report, {1: SOURCE_1, 2: SOURCE_2}, [1, 2])
        result = FactVerificationAgent().run(state)

        assert result["citation_verification"] == []
        assert result["citation_confidence"] is None

    def test_no_citations_short_circuits_without_calling_llm(self, monkeypatch):
        fake = FakeLLM([])
        monkeypatch.setattr(fv_mod, "get_llm", lambda temperature=0.0: fake)

        state = _make_state_with_report("A report with no citations at all.", {}, [])
        result = FactVerificationAgent().run(state)

        assert result["citation_verification"] == []
        assert result["citation_confidence"] is None
        assert fake.call_count == 0


class TestObservabilityIntegration:
    """FactVerificationAgent is a proper Agent subclass -- confirms it
    gets the same tracker/tracer wrapping as every other agent (Milestone 1
    infrastructure), satisfying 'integrate with observability'."""

    def test_is_agent_subclass_with_name(self):
        from agent.agents.base import Agent

        agent = FactVerificationAgent()
        assert isinstance(agent, Agent)
        assert agent.name == "fact_verify"

    def test_registered_in_graph(self):
        from agent.graph import build_graph

        g = build_graph()
        assert "fact_verify" in g.nodes

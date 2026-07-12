"""
Phase 6 (final polish): regression tests for the IEEE-style References
section, deduplication, and full-title preservation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.agents.citation import CitationAgent, format_ieee_reference
from agent.agents.revision import RevisionAgent
from agent.graph import initial_state


def _source(id, url, title, snippet, source_type="web"):
    return {"id": id, "url": url, "title": title, "snippet": snippet, "source_type": source_type}


class TestIEEEFormat:
    def test_web_source_format(self):
        ref = format_ieee_reference(1, _source(1, "https://arxiv.org/abs/x", "A Study of Things", "..."))
        assert ref == '[1] "A Study of Things," [Online]. Available: https://arxiv.org/abs/x'

    def test_pdf_source_tagged(self):
        ref = format_ieee_reference(
            2, _source(2, "pdf://doc.pdf#page1", "Internal Report (p.1)", "...", source_type="pdf")
        )
        assert ref.startswith('[2] [PDF] "Internal Report (p.1),"')

    def test_full_title_not_truncated(self):
        long_title = "A Comprehensive Survey of Hybrid Convolutional Neural Network and Long Short-Term Memory Architectures for Medical Image Classification Tasks"
        ref = format_ieee_reference(1, _source(1, "https://x.com/a", long_title, "..."))
        assert long_title in ref  # not shortened, unlike the frontend's compact citation badge


class TestCitationAgentUsesIEEEFormat:
    def test_references_header_is_ieee_style(self):
        state = initial_state("q")
        state["sources"] = {1: _source(1, "https://x.com/a", "Paper A", "supports")}
        state["report"] = "A claim [1]."
        result = CitationAgent().run(state)
        assert "**References**" in result["report"]
        assert '[1] "Paper A," [Online]. Available: https://x.com/a' in result["report"]

    def test_dedup_by_url_defensive(self):
        """Two different citation ids that (abnormally) point at the same
        URL should only produce one reference entry."""
        state = initial_state("q")
        state["sources"] = {
            1: _source(1, "https://x.com/a", "Paper A", "s"),
            2: _source(
                2, "https://x.com/a", "Paper A (duplicate)", "s"
            ),  # same URL, shouldn't normally happen
        }
        state["report"] = "Claim one [1]. Claim two [2]."
        result = CitationAgent().run(state)
        refs_section = result["report"].split("**References**")[1]
        assert refs_section.count("https://x.com/a") == 1

    def test_llm_generated_references_section_stripped(self):
        """Regression test for a real bug found in production: the
        academic synthesis prompt asks the model to write its own
        "## 9. References" section, which -- left alone -- produces two
        disagreeing References sections once this agent appends its own
        validated one. Also confirms the model's own reference-list
        numbers (e.g. "[2] Some other paper...") don't get miscounted as
        real citations to source 2."""
        report = (
            "## 8. Limitations\n\n"
            "Our proposed method has several limitations [1].\n\n"
            "## 9. References\n\n"
            "[1] Transfer learning based deep architecture for lung cancer "
            "classification | Scientific Reports https://www.nature.com/articles/fake\n"
            "[2] Deep Learning-Based Lung Cancer Detection https://ieeexplore.ieee.org/document/fake"
        )
        sources = {
            1: _source(
                1,
                "https://www.nature.com/articles/fake",
                "Transfer learning based deep architecture",
                "supports",
            ),
        }
        state = initial_state("q")
        state["report"] = report
        state["sources"] = sources
        result = CitationAgent().run(state)

        assert "## 9. References" not in result["report"]
        assert result["report"].count("**References**") == 1
        assert "Limitations" in result["report"]
        # The model's own "[2] ..." reference-list line must NOT be
        # miscounted as a real citation to a (non-existent) source 2.
        assert result["citations_used"] == [1]

    def test_no_references_section_is_a_noop(self):
        """Most reports (general/technical intent) never had a
        model-generated References section at all -- confirms stripping
        logic doesn't touch anything when there's nothing to strip."""
        report = "A claim [1]."
        state = initial_state("q")
        state["report"] = report
        state["sources"] = {1: _source(1, "https://x.com/a", "Paper A", "s")}
        result = CitationAgent().run(state)
        assert "A claim" in result["report"]
        assert result["citations_used"] == [1]


class TestRevisionAgentPreservesIEEEFormat:
    def test_rebuilt_footer_uses_ieee_format(self):
        state = initial_state("q")
        sources = {1: _source(1, "https://x.com/a", "Paper A", "supports the claim")}
        state["sources"] = sources
        state["report"] = "A claim [1].\n\n---\n\n**References**\n\n" + format_ieee_reference(1, sources[1])
        state["citation_verification"] = [
            {
                "sentence": "A claim [1].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
        ]
        result = RevisionAgent().run(state)
        assert '[1] "Paper A," [Online]. Available: https://x.com/a' in result["report"]

    def test_rebuilt_footer_dedups_after_revision(self):
        state = initial_state("q")
        sources = {
            1: _source(1, "https://x.com/a", "Paper A", "s"),
            2: _source(2, "https://x.com/a", "Paper A dup", "s"),
        }
        state["sources"] = sources
        state["report"] = "Claim [1][2]."
        state["citation_verification"] = [
            {
                "sentence": "Claim [1][2].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
            {
                "sentence": "Claim [1][2].",
                "citation_id": 2,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
        ]
        result = RevisionAgent().run(state)
        refs_section = result["report"].split("**References**")[1]
        assert refs_section.count("https://x.com/a") == 1

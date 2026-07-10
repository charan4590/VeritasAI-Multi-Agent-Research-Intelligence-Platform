"""
Phase 3 Milestone 2: regression tests for PDF Agent integration.

Covers the four scenarios called out in the milestone:
  - web-only research (no PDFs uploaded) -> behavior identical to before
  - PDF-only research (no web results)
  - mixed web + PDF research
  - empty PDF retrieval results (PDFAgent degrades gracefully)

These test SupervisorAgent directly (unit-style, no LLM/graph needed)
since that's where the merge logic lives — this matches the existing
test suite's style (heuristic/logic tests, no live network or LLM calls).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent.graph import initial_state
from agent.agents.supervisor import SupervisorAgent


WEB_RESULT = {
    "url": "https://arxiv.org/abs/fake-paper",
    "title": "A Fake Paper About Testing",
    "content": "This is web-sourced content about the research topic.",
}

PDF_CHUNK = {
    "url": "pdf://my_report.pdf#page3",
    "title": "my_report.pdf (p.3)",
    "content": "This is PDF-sourced content about the research topic.",
    "relevance": 0.87,
    "source_type": "pdf",
}


def _make_state(question="test research question"):
    state = initial_state(question, max_rounds=1)
    state["plan"] = ["test query one", "test query two"]
    state["round"] = 0
    return state


class TestWebOnlyResearch:
    """No PDFs uploaded -> behavior must be identical to before this milestone."""

    def test_no_pdf_log_line_emitted(self, monkeypatch):
        agent = SupervisorAgent()
        monkeypatch.setattr(agent.web_agent, "search", lambda queries: {q: [WEB_RESULT] for q in queries})
        monkeypatch.setattr(agent.academic_agent, "search", lambda queries: {q: [] for q in queries})
        monkeypatch.setattr(agent.pdf_agent, "search", lambda queries, top_k=3: {q: [] for q in queries})
        monkeypatch.setattr("agent.agents.supervisor._fetch_full_content_batch", lambda urls: {})

        result = agent.run(_make_state())

        assert not any("PDF" in entry for entry in result["log"])
        assert all(s.get("source_type", "web") == "web" for s in result["sources"].values())

    def test_sources_default_to_web_type(self, monkeypatch):
        agent = SupervisorAgent()
        monkeypatch.setattr(agent.web_agent, "search", lambda queries: {q: [WEB_RESULT] for q in queries})
        monkeypatch.setattr(agent.pdf_agent, "search", lambda queries, top_k=3: {q: [] for q in queries})
        monkeypatch.setattr("agent.agents.supervisor._fetch_full_content_batch", lambda urls: {})

        result = agent.run(_make_state())

        assert len(result["sources"]) > 0
        for s in result["sources"].values():
            assert s["source_type"] == "web"


class TestPDFOnlyResearch:
    """No web results (e.g. Tavily unavailable) but PDFs ARE uploaded and relevant."""

    def test_pdf_sources_appear_when_web_is_empty(self, monkeypatch):
        agent = SupervisorAgent()
        monkeypatch.setattr(agent.web_agent, "search", lambda queries: {q: [] for q in queries})
        monkeypatch.setattr(agent.pdf_agent, "search", lambda queries, top_k=3: {q: [PDF_CHUNK] for q in queries})
        monkeypatch.setattr("agent.agents.supervisor._fetch_full_content_batch", lambda urls: {})

        result = agent.run(_make_state())

        assert len(result["sources"]) > 0
        assert all(s["source_type"] == "pdf" for s in result["sources"].values())
        assert any("PDF search" in entry for entry in result["log"])


class TestMixedWebAndPDFResearch:
    """Both web and PDF results exist for the same queries — must merge
    without clobbering either, with correct labeling and continuous
    citation numbering."""

    def test_both_source_types_present(self, monkeypatch):
        agent = SupervisorAgent()
        monkeypatch.setattr(agent.web_agent, "search", lambda queries: {q: [WEB_RESULT] for q in queries})
        monkeypatch.setattr(agent.pdf_agent, "search", lambda queries, top_k=3: {q: [PDF_CHUNK] for q in queries})
        monkeypatch.setattr("agent.agents.supervisor._fetch_full_content_batch", lambda urls: {})

        result = agent.run(_make_state())

        types = {s["source_type"] for s in result["sources"].values()}
        assert types == {"web", "pdf"}

    def test_citation_ids_are_one_continuous_sequence(self, monkeypatch):
        """Web and PDF sources must share the same id counter — no
        separate numbering scheme for PDF-derived sources."""
        agent = SupervisorAgent()
        monkeypatch.setattr(agent.web_agent, "search", lambda queries: {q: [WEB_RESULT] for q in queries})
        monkeypatch.setattr(agent.pdf_agent, "search", lambda queries, top_k=3: {q: [PDF_CHUNK] for q in queries})
        monkeypatch.setattr("agent.agents.supervisor._fetch_full_content_batch", lambda urls: {})

        result = agent.run(_make_state())

        ids = sorted(result["sources"].keys())
        assert ids == list(range(1, len(ids) + 1)), "citation ids must be one contiguous sequence"

    def test_pdf_metrics_logged(self, monkeypatch):
        agent = SupervisorAgent()
        monkeypatch.setattr(agent.web_agent, "search", lambda queries: {q: [WEB_RESULT] for q in queries})
        monkeypatch.setattr(agent.pdf_agent, "search", lambda queries, top_k=3: {q: [PDF_CHUNK] for q in queries})
        monkeypatch.setattr("agent.agents.supervisor._fetch_full_content_batch", lambda urls: {})

        result = agent.run(_make_state())

        pdf_log_lines = [entry for entry in result["log"] if "PDF search" in entry]
        assert len(pdf_log_lines) == 1
        assert "chunks retrieved" in pdf_log_lines[0]
        assert "added as sources" in pdf_log_lines[0]


class TestEmptyPDFRetrieval:
    """PDFAgent itself must degrade gracefully — this is what search_pdfs()
    already does when no PDFs are uploaded (no ChromaDB "pdf_docs"
    collection yet), and PDFAgent must not blow up if search_pdfs()
    raises for any other reason either."""

    def test_pdf_agent_search_returns_empty_dict_values_when_no_pdfs(self, monkeypatch):
        import agent.agents.pdf_agent as pdf_agent_mod
        monkeypatch.setattr(pdf_agent_mod, "search_pdfs", lambda query, top_k: [])

        agent = pdf_agent_mod.PDFAgent()
        results = agent.search(["q1", "q2"])

        assert results == {"q1": [], "q2": []}

    def test_pdf_agent_search_survives_exception_per_query(self, monkeypatch):
        import agent.agents.pdf_agent as pdf_agent_mod

        def flaky_search(query, top_k):
            if query == "q1":
                raise RuntimeError("simulated chromadb failure")
            return [PDF_CHUNK]

        monkeypatch.setattr(pdf_agent_mod, "search_pdfs", flaky_search)

        agent = pdf_agent_mod.PDFAgent()
        results = agent.search(["q1", "q2"])

        assert results["q1"] == []  # failed query degrades to empty, doesn't crash the batch
        assert results["q2"] == [PDF_CHUNK]

    def test_supervisor_unaffected_by_empty_pdf_results(self, monkeypatch):
        agent = SupervisorAgent()
        monkeypatch.setattr(agent.web_agent, "search", lambda queries: {q: [WEB_RESULT] for q in queries})
        monkeypatch.setattr(agent.pdf_agent, "search", lambda queries, top_k=3: {q: [] for q in queries})
        monkeypatch.setattr("agent.agents.supervisor._fetch_full_content_batch", lambda urls: {})

        result = agent.run(_make_state())

        assert len(result["sources"]) > 0
        assert not any("PDF search" in entry for entry in result["log"])


class TestCitationLabeling:
    """CitationAgent must visibly tag PDF sources in the references list
    without changing citation numbering."""

    def test_pdf_sources_tagged_in_references(self):
        from agent.agents.citation import CitationAgent

        state = initial_state("test question")
        state["sources"] = {
            1: {"id": 1, "url": "https://arxiv.org/abs/x", "title": "Web Paper", "snippet": "...", "source_type": "web"},
            2: {"id": 2, "url": "pdf://doc.pdf#page1", "title": "doc.pdf (p.1)", "snippet": "...", "source_type": "pdf"},
        }
        state["report"] = "Web claim [1]. PDF claim [2]."

        result = CitationAgent().run(state)

        assert "[1] Web Paper" in result["report"]
        assert "[2] [PDF] doc.pdf (p.1)" in result["report"]
        assert result["citations_used"] == [1, 2]

    def test_pdf_citation_count_in_log(self):
        from agent.agents.citation import CitationAgent

        state = initial_state("test question")
        state["sources"] = {
            1: {"id": 1, "url": "https://arxiv.org/abs/x", "title": "Web Paper", "snippet": "...", "source_type": "web"},
            2: {"id": 2, "url": "pdf://doc.pdf#page1", "title": "doc.pdf (p.1)", "snippet": "...", "source_type": "pdf"},
        }
        state["report"] = "Web claim [1]. PDF claim [2]."

        result = CitationAgent().run(state)

        assert "1 from uploaded PDFs" in result["log"][-1]

    def test_web_only_citation_message_unchanged(self):
        """No PDF citations -> the validation log message must be
        byte-identical to before this milestone (no '(N from uploaded
        PDFs)' suffix at all)."""
        from agent.agents.citation import CitationAgent

        state = initial_state("test question")
        state["sources"] = {
            1: {"id": 1, "url": "https://arxiv.org/abs/x", "title": "Web Paper", "snippet": "...", "source_type": "web"},
        }
        state["report"] = "Web claim [1]."

        result = CitationAgent().run(state)

        assert result["log"][-1] == "Validated 1 citations"

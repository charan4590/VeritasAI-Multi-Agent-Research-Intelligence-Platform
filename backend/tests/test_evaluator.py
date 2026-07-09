"""
Unit tests for the evaluation framework.
These tests are all heuristic (no LLM calls) so they run instantly.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.evaluator import (
    citation_score, source_diversity_score,
    hallucination_risk_score, _grade,
)


class TestCitationScore:
    def test_no_citations_returns_zero(self):
        report = "This is a paragraph.\n\nAnother paragraph here."
        score, msg = citation_score(report, [])
        assert score == 0

    def test_all_cited_returns_hundred(self):
        report = "This is the first substantive claim supporting our argument [1].\n\nThis is the second detailed claim with evidence from multiple sources [2].\n\nThis is the third point which builds on previous arguments made above [1]."
        score, msg = citation_score(report, [1, 2])
        assert score == 100

    def test_half_cited_returns_fifty(self):
        report = "Cited paragraph [1].\n\nUncited paragraph here with enough words."
        score, msg = citation_score(report, [1])
        assert 40 <= score <= 60

    def test_short_paragraphs_ignored(self):
        report = "Hi\n\nLong paragraph with actual content and citations [1]."
        score, msg = citation_score(report, [1])
        assert score == 100  # Only the long paragraph counts


class TestSourceDiversityScore:
    def test_no_sources_returns_zero(self):
        score, msg = source_diversity_score({}, [])
        assert score == 0

    def test_five_unique_domains_returns_hundred(self, sample_sources):
        extra_sources = {
            4: {"id": 4, "url": "https://nature.com/paper", "title": "Nature", "snippet": ""},
            5: {"id": 5, "url": "https://reuters.com/news", "title": "Reuters", "snippet": ""},
        }
        all_sources = {**sample_sources, **extra_sources}
        score, msg = source_diversity_score(all_sources, [1, 2, 3, 4, 5])
        assert score == 100

    def test_single_domain_returns_twenty(self):
        sources = {
            1: {"id": 1, "url": "https://bbc.com/a", "title": "A", "snippet": ""},
            2: {"id": 2, "url": "https://bbc.com/b", "title": "B", "snippet": ""},
        }
        score, msg = source_diversity_score(sources, [1, 2])
        assert score == 20

    def test_three_domains_returns_sixty(self, sample_sources):
        score, msg = source_diversity_score(sample_sources, [1, 2, 3])
        assert score == 60


class TestHallucinationRisk:
    def test_all_valid_citations_returns_hundred(self, sample_report, sample_sources):
        score, msg = hallucination_risk_score(sample_report, sample_sources, [1, 2, 3])
        assert score == 100

    def test_hallucinated_citation_lowers_score(self, sample_sources):
        report = "Some claim [1]. Made up source [99]. Another claim [2]."
        score, msg = hallucination_risk_score(report, sample_sources, [1, 2])
        assert score < 100
        assert "hallucinated" in msg

    def test_no_citations_returns_fifty(self, sample_sources):
        report = "A report with no citation markers at all."
        score, msg = hallucination_risk_score(report, sample_sources, [])
        assert score == 50


class TestGrade:
    def test_grade_boundaries(self):
        assert _grade(90) == "A"
        assert _grade(85) == "A"
        assert _grade(84) == "B"
        assert _grade(70) == "B"
        assert _grade(69) == "C"
        assert _grade(55) == "C"
        assert _grade(54) == "D"
        assert _grade(39) == "F"

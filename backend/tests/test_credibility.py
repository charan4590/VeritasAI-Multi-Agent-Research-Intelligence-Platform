"""Unit tests for source credibility scoring."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.credibility import score_url, score_label, compute_confidence


class TestScoreUrl:
    def test_arxiv_scores_95(self):
        assert score_url("https://arxiv.org/abs/2401.12345") == 95

    def test_bbc_scores_80(self):
        assert score_url("https://bbc.com/news/technology") == 80

    def test_edu_domain_scores_70(self):
        assert score_url("https://mit.edu/paper") == 80

    def test_gov_domain_scores_70(self):
        assert score_url("https://cdc.gov/report") == 70

    def test_unknown_domain_scores_50(self):
        assert score_url("https://randomwebsite123.com/post") == 50

    def test_empty_url_scores_40(self):
        assert score_url("") == 40

    def test_www_stripped(self):
        assert score_url("https://www.arxiv.org/abs/123") == 95


class TestScoreLabel:
    def test_high_label(self):
        assert score_label(95) == "High"
        assert score_label(90) == "High"

    def test_good_label(self):
        assert score_label(80) == "Good"
        assert score_label(70) == "Good"

    def test_medium_label(self):
        assert score_label(60) == "Medium"
        assert score_label(55) == "Medium"

    def test_low_label(self):
        assert score_label(40) == "Low"
        assert score_label(0) == "Low"


class TestComputeConfidence:
    def test_no_citations_returns_zero(self):
        assert compute_confidence({}, []) == 0

    def test_high_credibility_sources_give_high_confidence(self):
        sources = {
            1: {"id": 1, "url": "https://arxiv.org/paper", "title": "A", "snippet": ""},
            2: {"id": 2, "url": "https://nature.com/paper", "title": "B", "snippet": ""},
        }
        confidence = compute_confidence(sources, [1, 2])
        assert confidence >= 35

    def test_five_citations_gives_full_credit(self):
        sources = {i: {"id": i, "url": f"https://arxiv.org/{i}", "title": f"P{i}", "snippet": ""} for i in range(1, 6)}
        confidence = compute_confidence(sources, list(range(1, 6)))
        assert confidence == 95  # arxiv score * 1.0 factor

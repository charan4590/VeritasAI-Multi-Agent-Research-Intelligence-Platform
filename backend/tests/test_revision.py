"""
Phase 5: regression tests for the Revision Agent (Generate -> Verify ->
Revise -> Finalize).

Style matches test_risk_analysis.py / test_fact_verification.py:
unit-level tests against the agent directly, no graph/LLM needed (this
agent makes zero LLM calls by design -- see revision.py's module
docstring), fully deterministic.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.agents.revision import (
    RevisionAgent,
    _detect_report_type,
    _numbers_grounded,
)
from agent.graph import initial_state


def _source(id, url, title, snippet):
    return {"id": id, "url": url, "title": title, "snippet": snippet, "source_type": "web"}


def _make_state(question, report, sources, citation_verification=None, citations_used=None):
    state = initial_state(question)
    state["report"] = report
    state["sources"] = sources
    state["citations_used"] = citations_used or list(sources.keys())
    state["citation_verification"] = citation_verification or []
    return state


class TestUnsupportedClaimRemoval:
    def test_fully_unsupported_sentence_removed(self):
        report = "The drug is effective for most patients [1]. It also cures unrelated conditions [1]."
        sources = {1: _source(1, "https://x.com/a", "Study A", "The drug is effective for most patients.")}
        verification = [
            {
                "sentence": "The drug is effective for most patients [1].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
            {
                "sentence": "It also cures unrelated conditions [1].",
                "citation_id": 1,
                "verdict": "unsupported",
                "confidence": 85,
                "reasoning": "not in source",
            },
        ]
        state = _make_state("is this drug effective", report, sources, verification)
        result = RevisionAgent().run(state)

        assert "cures unrelated conditions" not in result["report"]
        assert "effective for most patients" in result["report"]
        assert len(result["claims_removed"]) == 1

    def test_supported_claims_fully_preserved(self):
        report = "The drug is effective for most patients [1]."
        sources = {1: _source(1, "https://x.com/a", "Study A", "The drug is effective for most patients.")}
        verification = [
            {
                "sentence": "The drug is effective for most patients [1].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
        ]
        state = _make_state("q", report, sources, verification)
        result = RevisionAgent().run(state)

        assert "effective for most patients [1]" in result["report"]
        assert result["claims_removed"] == []


class TestPartiallySupportedRewriting:
    def test_partial_claim_annotated_not_removed(self):
        report = "The replication study confirmed similar results [2]."
        sources = {
            2: _source(2, "https://x.com/b", "Study B", "Our replication achieved comparable results.")
        }
        verification = [
            {
                "sentence": "The replication study confirmed similar results [2].",
                "citation_id": 2,
                "verdict": "partially_supported",
                "confidence": 60,
                "reasoning": "close but not exact",
            },
        ]
        state = _make_state("q", report, sources, verification)
        result = RevisionAgent().run(state)

        assert "confirmed similar results [2]" in result["report"]
        assert "partially supported" in result["report"].lower()
        assert result["claims_rewritten"] == ["The replication study confirmed similar results [2]."]


class TestMixedVerdictSentence:
    def test_only_unsupported_citation_marker_stripped(self):
        report = "The model achieves strong results [1][2]."
        sources = {
            1: _source(1, "https://x.com/a", "Paper A", "The model achieves strong results."),
            2: _source(2, "https://x.com/b", "Paper B", "No results are reported in this preliminary note."),
        }
        verification = [
            {
                "sentence": "The model achieves strong results [1][2].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
            {
                "sentence": "The model achieves strong results [1][2].",
                "citation_id": 2,
                "verdict": "unsupported",
                "confidence": 80,
                "reasoning": "not reported",
            },
        ]
        state = _make_state("q", report, sources, verification)
        result = RevisionAgent().run(state)

        assert "[1]" in result["report"]
        assert (
            "[2]" not in result["report"].split("---")[0]
        )  # stripped from body, not from footer coincidentally
        assert "achieves strong results" in result["report"]


class TestStrictGroundingNumericClaims:
    def test_grounded_number_untouched(self):
        assert _numbers_grounded(
            "Accuracy reached 94.2% on the benchmark [1].",
            "The model reached 94.2% accuracy on the benchmark.",
        )

    def test_ungrounded_number_detected(self):
        assert not _numbers_grounded(
            "Accuracy reached 99.9% on the benchmark [1].",
            "The model reached 94.2% accuracy on the benchmark.",
        )

    def test_citation_marker_digit_not_treated_as_a_claim(self):
        """Regression test for a real bug caught during development: a
        bare number regex matches the digit inside "[1]" itself unless
        citation markers are stripped first."""
        assert _numbers_grounded(
            "The model performs well [1].",
            "Completely unrelated source text with no numbers at all.",
        )

    def test_fabricated_accuracy_marked_unverified_in_full_pipeline(self):
        report = "The model achieves 99.9% accuracy on the held-out set [1]."
        sources = {
            1: _source(1, "https://x.com/a", "Paper A", "We describe a hybrid architecture for the task.")
        }
        verification = [
            {
                "sentence": report,
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 70,
                "reasoning": "topically related",
            },
        ]
        state = _make_state("q", report, sources, verification)
        result = RevisionAgent().run(state)

        assert "[unverified" in result["report"]
        assert "99.9%" in result["report"]  # marked, not silently deleted


class TestTableGrounding:
    def test_ungrounded_table_row_marked(self):
        report = "## Results\n\n" "| Method | Accuracy |\n" "|-|-|\n" "| Ours | 99.9% |\n"
        sources = {
            1: _source(
                1, "https://x.com/a", "Paper A", "We propose a new method with no reported accuracy figure."
            )
        }
        state = _make_state("q", report, sources, citation_verification=[])
        result = RevisionAgent().run(state)

        assert "unverified" in result["report"]
        assert "99.9%" in result["report"]

    def test_grounded_table_row_untouched(self):
        report = "## Results\n\n" "| Method | Accuracy |\n" "|-|-|\n" "| Ours | 94.2% |\n"
        sources = {1: _source(1, "https://x.com/a", "Paper A", "Our method achieves 94.2% accuracy.")}
        state = _make_state("q", report, sources, citation_verification=[])
        result = RevisionAgent().run(state)

        assert "unverified" not in result["report"]
        assert "94.2%" in result["report"]

    def test_table_header_and_separator_rows_never_flagged(self):
        report = "| Method | Accuracy |\n|-|-|\n"
        sources = {1: _source(1, "https://x.com/a", "Paper A", "No figures here.")}
        state = _make_state("q", report, sources, citation_verification=[])
        result = RevisionAgent().run(state)
        assert "unverified" not in result["report"]


class TestReportTypeDetectionAndSectionPruning:
    def test_comparative_question_detected(self):
        assert _detect_report_type("compare CNN vs transformer models", {}) == "Comparative Analysis"

    def test_survey_question_detected(self):
        assert _detect_report_type("survey of current approaches to NLP", {}) == "Research Survey"

    def test_experimental_study_requires_source_evidence(self):
        sources = {
            1: _source(
                1,
                "https://x.com/a",
                "Paper A",
                "We propose a novel CNN architecture achieving 95% accuracy on the benchmark dataset.",
            ),
        }
        assert _detect_report_type("tell me about lung cancer detection", sources) == "Experimental Study"

    def test_general_question_without_experimental_evidence(self):
        sources = {
            1: _source(
                1, "https://x.com/a", "Article", "A general overview article with no methodology details."
            )
        }
        assert _detect_report_type("what is lung cancer", sources) == "General Research Answer"

    def test_experimental_sections_pruned_for_literature_review(self):
        report = (
            "## 1. Introduction\n\nBackground on the topic [1].\n\n"
            "## 4. Model Architecture\n\nA CNN with 12 layers [1].\n\n"
            "## 6. Experimental Results\n\n| M | Acc |\n|-|-|\n| X | 99% |\n\n"
            "## 8. Limitations\n\nSome limitations apply [1].\n"
        )
        sources = {1: _source(1, "https://x.com/a", "Review", "A review of prior approaches.")}
        state = _make_state("literature review of the topic", report, sources)
        result = RevisionAgent().run(state)

        assert "Model Architecture" not in result["report"]
        assert "Experimental Results" not in result["report"]
        assert "Introduction" in result["report"]
        assert "Limitations" in result["report"]

    def test_experimental_sections_kept_when_explicitly_requested(self):
        report = "## 4. Model Architecture\n\nA CNN with 12 layers [1].\n"
        sources = {1: _source(1, "https://x.com/a", "Paper", "Describes the architecture.")}
        state = _make_state("what is the model architecture used", report, sources)
        result = RevisionAgent().run(state)
        assert "Model Architecture" in result["report"]

    def test_sections_kept_for_genuine_experimental_study(self):
        report = "## 6. Experimental Results\n\n| M | Acc |\n|-|-|\n| X | 94.2% |\n"
        sources = {
            1: _source(
                1,
                "https://x.com/a",
                "Paper",
                "Our method achieves 94.2% accuracy using a novel CNN architecture, evaluated with standard metrics.",
            )
        }
        state = _make_state("hybrid CNN architecture for detection", report, sources)
        result = RevisionAgent().run(state)
        assert "Experimental Results" in result["report"]


class TestCitationNumberingAndFooterRebuild:
    def test_citation_numbers_unchanged_for_surviving_claims(self):
        report = "First claim [1]. Second claim [2]. Third claim [3]."
        sources = {
            1: _source(1, "https://x.com/a", "A", "supports first"),
            2: _source(2, "https://x.com/b", "B", "supports second"),
            3: _source(3, "https://x.com/c", "C", "unrelated"),
        }
        verification = [
            {
                "sentence": "First claim [1].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
            {
                "sentence": "Second claim [2].",
                "citation_id": 2,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
            {
                "sentence": "Third claim [3].",
                "citation_id": 3,
                "verdict": "unsupported",
                "confidence": 90,
                "reasoning": "no",
            },
        ]
        state = _make_state("q", report, sources, verification)
        result = RevisionAgent().run(state)

        assert "First claim [1]" in result["report"]
        assert "Second claim [2]" in result["report"]
        assert "Third claim" not in result["report"]
        # Surviving citations keep their ORIGINAL numbers -- no renumbering to [1][2]
        assert result["citations_used"] == [1, 2]

    def test_footer_rebuilt_without_orphaned_entries(self):
        report = "Only claim [1][2]."
        sources = {
            1: _source(1, "https://x.com/a", "A", "supports"),
            2: _source(2, "https://x.com/b", "B", "does not support"),
        }
        verification = [
            {
                "sentence": "Only claim [1][2].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
            {
                "sentence": "Only claim [1][2].",
                "citation_id": 2,
                "verdict": "unsupported",
                "confidence": 90,
                "reasoning": "no",
            },
        ]
        state = _make_state("q", report, sources, verification)
        result = RevisionAgent().run(state)

        assert "[1] A" in result["report"]
        assert "[2] B" not in result["report"]


class TestGroundingScore:
    def test_all_supported_scores_high(self):
        report = "Claim one [1]. Claim two [2]."
        sources = {1: _source(1, "https://x.com/a", "A", "s"), 2: _source(2, "https://x.com/b", "B", "s")}
        verification = [
            {
                "sentence": "Claim one [1].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
            {
                "sentence": "Claim two [2].",
                "citation_id": 2,
                "verdict": "supported",
                "confidence": 90,
                "reasoning": "ok",
            },
        ]
        state = _make_state("q", report, sources, verification)
        result = RevisionAgent().run(state)
        assert result["final_grounding_score"] == 100

    def test_no_verification_data_yields_none(self):
        report = "Claim one [1]."
        sources = {1: _source(1, "https://x.com/a", "A", "s")}
        state = _make_state("q", report, sources, citation_verification=[])
        result = RevisionAgent().run(state)
        assert result["final_grounding_score"] is None


class TestGracefulFallback:
    def test_no_report_short_circuits(self):
        state = _make_state("q", "", {})
        result = RevisionAgent().run(state)
        assert result["report_type"] is None
        assert result["final_grounding_score"] is None
        assert result["claims_removed"] == []

    def test_no_sources_short_circuits(self):
        state = _make_state("q", "Some report text.", {})
        result = RevisionAgent().run(state)
        assert result["final_grounding_score"] is None

    def test_internal_exception_falls_back_to_unchanged_report(self, monkeypatch):
        import agent.agents.revision as revision_mod

        def broken_detect(question, sources):
            raise RuntimeError("simulated bug")

        monkeypatch.setattr(revision_mod, "_detect_report_type", broken_detect)

        original_report = "Some claim [1]."
        sources = {1: _source(1, "https://x.com/a", "A", "s")}
        state = _make_state("q", original_report, sources, citation_verification=[])

        result = RevisionAgent().run(state)

        assert result["final_grounding_score"] is None
        assert any("unavailable" in entry.lower() for entry in result["log"])


class TestObservabilityIntegration:
    def test_is_agent_subclass_with_name(self):
        from agent.agents.base import Agent

        agent = RevisionAgent()
        assert isinstance(agent, Agent)
        assert agent.name == "revise"

    def test_registered_in_graph(self):
        from agent.graph import build_graph

        g = build_graph()
        assert "revise" in g.nodes


class TestLungCancerRegression:
    """
    Regression test using the lung-cancer-style report that originally
    motivated this milestone: a synthesis prompt forced into a rigid
    academic template fabricates an "Experimental Results" table (exact
    accuracy/AUC figures) that the actual retrieved source never
    reported. Confirms the revised report either removes those figures
    or clearly marks them unverified -- never leaves them looking
    authoritative and unflagged.
    """

    def test_fabricated_accuracy_and_auc_table_flagged(self):
        report = (
            "## 1. Introduction\n\n"
            "Lung cancer detection using deep learning has advanced rapidly [1].\n\n"
            "## 4. Model Architecture\n\n"
            "The hybrid CNN-LSTM model uses 12 convolutional layers and 3 LSTM layers [1].\n\n"
            "## 5. Dataset\n\n"
            "Trained on 50,000 CT scans from the LIDC-IDRI dataset [1].\n\n"
            "## 6. Experimental Results\n\n"
            "The model achieves 96.7% accuracy and an AUC of 0.98 [1].\n\n"
            "| Method | Accuracy | AUC |\n"
            "|-|-|-|\n"
            "| Hybrid CNN-LSTM | 96.7% | 0.98 |\n"
            "| Baseline CNN | 91.2% | 0.89 |\n\n"
            "---\n\n"
            "**Sources**\n\n"
            "[1] Deep Learning for Lung Cancer Detection — https://arxiv.org/abs/fake-lung-paper"
        )
        # The real source only supports the *existence* of the approach,
        # not any of the specific architecture/dataset-size/accuracy
        # figures the synthesis prompt's rigid template forced the model
        # to fabricate -- this is deliberately representative of what
        # Tavily snippets actually look like (short, general).
        sources = {
            1: _source(
                1,
                "https://arxiv.org/abs/fake-lung-paper",
                "Deep Learning for Lung Cancer Detection",
                "This paper discusses deep learning approaches to lung cancer detection from CT imaging.",
            )
        }
        citation_verification = [
            {
                "sentence": "Lung cancer detection using deep learning has advanced rapidly [1].",
                "citation_id": 1,
                "verdict": "supported",
                "confidence": 80,
                "reasoning": "General topic match.",
            },
            {
                "sentence": "The hybrid CNN-LSTM model uses 12 convolutional layers and 3 LSTM layers [1].",
                "citation_id": 1,
                "verdict": "unsupported",
                "confidence": 85,
                "reasoning": "No architecture details in source.",
            },
            {
                "sentence": "Trained on 50,000 CT scans from the LIDC-IDRI dataset [1].",
                "citation_id": 1,
                "verdict": "unsupported",
                "confidence": 85,
                "reasoning": "No dataset size mentioned in source.",
            },
            {
                "sentence": "The model achieves 96.7% accuracy and an AUC of 0.98 [1].",
                "citation_id": 1,
                "verdict": "unsupported",
                "confidence": 90,
                "reasoning": "No accuracy or AUC figures in source.",
            },
        ]

        state = _make_state(
            "lung cancer detection using hybrid CNN-LSTM deep learning",
            report,
            sources,
            citation_verification,
        )
        result = RevisionAgent().run(state)
        revised = result["report"]

        print("\n" + "=" * 70)
        print("LUNG CANCER REGRESSION — REVISED REPORT:")
        print("=" * 70)
        print(revised)
        print("report_type:", result["report_type"])
        print("claims_removed:", result["claims_removed"])
        print("final_grounding_score:", result["final_grounding_score"])

        # The fabricated architecture/dataset/accuracy claims must be
        # gone from the prose...
        assert "12 convolutional layers" not in revised
        assert "50,000 CT scans" not in revised
        assert "96.7% accuracy and an AUC of 0.98" not in revised

        # This source is a typical thin Tavily snippet -- no methodology
        # or metric keywords at all -- so report_type correctly resolves
        # to "General Research Answer", not "Experimental Study". That
        # means _prune_sections removes the entire fabricated Model
        # Architecture / Dataset / Experimental Results sections outright
        # (the "removed" branch of "removed or clearly marked as
        # unverified"), rather than leaving a marked-but-present table.
        # See test_fabricated_table_marked_when_report_type_is_experimental
        # below for the "clearly marked" branch, which applies when the
        # sources genuinely do support an experimental-study report type.
        assert result["report_type"] == "General Research Answer"
        assert "Model Architecture" not in revised
        assert "Dataset" not in revised
        assert "Experimental Results" not in revised
        assert "96.7%" not in revised
        assert "unverified" not in revised  # nothing left to mark -- it was pruned instead

        # The one genuinely supported sentence must survive untouched.
        assert "advanced rapidly [1]" in revised

        # Grounding score must reflect that most of the original claim
        # surface was unsupported.
        assert result["final_grounding_score"] is not None
        assert result["final_grounding_score"] < 50

    def test_fabricated_table_marked_when_report_type_is_experimental(self):
        """Companion to the test above: when the sources DO genuinely
        support an "Experimental Study" report type (methodology +
        metric keywords actually present), _prune_sections correctly
        keeps the Experimental Results section -- and the separate
        strict-grounding table pass is what catches a fabricated figure
        within it, marking rather than removing. Together these two
        tests cover both halves of "removed or clearly marked as
        unverified"."""
        report = (
            "## 6. Experimental Results\n\n"
            "The model achieves 96.7% accuracy and an AUC of 0.98 [1].\n\n"
            "| Method | Accuracy | AUC |\n"
            "|-|-|-|\n"
            "| Hybrid CNN-LSTM | 96.7% | 0.98 |\n"
            "| Baseline CNN | 91.2% | 0.89 |\n"
        )
        sources = {
            1: _source(
                1,
                "https://arxiv.org/abs/fake-lung-paper-2",
                "Hybrid CNN-LSTM Architecture for Lung Cancer Detection",
                "We propose a hybrid CNN-LSTM architecture and evaluate it using standard accuracy "
                "and AUC metrics on a benchmark dataset, achieving 89.4% accuracy.",
            )
        }
        citation_verification = [
            {
                "sentence": "The model achieves 96.7% accuracy and an AUC of 0.98 [1].",
                "citation_id": 1,
                "verdict": "partially_supported",
                "confidence": 60,
                "reasoning": "Source reports accuracy but a different figure (89.4%), and no AUC at all.",
            },
        ]
        state = _make_state(
            "hybrid CNN LSTM architecture for lung cancer detection accuracy metrics",
            report,
            sources,
            citation_verification,
        )
        result = RevisionAgent().run(state)
        revised = result["report"]

        assert result["report_type"] == "Experimental Study"
        assert "Experimental Results" in revised  # kept, not pruned
        assert "96.7%" in revised  # present...
        assert "unverified" in revised  # ...but the table row is clearly flagged
        assert "0.98" in revised

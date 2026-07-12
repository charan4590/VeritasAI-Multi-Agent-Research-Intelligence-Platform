"""
RevisionAgent — Phase 5.

Runs as the graph's new final node: validate -> fact_verify -> risk_analyze
-> revise -> END. Implements the "Revise -> Finalize" half of a
Generate -> Verify -> Revise -> Finalize pipeline (Generate = synthesize,
Verify = fact_verify + risk_analyze, both already existed).

Design decision worth stating up front: this agent makes ZERO LLM calls.
Every prior enrichment agent needed an LLM for the one piece that's
genuinely a natural-language task (entailment judgment for fact
verification, question phrasing for risk analysis's follow-ups). Revision
doesn't have an equivalent — having an LLM *rewrite* prose to fix
hallucinations risks introducing new hallucinations in the rewrite
itself, which is a strange way to build a hallucination-reduction
feature. Instead, every operation here is deterministic post-processing
over data the pipeline already computed: citation_verification verdicts
(Milestone 3) decide what to remove or annotate, source snippet text
decides what numeric claims are grounded, and report/source keyword
signals decide what sections belong in this report at all.

Failure contract (same as fact_verification.py / risk_analysis.py):
run() never raises. Any internal failure returns the report completely
unchanged with empty grounding-summary fields and a log line.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..reflection import ACADEMIC_REQUIRED_SIGNALS
from ..state import AgentState
from .base import Agent
from .citation import format_ieee_reference
from .fact_verification import (
    CITATION_RE,
    NO_CITATIONS_FOOTER_MARKER,
    SENTENCE_SPLIT_RE,
    SOURCES_FOOTER_MARKER,
    _extract_body,
)

logger = logging.getLogger(__name__)

# When enabled (default), any sentence containing a number (percentage,
# decimal, count) attached to a citation is checked against that
# citation's own source text -- a number that doesn't appear verbatim in
# any cited source gets marked [unverified] rather than trusted at face
# value. This is specifically what prevents a fabricated "94.2% accuracy"
# figure from surviving into a final report.
STRICT_GROUNDING_MODE = os.environ.get("STRICT_GROUNDING_MODE", "true").lower() == "true"

NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
SECTION_HEADER_RE = re.compile(r"^##\s+(?:\d+\.\s*)?(.+?)\s*$", re.MULTILINE)
TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:\-|]+\|$")

# Sections that only belong in a report where the sources actually
# demonstrate methodology + results -- not appropriate for a literature
# review, survey, comparative analysis, or general question, where the
# synthesis prompt's rigid academic template would otherwise force the
# model to invent an architecture or a results table with no basis.
EXPERIMENTAL_ONLY_SECTIONS = {
    "proposed method",
    "model architecture",
    "dataset",
    "datasets",
    "experimental results",
}

REPORT_TYPE_SIGNALS: Dict[str, List[str]] = {
    "Comparative Analysis": [
        "compare",
        "comparison",
        "versus",
        " vs ",
        "vs.",
        "difference between",
        "compared to",
    ],
    "Research Survey": ["survey", "overview of", "state of the art", "landscape", "current approaches"],
    "Literature Review": [
        "literature review",
        "review of",
        "prior work",
        "existing research",
        "past studies",
    ],
}


def _detect_report_type(question: str, sources: Dict[int, Any]) -> str:
    """
    Deterministic, keyword-based -- deliberately not an LLM call (same
    "heuristics before spending anything expensive" philosophy as
    reflection.py's smart_reflect). Checks the question first for an
    explicit signal (comparative/survey/review language), then falls back
    to asking whether the *sources themselves* actually demonstrate
    methodology + result signals -- reusing reflection.py's
    ACADEMIC_REQUIRED_SIGNALS unchanged rather than reinventing keyword
    lists. "Experimental Study" is the only type requiring that evidence
    bar; everything else defaults to "General Research Answer".
    """
    q = question.lower()
    for report_type, signals in REPORT_TYPE_SIGNALS.items():
        if any(sig in q for sig in signals):
            return report_type

    all_text = " ".join(f"{s.get('title', '')} {s.get('snippet', '')}".lower() for s in sources.values())
    has_methodology = any(kw in all_text for kw in ACADEMIC_REQUIRED_SIGNALS["methodology"])
    has_metrics = any(kw in all_text for kw in ACADEMIC_REQUIRED_SIGNALS["metrics"])
    if has_methodology and has_metrics:
        return "Experimental Study"
    return "General Research Answer"


def _numbers_grounded(sentence: str, source_text: str) -> bool:
    """True if every number-like token in `sentence` appears verbatim
    somewhere in `source_text`. No numbers at all -> trivially grounded
    (nothing to check).

    Citation markers are stripped before scanning: a bare NUMBER_RE would
    otherwise match the digit inside "[1]" as if it were itself a
    numeric claim needing grounding, which it isn't -- it's a citation
    id, not a statistic.
    """
    numbers = NUMBER_RE.findall(CITATION_RE.sub("", sentence))
    if not numbers:
        return True
    return all(num in source_text for num in numbers)


class RevisionAgent(Agent):
    """Deterministic grounding/self-correction pass over the already-
    validated, already-verified report. See module docstring for the
    full design rationale and failure contract."""

    name = "revise"

    def trace_inputs(self, state: AgentState):
        return {
            "citation_verification_count": len(state.get("citation_verification", []) or []),
            "strict_grounding": STRICT_GROUNDING_MODE,
        }

    def run(self, state: AgentState) -> dict:
        report = state.get("report", "")
        sources = state.get("sources", {})
        citation_verification = state.get("citation_verification") or []

        empty_result = {
            "report_type": None,
            "claims_removed": [],
            "claims_rewritten": [],
            "unsupported_claims": [],
            "final_grounding_score": None,
        }

        if not report or not sources:
            return {
                **empty_result,
                "log": state["log"] + ["Revision: no report/sources to ground"],
            }

        try:
            report_type = _detect_report_type(state.get("question", ""), sources)

            body, footer_used_ids = self._split_body_and_footer(report, sources)

            body, claims_removed, claims_rewritten, unsupported_claims = self._revise_claims(
                body, citation_verification, sources
            )

            if STRICT_GROUNDING_MODE:
                body, marked_count = self._ground_numeric_claims(body, citation_verification, sources)
                body, table_marked_count = self._ground_tables(body, sources)
            else:
                marked_count = 0
                table_marked_count = 0

            body = self._prune_sections(body, report_type, state.get("question", ""))

            final_citations_used = sorted({int(m) for m in CITATION_RE.findall(body)} & set(sources.keys()))
            revised_report = self._rebuild_footer(body, final_citations_used, sources)

            final_grounding_score = self._compute_grounding_score(
                citation_verification, claims_removed, len(claims_rewritten)
            )

            log_msg = (
                f"Revision: report_type={report_type}, "
                f"{len(claims_removed)} claim(s) removed, "
                f"{len(claims_rewritten)} marked partially-supported, "
                f"{marked_count + table_marked_count} numeric claim(s) flagged unverified"
                + (
                    f", grounding score {final_grounding_score}/100"
                    if final_grounding_score is not None
                    else ""
                )
            )
            logger.info(f"[revise] {log_msg}")

            return {
                "report": revised_report,
                "citations_used": final_citations_used,
                "report_type": report_type,
                "claims_removed": claims_removed,
                "claims_rewritten": claims_rewritten,
                "unsupported_claims": unsupported_claims,
                "final_grounding_score": final_grounding_score,
                "log": state["log"] + [log_msg],
            }

        except Exception as exc:
            logger.warning(f"Revision failed ({exc}) — report left unchanged")
            return {
                **empty_result,
                "log": state["log"]
                + ["Revision unavailable — report unchanged (prior validation still applies)"],
            }

    # -----------------------------------------------------------------
    # Steps
    # -----------------------------------------------------------------

    @staticmethod
    def _split_body_and_footer(report: str, sources: Dict[int, Any]) -> Tuple[str, set]:
        body = _extract_body(report)
        used_ids = {int(m) for m in CITATION_RE.findall(body)} & set(sources.keys())
        return body, used_ids

    @staticmethod
    def _revise_claims(
        body: str, citation_verification: List[Dict[str, Any]], sources: Dict[int, Any]
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Removes sentences where EVERY cited id is 'unsupported' (including
        the contradiction case -- FactVerificationAgent's own verdict
        definition already folds "source contradicts the claim" into
        'unsupported', so no separate contradiction handling is needed
        here). Sentences with a mix of verdicts keep their supported
        citation markers and only strip the unsupported ones (same
        surgical approach CitationAgent already uses for hallucinated
        ids). 'partially_supported' sentences are kept but annotated.

        Operates paragraph-by-paragraph (split on blank lines, i.e. "\\n\\n"
        -- markdown's own block separator) rather than sentence-splitting
        the whole body and rejoining with " ".join(): the latter would
        collapse headers, table rows, and paragraph breaks onto a single
        line. Header and table paragraphs are passed through completely
        unchanged (sentence-level revision only makes sense for prose,
        and FactVerificationAgent's own claim extraction never produced
        verdicts for header/table text in the first place, since it uses
        the same sentence regex on the same body).
        """
        if not citation_verification:
            return body, [], [], []

        verdict_by_key: Dict[Tuple[str, int], str] = {
            (v["sentence"], v["citation_id"]): v.get("verdict", "cannot_determine")
            for v in citation_verification
        }

        claims_removed: List[str] = []
        claims_rewritten: List[str] = []
        unsupported_claims: List[str] = []
        out_paragraphs: List[str] = []

        for para in body.split("\n\n"):
            if not para.strip() or para.lstrip().startswith(("|", "#")):
                out_paragraphs.append(para)
                continue

            sentences = SENTENCE_SPLIT_RE.split(para.strip())
            kept_sentences: List[str] = []

            for sentence in sentences:
                if not sentence.strip():
                    continue
                cited_ids = [int(m) for m in CITATION_RE.findall(sentence)]
                verdicts = {
                    cid: verdict_by_key.get((sentence, cid))
                    for cid in cited_ids
                    if (sentence, cid) in verdict_by_key
                }

                if not verdicts:
                    kept_sentences.append(sentence)
                    continue

                if all(v == "unsupported" for v in verdicts.values()):
                    claims_removed.append(sentence)
                    unsupported_claims.append(sentence)
                    continue

                revised_sentence = sentence
                for cid, verdict in verdicts.items():
                    if verdict == "unsupported":
                        revised_sentence = re.sub(rf"\[{cid}\]", "", revised_sentence)
                        unsupported_claims.append(sentence)

                if any(v == "partially_supported" for v in verdicts.values()):
                    revised_sentence = revised_sentence.rstrip() + " *(partially supported by cited source)*"
                    claims_rewritten.append(sentence)

                kept_sentences.append(revised_sentence)

            out_paragraphs.append(" ".join(kept_sentences))

        return "\n\n".join(out_paragraphs), claims_removed, claims_rewritten, unsupported_claims

    @staticmethod
    def _ground_numeric_claims(
        body: str, citation_verification: List[Dict[str, Any]], sources: Dict[int, Any]
    ) -> Tuple[str, int]:
        """Strict grounding mode: for each remaining cited sentence with a
        numeric claim, verify the number appears verbatim in that
        citation's source text. Ungrounded numbers get an inline
        [unverified] marker rather than being silently trusted or
        silently deleted -- deletion could remove non-numeric context the
        sentence also carries. Same paragraph-preserving approach as
        _revise_claims -- see that method's docstring for why."""
        marked = 0
        out_paragraphs: List[str] = []

        for para in body.split("\n\n"):
            if not para.strip() or para.lstrip().startswith(("|", "#")):
                out_paragraphs.append(para)
                continue

            sentences = SENTENCE_SPLIT_RE.split(para.strip())
            out_sentences = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                cited_ids = [int(m) for m in CITATION_RE.findall(sentence)]
                if cited_ids and NUMBER_RE.search(CITATION_RE.sub("", sentence)):
                    source_text = " ".join(
                        sources[cid].get("snippet", "") for cid in cited_ids if cid in sources
                    )
                    if not _numbers_grounded(sentence, source_text):
                        sentence = sentence.rstrip() + " **[unverified — figure not found in source text]**"
                        marked += 1
                out_sentences.append(sentence)
            out_paragraphs.append(" ".join(out_sentences))

        return "\n\n".join(out_paragraphs), marked

    @staticmethod
    def _ground_tables(body: str, sources: Dict[int, Any]) -> Tuple[str, int]:
        """Same strict-grounding check applied to markdown table rows,
        checked against ALL source text (a table row doesn't reliably
        carry its own citation marker the way a sentence does)."""
        all_source_text = " ".join(s.get("snippet", "") for s in sources.values())
        lines = body.split("\n")
        marked = 0
        out_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and not TABLE_SEPARATOR_RE.match(stripped):
                numbers = NUMBER_RE.findall(stripped)
                if numbers and not all(n in all_source_text for n in numbers):
                    line = line.rstrip() + "  *(unverified — figures not found in source text)*"
                    marked += 1
            out_lines.append(line)
        return "\n".join(out_lines), marked

    @staticmethod
    def _prune_sections(body: str, report_type: str, question: str) -> str:
        """Removes Proposed Method / Model Architecture / Dataset /
        Experimental Results sections when report_type isn't
        "Experimental Study" (which already required methodology+metric
        evidence in the sources to be assigned) and the question didn't
        explicitly ask about them."""
        if report_type == "Experimental Study":
            return body

        q = question.lower()
        headers = list(SECTION_HEADER_RE.finditer(body))
        if not headers:
            return body

        keep_ranges: List[Tuple[int, int]] = []
        preamble_end = headers[0].start()
        if preamble_end > 0:
            keep_ranges.append((0, preamble_end))

        for i, match in enumerate(headers):
            section_title = match.group(1).strip().lower()
            section_start = match.start()
            section_end = headers[i + 1].start() if i + 1 < len(headers) else len(body)

            should_prune = section_title in EXPERIMENTAL_ONLY_SECTIONS and not any(
                kw in q for kw in section_title.split()
            )
            if not should_prune:
                keep_ranges.append((section_start, section_end))

        return "".join(body[start:end] for start, end in keep_ranges).strip()

    @staticmethod
    def _rebuild_footer(body: str, final_citations_used: List[int], sources: Dict[int, Any]) -> str:
        if not final_citations_used:
            return body + NO_CITATIONS_FOOTER_MARKER

        # Same defensive URL dedup as CitationAgent's initial References
        # section (see citation.py) -- belt-and-suspenders here since
        # this is a second, independent place a References list gets
        # built, after revision may have changed which citations survive.
        seen_urls = set()
        deduped_ids = []
        for i in final_citations_used:
            url = sources[i].get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped_ids.append(i)

        refs = "\n".join(format_ieee_reference(i, sources[i]) for i in deduped_ids)
        return body + f"{SOURCES_FOOTER_MARKER}\n\n{refs}"

    @staticmethod
    def _compute_grounding_score(
        citation_verification: List[Dict[str, Any]], claims_removed: List[str], rewritten_count: int
    ) -> Optional[int]:
        """
        Fraction of the ORIGINAL claim surface that is now cleanly
        grounded, after revision: removed (unsupported) claims are gone
        from the report entirely, so they no longer count against it;
        remaining supported claims count fully; remaining
        partially-supported (rewritten/annotated) claims count at half
        weight, matching risk_analysis.py's existing VERDICT_WEIGHT
        convention for "partially_supported" being worth 60/100 of a
        fully "supported" claim -- here simplified to 0.5 since this is
        a post-revision surface-level score, not a full re-verification.
        """
        if not citation_verification:
            return None
        total = len(citation_verification)
        removed = len(claims_removed)
        remaining = total - removed
        if remaining <= 0:
            return 100  # nothing left ungrounded because nothing supportable remained
        supported_weight = (remaining - rewritten_count) + (rewritten_count * 0.5)
        return int(round((supported_weight / total) * 100))

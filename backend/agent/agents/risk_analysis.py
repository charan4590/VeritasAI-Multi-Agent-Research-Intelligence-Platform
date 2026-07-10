"""
RiskAnalysisAgent — Phase 3 Milestone 4.

Runs as the graph's new final node: validate -> fact_verify -> risk_analyze
-> END. Evaluates the *reliability* of the finished report — contradictions,
weak evidence, missing perspectives, outdated information, and source-
quality concerns — as a distinct concern from CitationAgent (existence)
and FactVerificationAgent (per-claim entailment).

Design principle: almost everything this agent produces is computed from
data already sitting in `state`, reusing existing modules rather than
recomputing anything:
  - reflection._check_contradictions() / _count_domains()  (unchanged,
    imported directly — the exact same heuristics smart_reflect() already
    uses mid-run, reapplied here to the final source set)
  - credibility.score_url()                                (per-source
    credibility, already used everywhere else in the pipeline)
  - state["citation_verification"]                         (Milestone 3 —
    unsupported/partially_supported verdicts feed conflicting_claims /
    evidence_gaps directly, no new entailment logic needed)
  - state["retrieved_chunks"]                               (RAG, Phase 1)

Because of this, risk_score / risk_level / identified_risks / evidence_gaps
/ conflicting_claims are ALL pure heuristics — deterministic, instant, and
have zero LLM dependency, so they can never silently degrade. Only
recommended_follow_up_questions involves an LLM call (phrasing targeted
questions is genuinely a natural-language task), and that call is cached
and has a templated fallback derived directly from evidence_gaps — see
_fallback_followups(). This mirrors followup.py's existing
generate_follow_ups() pattern (LLM call -> regex JSON extraction ->
templated fallback) almost exactly, on purpose.

Naming note (read this before comparing to evaluator.py): risk_score here
is HIGHER = MORE RISK (0 = clean, 100 = high risk) so that risk_level
(Low/Medium/High) reads naturally against it. This is the OPPOSITE polarity
of evaluator.py's hallucination_risk_score, where higher is better (it's
really a "citation health" score despite its name). Both scores coexist in
the same report/API response — this divergence is deliberate, not a bug.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..cache import get_risk_cache
from ..credibility import score_url
from ..llm import get_llm
from ..reflection import _check_contradictions, _count_domains
from ..state import AgentState
from .base import Agent

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d+)\]")
YEAR_RE = re.compile(r"\b(19[8-9]\d|20[0-3]\d)\b")

# Thresholds checked high-to-low; first match wins.
_RISK_LEVEL_THRESHOLDS = [(67, "High"), (34, "Medium"), (0, "Low")]


def _risk_level(score: int) -> str:
    for threshold, label in _RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Low"


# ---------------------------------------------------------------------------
# Heuristic signals — each is independently testable and has no LLM dependency
# ---------------------------------------------------------------------------


def _low_credibility_ratio(sources: Dict) -> Tuple[float, int, int]:
    if not sources:
        return 0.0, 0, 0
    total = len(sources)
    low = sum(1 for s in sources.values() if score_url(s.get("url", "")) < 55)
    return low / total, low, total


def _citation_frequency(state: AgentState) -> Dict[int, int]:
    """
    How many times each citation id is actually used in the report.
    Prefers citation_verification (Milestone 3 — one record per claim) so
    this stays accurate even if a citation appears in multiple sentences;
    falls back to counting [n] occurrences in the report body directly
    (same regex FactVerificationAgent's own claim extraction uses) so
    single-source-dominance detection still works even if fact
    verification itself fell back to empty.
    """
    verification = state.get("citation_verification") or []
    counts: Dict[int, int] = {}
    if verification:
        for v in verification:
            cid = v.get("citation_id")
            if cid is not None:
                counts[cid] = counts.get(cid, 0) + 1
        return counts

    report = state.get("report", "")
    for m in CITATION_RE.findall(report):
        cid = int(m)
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def _single_source_dominance(citation_counts: Dict[int, int]) -> Tuple[bool, Optional[int], float]:
    """Flags when one citation accounts for more than half of all cited
    claims. Requires at least 2 distinct citations — a report with only
    one source overall isn't "dominance", it's just a single-source report
    (already caught separately by the domain-diversity check)."""
    total = sum(citation_counts.values())
    if total == 0 or len(citation_counts) < 2:
        return False, None, 0.0
    top_id, top_count = max(citation_counts.items(), key=lambda kv: kv[1])
    share = top_count / total
    return share > 0.5, top_id, share


def _recency_signal(sources: Dict) -> Tuple[bool, Optional[int]]:
    """
    Returns (has_recent_evidence, most_recent_year_found). A deliberately
    cheap heuristic — scans title+snippet text for 4-digit years, since no
    source in this pipeline (Tavily results or PDF chunks) currently
    exposes a structured publish date. "Recent" = within the last 5 years.
    """
    years: List[int] = []
    for s in sources.values():
        text = f"{s.get('title', '')} {s.get('snippet', '')}"
        years.extend(int(y) for y in YEAR_RE.findall(text))
    if not years:
        return False, None
    most_recent = max(years)
    current_year = datetime.now().year
    return (current_year - most_recent) <= 5, most_recent


def _compute_risk_signals(state: AgentState) -> Dict[str, Any]:
    """Pure function of `state` — no LLM calls, no I/O, fully deterministic
    and independently testable without mocking anything."""
    question = state.get("question", "")
    sources = state.get("sources", {})
    citation_verification = state.get("citation_verification") or []
    retrieved_chunks = state.get("retrieved_chunks") or []

    identified_risks: List[str] = []
    evidence_gaps: List[str] = []
    conflicting_claims: List[str] = []
    score = 0

    # --- source-quality concerns ---
    low_ratio, low_count, total = _low_credibility_ratio(sources)
    if total and low_ratio > 0.4:
        score += 35 if low_ratio > 0.7 else 20
        identified_risks.append(f"{low_count} of {total} sources have low or unverified credibility.")

    # --- contradictions (reflection.py, reused unchanged) ---
    if sources and _check_contradictions(sources, question):
        score += 20
        conflicting_claims.append(
            "Sources contain opposing language on the same topic (e.g. "
            "'effective' vs 'ineffective', 'safe' vs 'dangerous') — treat "
            "the report's conclusions with caution."
        )

    # --- missing perspectives (domain diversity, reflection.py reused) ---
    domain_count = _count_domains(sources) if sources else 0
    if sources and domain_count < 2:
        score += 15
        evidence_gaps.append(
            f"All sources come from {domain_count} domain(s) — limited " "diversity of perspective."
        )

    # --- single-source dominance ---
    citation_counts = _citation_frequency(state)
    dominant, dominant_id, share = _single_source_dominance(citation_counts)
    if dominant:
        score += 15
        identified_risks.append(
            f"Citation [{dominant_id}] accounts for {int(share * 100)}% of "
            "all cited claims — the report leans heavily on a single source."
        )

    # --- citation_verification (Milestone 3, reused directly) ---
    unsupported = [v for v in citation_verification if v.get("verdict") == "unsupported"]
    partial = [v for v in citation_verification if v.get("verdict") == "partially_supported"]
    if unsupported:
        score += min(30, 10 * len(unsupported))
        for v in unsupported[:3]:
            conflicting_claims.append(
                f"Citation [{v['citation_id']}] does not appear to support "
                f"its claim: {v.get('reasoning', '')}"
            )
    if partial:
        score += min(15, 5 * len(partial))
        for v in partial[:3]:
            evidence_gaps.append(
                f"Citation [{v['citation_id']}] only partially supports its "
                f"claim: {v.get('reasoning', '')}"
            )

    # --- evidence volume (RAG, reused) ---
    if sources and len(retrieved_chunks) < 3:
        score += 10
        evidence_gaps.append(
            f"Only {len(retrieved_chunks)} relevant passage(s) were retrieved "
            "for synthesis — the evidence base may be thin."
        )

    # --- recency ---
    if sources:
        has_recent, most_recent_year = _recency_signal(sources)
        if most_recent_year is None:
            score += 5
            evidence_gaps.append("No publication dates found in any source — recency cannot be assessed.")
        elif not has_recent:
            score += 10
            identified_risks.append(
                f"The most recent evidence found appears to be from "
                f"{most_recent_year} — findings may be outdated."
            )

    score = min(100, score)
    return {
        "risk_score": score,
        "risk_level": _risk_level(score),
        "identified_risks": identified_risks,
        "evidence_gaps": evidence_gaps,
        "conflicting_claims": conflicting_claims,
    }


# ---------------------------------------------------------------------------
# Follow-up questions — the one part of this agent that needs an LLM
# ---------------------------------------------------------------------------


def _build_followup_prompt(question: str, signals: Dict[str, Any]) -> Tuple[str, str]:
    system = (
        "You suggest targeted research follow-up questions given specific "
        "reliability concerns found in a report. Respond ONLY with a JSON "
        "array of 3-5 short questions, each one directly addressing one of "
        "the concerns listed below — not generic questions about the topic.\n"
        'Example: ["What peer-reviewed studies confirm X?", "..."]'
    )
    concerns = signals["identified_risks"] + signals["evidence_gaps"] + signals["conflicting_claims"]
    concerns_text = "\n".join(f"- {c}" for c in concerns)
    human = f"Original question: {question}\n\nIdentified concerns:\n{concerns_text}"
    return system, human


def _fallback_followups(signals: Dict[str, Any]) -> List[str]:
    """Used when the LLM call fails. Templated directly from evidence_gaps
    (not a generic canned list) so it's still relevant to what was
    actually found, matching followup.py's existing fallback philosophy."""
    gaps = signals["evidence_gaps"][:3]
    return [f"What additional sources address: {g.rstrip('.')}?" for g in gaps]


class RiskAnalysisAgent(Agent):
    """Reliability assessment layered on top of everything the pipeline has
    already produced — CitationAgent's validated report, FactVerification's
    per-claim verdicts, RAG's retrieved evidence, and each source's
    credibility. See module docstring for the full design rationale."""

    name = "risk_analyze"

    def trace_inputs(self, state: AgentState):
        return {
            "source_count": len(state.get("sources", {})),
            "citations_used": len(state.get("citations_used", [])),
        }

    def run(self, state: AgentState) -> dict:
        sources = state.get("sources", {})
        report = state.get("report", "")

        empty_result = {
            "risk_score": None,
            "risk_level": None,
            "identified_risks": [],
            "evidence_gaps": [],
            "conflicting_claims": [],
            "recommended_follow_up_questions": [],
        }

        if not sources or not report:
            return {
                **empty_result,
                "log": state["log"] + ["Risk analysis: no sources/report to assess"],
            }

        try:
            signals = _compute_risk_signals(state)
        except Exception as exc:
            logger.warning(f"Risk analysis: signal computation failed, skipping: {exc}")
            return {
                **empty_result,
                "log": state["log"] + ["Risk analysis unavailable (signal computation failed)"],
            }

        followups = self._get_followups(state["question"], signals)

        log_msg = (
            f"Risk analysis: score {signals['risk_score']}/100 ({signals['risk_level']}) — "
            f"{len(signals['identified_risks'])} risk(s), "
            f"{len(signals['evidence_gaps'])} gap(s), "
            f"{len(signals['conflicting_claims'])} conflict(s)"
        )
        logger.info(f"[risk_analyze] {log_msg}")

        return {
            **signals,
            "recommended_follow_up_questions": followups,
            "log": state["log"] + [log_msg],
        }

    def _get_followups(self, question: str, signals: Dict[str, Any]) -> List[str]:
        all_concerns = signals["identified_risks"] + signals["evidence_gaps"] + signals["conflicting_claims"]
        if not all_concerns:
            # Nothing concerning found — no follow-ups needed, and no LLM
            # call spent confirming that.
            return []

        cache = get_risk_cache()
        cache_key = question + "|" + "|".join(sorted(all_concerns))
        cached, hit = cache.get(cache_key)
        if hit:
            return cached

        try:
            system, human = _build_followup_prompt(question, signals)
            llm = get_llm(temperature=0.3)
            response = llm.invoke([("system", system), ("human", human)])
            match = re.search(r"\[[\s\S]*\]", response.content)
            if not match:
                raise ValueError("No JSON array found in verifier response")
            questions = json.loads(match.group(0))
            questions = [q for q in questions if isinstance(q, str)][:5]
            if not questions:
                raise ValueError("LLM returned an empty questions list")
            cache.set(cache_key, questions)
            return questions
        except Exception as exc:
            logger.warning(f"Risk analysis: follow-up generation failed, using fallback: {exc}")
            return _fallback_followups(signals)

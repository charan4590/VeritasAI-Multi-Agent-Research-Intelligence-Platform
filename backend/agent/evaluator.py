"""
Phase 3: Evaluation Framework
===============================
Architecture decision: hybrid evaluation — fast heuristic scores (code-based,
instant, free) combined with one LLM eval for relevance. This keeps cost low
while giving meaningful signal.

Four scores (each 0–100):
  1. relevance_score     — Does the report answer the question? (LLM judge)
  2. citation_score      — What % of paragraphs have at least one citation?
  3. source_diversity    — How many unique domains are cited?
  4. hallucination_risk  — Inverse of verified-citation ratio (code-based)

Overall: weighted average stored alongside every session.

Why not pure LLM eval? It's expensive, slow, and circular (asking the same
model to evaluate its own output is like marking your own homework). The
heuristic scores are fast, deterministic, and catch the most common failures.
"""

import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from .llm import get_llm

# ---------------------------------------------------------------------------
# Heuristic evaluators (instant, no LLM call)
# ---------------------------------------------------------------------------

CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_score(report: str, citations_used: List[int]) -> Tuple[int, str]:
    """
    What fraction of paragraphs contain at least one citation?
    Score: 0–100. Rationale: a well-grounded report cites every claim.
    """
    if not citations_used:
        return 0, "No citations found in report"

    paragraphs = [p.strip() for p in report.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return 50, "Could not parse paragraphs"

    cited = sum(1 for p in paragraphs if CITATION_RE.search(p))
    score = int((cited / len(paragraphs)) * 100)
    return score, f"{cited}/{len(paragraphs)} paragraphs have citations"


def source_diversity_score(sources: Dict, citations_used: List[int]) -> Tuple[int, str]:
    """
    How many unique domains appear in the cited sources?
    Score: 0–100 (5+ unique domains = 100, 1 domain = 20).
    Rationale: a research report should draw from multiple independent sources.
    """
    if not citations_used or not sources:
        return 0, "No cited sources"

    cited_sources = [sources[i] for i in citations_used if i in sources]
    domains = set()
    for s in cited_sources:
        url = s.get("url", "") if isinstance(s, dict) else ""
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            if domain:
                domains.add(domain)
        except Exception:
            pass

    count = len(domains)
    score = min(100, count * 20)  # 5 domains = 100
    return score, f"{count} unique domains cited"


def hallucination_risk_score(report: str, sources: Dict, citations_used: List[int]) -> Tuple[int, str]:
    """
    What fraction of citation markers in the report are valid?
    Score: 100 = all citations verified, 0 = all citations hallucinated.

    This catches the most dangerous failure mode in research agents:
    the model inventing source numbers that don't exist.
    """
    all_cited = {int(m) for m in CITATION_RE.findall(report)}
    if not all_cited:
        return 50, "No citations to verify"

    valid_ids = set(sources.keys()) if sources else set(citations_used)
    verified = all_cited & valid_ids
    hallucinated = all_cited - valid_ids

    ratio = len(verified) / len(all_cited)
    score = int(ratio * 100)
    msg = f"{len(verified)} verified, {len(hallucinated)} hallucinated"
    return score, msg


# ---------------------------------------------------------------------------
# LLM evaluator (one call, focused prompt)
# ---------------------------------------------------------------------------


def relevance_score(question: str, report: str) -> Tuple[int, str]:
    """
    Ask the LLM to score how well the report answers the question.
    Uses a structured 1–10 scale then maps to 0–100.

    Why a separate LLM call: the synthesis LLM already wrote the report
    so we need a "judge" perspective. In production you'd use a different,
    stronger model (e.g. GPT-4o judging a llama output).
    """
    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(
            [
                (
                    "system",
                    "You are an expert evaluator. Score how well the report answers "
                    "the question on a scale of 1 to 10. Be strict.\n"
                    "Respond with ONLY a JSON object: "
                    '{"score": 7, "reason": "one sentence"}',
                ),
                ("human", f"Question: {question}\n\nReport (first 1000 chars):\n{report[:1000]}"),
            ]
        )
        import json
        import re as _re

        text = response.content.strip()
        match = _re.search(r"\{[\s\S]*?\}", text)
        if match:
            data = json.loads(match.group(0))
            raw = int(data.get("score", 5))
            score = min(100, max(0, raw * 10))
            return score, data.get("reason", "")
    except Exception as exc:
        print(f"[eval] relevance scoring failed: {exc}")
    return 50, "Could not evaluate (LLM unavailable)"


# ---------------------------------------------------------------------------
# Master evaluator
# ---------------------------------------------------------------------------


def evaluate_report(
    question: str,
    report: str,
    sources: Dict,
    citations_used: List[int],
) -> Dict:
    """
    Run all four evaluators and compute a weighted overall score.

    Weights chosen to match what hiring managers care about:
    - Relevance: most important (does it answer the question?)
    - Citations: second most (are claims supported?)
    - Diversity: third (is it well-researched?)
    - Hallucination: always high stakes
    """
    rel_score, rel_reason = relevance_score(question, report)
    cit_score, cit_reason = citation_score(report, citations_used)
    div_score, div_reason = source_diversity_score(sources, citations_used)
    hal_score, hal_reason = hallucination_risk_score(report, sources, citations_used)

    # Weighted average: relevance 40%, citations 25%, diversity 20%, hallucination 15%
    overall = int(rel_score * 0.40 + cit_score * 0.25 + div_score * 0.20 + hal_score * 0.15)

    return {
        "overall_score": overall,
        "relevance_score": rel_score,
        "relevance_reason": rel_reason,
        "citation_score": cit_score,
        "citation_reason": cit_reason,
        "diversity_score": div_score,
        "diversity_reason": div_reason,
        "hallucination_risk_score": hal_score,
        "hallucination_reason": hal_reason,
        "grade": _grade(overall),
    }


def _grade(score: int) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"

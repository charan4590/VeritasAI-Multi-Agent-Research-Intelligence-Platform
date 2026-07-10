"""
Improvement 2: Smarter Reflection
====================================
Old reflection: "Do we have enough sources?" (single LLM judgment)
New reflection: Multi-signal analysis before asking LLM

Checks in order:
  1. Source count          — hard minimum (< 3 = always search more)
  2. Source diversity      — single domain = likely biased, search more
  3. Subtopic coverage     — extract key subtopics, check if covered
  4. Contradiction check   — conflicting facts = search authoritative source
  5. LLM judgment          — only if above checks pass

This makes the agent genuinely smarter — it fails fast on obvious gaps
without wasting an LLM call, and catches subtle gaps the LLM might miss.
"""

import json
import logging
import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from .llm import get_llm
from .state import ReflectionDecision

logger = logging.getLogger(__name__)

MIN_SOURCES = int(3)
MIN_DOMAINS = int(2)


def _count_domains(sources: Dict) -> int:
    domains = set()
    for s in sources.values():
        try:
            domain = urlparse(s.get("url", "")).netloc.replace("www.", "")
            if domain:
                domains.add(domain)
        except Exception:
            pass
    return len(domains)


def _extract_subtopics(question: str) -> List[str]:
    """
    Extract key subtopics from the question using the LLM.
    These become a checklist for coverage verification.
    """
    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(
            [
                (
                    "system",
                    "Extract 3-5 key subtopics or aspects that a complete answer to this "
                    "question must cover. Respond ONLY with a JSON array of short strings.\n"
                    'Example: ["definition", "mechanism", "applications", "limitations"]',
                ),
                ("human", f"Question: {question}"),
            ]
        )
        text = response.content.strip()
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            return json.loads(match.group(0))
    except Exception as exc:
        logger.warning(f"Subtopic extraction failed: {exc}")
    return []


def _check_subtopic_coverage(subtopics: List[str], sources: Dict) -> Tuple[bool, List[str]]:
    """
    Check which subtopics are covered by the collected sources.
    Returns (all_covered, missing_subtopics).
    """
    if not subtopics:
        return True, []

    all_text = " ".join((s.get("title", "") + " " + s.get("snippet", "")).lower() for s in sources.values())

    missing = []
    for topic in subtopics:
        # Simple keyword check — topic words appear in source text
        topic_words = topic.lower().split()
        if not any(word in all_text for word in topic_words if len(word) > 3):
            missing.append(topic)

    return len(missing) == 0, missing


def _check_contradictions(sources: Dict, question: str) -> bool:
    """
    Quick heuristic contradiction check.
    If sources use strongly opposing language about the same topic,
    flag for additional authoritative source search.
    """
    if len(sources) < 3:
        return False

    snippets = [s.get("snippet", "") for s in list(sources.values())[:6]]
    combined = " ".join(snippets).lower()

    # Contradiction signal pairs
    signal_pairs = [
        ("effective", "ineffective"),
        ("safe", "dangerous"),
        ("increases", "decreases"),
        ("beneficial", "harmful"),
        ("proven", "unproven"),
        ("supports", "contradicts"),
    ]

    contradictions_found = sum(1 for pos, neg in signal_pairs if pos in combined and neg in combined)
    return contradictions_found >= 2


# Academic research coverage signals
ACADEMIC_REQUIRED_SIGNALS = {
    "methodology": [
        "architecture",
        "model",
        "method",
        "approach",
        "algorithm",
        "network",
        "layer",
        "cnn",
        "lstm",
        "transformer",
        "hybrid",
        "proposed",
        "framework",
        "pipeline",
    ],
    "dataset": [
        "dataset",
        "data",
        "samples",
        "images",
        "patients",
        "training",
        "testing",
        "validation",
        "benchmark",
        "corpus",
        "collected",
    ],
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "roc",
        "sensitivity",
        "specificity",
        "dice",
        "iou",
        "mae",
        "rmse",
        "loss",
        "performance",
    ],
    "results": [
        "achieved",
        "outperforms",
        "compared",
        "baseline",
        "improvement",
        "state-of-the-art",
        "sota",
        "experiment",
        "result",
        "score",
    ],
}


def _is_academic_query(question: str) -> bool:
    """Check if this is an academic/research query needing technical coverage."""
    q = question.lower()
    academic_terms = [
        "deep learning",
        "neural network",
        "model",
        "architecture",
        "dataset",
        "classification",
        "detection",
        "segmentation",
        "hybrid",
        "novel",
        "proposed",
        "method",
        "algorithm",
        "accuracy",
        "benchmark",
        "cancer",
        "medical",
        "imaging",
        "diagnosis",
    ]
    return sum(1 for t in academic_terms if t in q) >= 2


def _check_academic_coverage(sources: Dict) -> Tuple[bool, List[str]]:
    """
    For academic queries, verify coverage of methodology, datasets,
    metrics, and results — not just source count and diversity.
    Returns (sufficient_coverage, missing_sections).
    """
    all_text = " ".join((s.get("title", "") + " " + s.get("snippet", "")).lower() for s in sources.values())

    missing = []
    for section, keywords in ACADEMIC_REQUIRED_SIGNALS.items():
        covered = any(kw in all_text for kw in keywords)
        if not covered:
            missing.append(section)

    return len(missing) == 0, missing


def smart_reflect(
    question: str,
    sources: Dict,
    current_round: int,
    max_rounds: int,
) -> ReflectionDecision:
    """
    Multi-signal reflection that replaces the naive LLM-only approach.

    Returns a ReflectionDecision with:
      - sufficient: bool
      - follow_up_queries: targeted queries for specific gaps
      - reasoning: explanation of why more search is needed
    """

    # --- Hard check 1: minimum source count ---
    if len(sources) < MIN_SOURCES and current_round < max_rounds:
        return ReflectionDecision(
            sufficient=False,
            follow_up_queries=[
                f"{question} research findings",
                f"{question} recent studies",
            ],
            reasoning=f"Only {len(sources)} sources found — minimum is {MIN_SOURCES}.",
        )

    # --- Hard check 2: source diversity ---
    domain_count = _count_domains(sources)
    if domain_count < MIN_DOMAINS and current_round < max_rounds:
        return ReflectionDecision(
            sufficient=False,
            follow_up_queries=[f"{question} alternative perspectives"],
            reasoning=f"All sources from {domain_count} domain(s) — need diverse perspectives.",
        )

    # --- Check 3: subtopic coverage ---
    subtopics = _extract_subtopics(question)
    all_covered, missing = _check_subtopic_coverage(subtopics, sources)

    if not all_covered and current_round < max_rounds:
        follow_ups = [f"{question} {topic}" for topic in missing[:2]]
        return ReflectionDecision(
            sufficient=False,
            follow_up_queries=follow_ups,
            reasoning=f"Missing coverage of: {', '.join(missing)}.",
        )

    # --- Check 4: academic coverage for research queries ---
    if _is_academic_query(question) and current_round < max_rounds:
        academic_ok, missing_sections = _check_academic_coverage(sources)
        if not academic_ok:
            section_queries = {
                "methodology": f"{question} architecture methodology deep learning approach",
                "dataset": f"{question} dataset benchmark training data evaluation",
                "metrics": f"{question} accuracy results evaluation metrics performance",
                "results": f"{question} experimental results comparison state of the art",
            }
            follow_ups = [section_queries[s] for s in missing_sections[:2] if s in section_queries]
            return ReflectionDecision(
                sufficient=False,
                follow_up_queries=follow_ups,
                reasoning=f"Academic gaps found: missing {', '.join(missing_sections)} coverage.",
            )

    # --- Check 5: contradiction detection ---
    if _check_contradictions(sources, question) and current_round < max_rounds:
        return ReflectionDecision(
            sufficient=False,
            follow_up_queries=[
                f"{question} authoritative source consensus",
                f"{question} peer reviewed research",
            ],
            reasoning="Conflicting information detected — searching for authoritative consensus.",
        )

    # --- Final: LLM judgment (only if all heuristics pass) ---
    try:
        sources_summary = "\n".join(
            f"[{s['id']}] {s['title']} — {s['snippet'][:150]}" for s in list(sources.values())[:8]
        )
        llm = get_llm(temperature=0.0)
        response = llm.invoke(
            [
                (
                    "system",
                    "You judge whether gathered research sources are sufficient to answer "
                    "a question comprehensively. Respond ONLY with JSON.\n"
                    '{"sufficient": true, "follow_up_queries": [], "reasoning": "one sentence"}',
                ),
                ("human", f"Question: {question}\n\nSources ({len(sources)} total):\n{sources_summary}"),
            ]
        )
        text = response.content.strip()
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            data = json.loads(match.group(0))
            return ReflectionDecision(
                sufficient=data.get("sufficient", True),
                follow_up_queries=data.get("follow_up_queries", []),
                reasoning=data.get("reasoning", "LLM judgment: sources sufficient."),
            )
    except Exception as exc:
        logger.warning(f"LLM reflection failed: {exc}")

    # Fallback: if we have enough sources and diversity, proceed
    return ReflectionDecision(
        sufficient=True,
        follow_up_queries=[],
        reasoning=f"Sufficient: {len(sources)} sources from {domain_count} domains.",
    )


# ---------------------------------------------------------------------------
# #4: Post-synthesis reflection — closes the loop after report is written
# ---------------------------------------------------------------------------


def post_synthesis_check(question: str, report: str) -> dict:
    """
    After synthesis, verify the report actually answers the original
    question — not just that sources were sufficient beforehand.
    Pre-synthesis reflection only gates evidence collection; this catches
    cases where good evidence existed but synthesis itself produced a
    weak or off-topic report.

    Returns {"satisfies_query": bool, "weak_sections": [...], "reasoning": str}
    Non-blocking — failures here are logged but don't crash the pipeline.
    """
    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(
            [
                (
                    "system",
                    "You are a strict quality reviewer. Given a research question and "
                    "the report written to answer it, judge if the report actually "
                    "satisfies the question. Respond ONLY with JSON:\n"
                    '{"satisfies_query": true, "weak_sections": [], "reasoning": "one sentence"}\n'
                    "weak_sections lists any section names that are too generic, vague, "
                    "or fail to address the question.",
                ),
                ("human", f"Question: {question}\n\nReport (first 2000 chars):\n{report[:2000]}"),
            ]
        )
        text = response.content.strip()
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            data = json.loads(match.group(0))
            return {
                "satisfies_query": data.get("satisfies_query", True),
                "weak_sections": data.get("weak_sections", []),
                "reasoning": data.get("reasoning", ""),
            }
    except Exception as exc:
        logger.warning(f"Post-synthesis check failed (non-blocking): {exc}")

    return {"satisfies_query": True, "weak_sections": [], "reasoning": "Check unavailable."}

"""
FactVerificationAgent — Phase 3 Milestone 3.

Layered on top of CitationAgent (validate_node), not a replacement for
it. CitationAgent already strips any [n] marker that doesn't map to a
real retrieved source — cheap, regex-only, always runs, validates
*existence*. This agent runs one step later and asks a harder question
CitationAgent was never designed to answer: does the source a citation
*does* point to actually *support* the specific sentence it's attached
to? That's entailment, not existence, and it needs an LLM.

Pipeline position: the last node in the graph, after validate ->
fact_verify -> END. See graph.py for the topology change and main.py
for how its output is merged into the final SSE payload / session record.

Failure contract (important): this agent must NEVER raise out of run().
If claim extraction or the verification LLM call fails for any reason,
run() catches it internally and returns the report/citations completely
unchanged, with citation_verification=[] and citation_confidence=None,
plus a log line explaining the fallback. If this method were allowed to
raise, LangGraph would propagate the exception up into main.py's generic
error handler and turn an already-good, already-validated report into a
failed run — exactly the opposite of "gracefully fall back to
CitationAgent behavior."
"""

import os
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..state import AgentState
from ..llm import get_llm
from ..cache import get_verification_cache
from .base import Agent

logger = logging.getLogger(__name__)

# Bounds the size (and therefore cost/latency) of the single batched
# verification call regardless of how many citations a report has.
MAX_CLAIMS_TO_VERIFY = int(os.environ.get("MAX_CLAIMS_TO_VERIFY", "15"))

VALID_VERDICTS = {"supported", "partially_supported", "unsupported", "cannot_determine"}
# Used to compute the aggregate 0-100 citation_confidence score — how
# much a verdict itself counts, independent of the LLM's own stated
# per-claim confidence (see _aggregate_confidence).
VERDICT_WEIGHT = {
    "supported": 100,
    "partially_supported": 60,
    "unsupported": 0,
    "cannot_determine": 50,
}

CITATION_RE = re.compile(r"\[(\d+)\]")
# Same sentence-boundary heuristic rag.py's chunk_text() already uses,
# applied here to the report body instead of source snippets.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

SOURCES_FOOTER_MARKER = "\n\n---\n\n**Sources**"
NO_CITATIONS_FOOTER_MARKER = "\n\n---\n\n*No verifiable citations produced.*"


def _extract_body(report: str) -> str:
    """Strips the Sources/footer section CitationAgent appended, so claim
    extraction doesn't treat reference-list lines ("[3] Title — url") as
    claims needing verification."""
    for marker in (SOURCES_FOOTER_MARKER, NO_CITATIONS_FOOTER_MARKER):
        idx = report.find(marker)
        if idx != -1:
            return report[:idx]
    return report


def _extract_claims(report: str, valid_ids: set) -> List[Dict[str, Any]]:
    """
    Splits the report body into sentences, keeping only sentences that
    contain at least one citation id that survived CitationAgent's
    validation. Returns one claim entry PER (sentence, citation_id) pair
    — a sentence citing [1][2] yields two independently-verifiable claims.
    """
    body = _extract_body(report).strip()
    if not body:
        return []

    sentences = SENTENCE_SPLIT_RE.split(body)
    claims: List[Dict[str, Any]] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        cited = {int(m) for m in CITATION_RE.findall(sentence)} & valid_ids
        for cid in sorted(cited):
            claims.append({"sentence": sentence, "citation_id": cid})
    return claims


def _parse_json_array(text: str) -> Optional[list]:
    """Same tolerant-parsing pattern as agents/planner.py's parse_json,
    adapted for a top-level JSON array instead of an object."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def _build_verification_prompt(claims: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = (
        "You are a strict fact-checking assistant. For each numbered claim "
        "below, decide whether the paired source text actually supports it. "
        "Respond ONLY with a JSON array, exactly one object per claim, in "
        "the SAME ORDER as given:\n"
        '[{"verdict": "supported|partially_supported|unsupported|cannot_determine", '
        '"confidence": 0-100, "reasoning": "one short sentence"}]\n\n'
        "supported: the source text directly confirms the claim.\n"
        "partially_supported: the source is related and consistent but "
        "doesn't fully confirm every specific detail (e.g. a number) in the claim.\n"
        "unsupported: the source text contradicts the claim, or doesn't "
        "address it at all.\n"
        "cannot_determine: the source text is too short or unclear to judge."
    )
    lines = []
    for i, c in enumerate(claims):
        lines.append(
            f'Claim {i + 1}: "{c["sentence"]}"\n'
            f'Source text: "{c["source_text"][:800]}"'
        )
    human = "\n\n".join(lines) + f"\n\nRespond with a JSON array of exactly {len(claims)} objects."
    return system, human


class FactVerificationAgent(Agent):
    """Claim-level entailment check for citations that already passed
    CitationAgent's existence check. See module docstring for the full
    failure-handling contract."""

    name = "fact_verify"

    def trace_inputs(self, state: AgentState):
        return {"citations_used": len(state.get("citations_used", []))}

    def run(self, state: AgentState) -> dict:
        report = state.get("report", "")
        citations_used = state.get("citations_used", [])
        sources = state.get("sources", {})

        # Nothing to verify — same "always return these keys" contract as
        # the failure-fallback path below, so API/frontend consumers never
        # need to special-case "key missing" vs. "key present but empty".
        empty_result = {
            "citation_verification": [],
            "citation_confidence": None,
        }

        if not citations_used or not report:
            return {
                **empty_result,
                "log": state["log"] + ["Fact verification: no citations to verify"],
            }

        try:
            claims = _extract_claims(report, set(citations_used))
        except Exception as exc:
            logger.warning(f"Fact verification: claim extraction failed, skipping: {exc}")
            return {
                **empty_result,
                "log": state["log"] + ["Fact verification skipped (claim extraction failed)"],
            }

        if not claims:
            return {
                **empty_result,
                "log": state["log"] + ["Fact verification: no verifiable claims found"],
            }

        if len(claims) > MAX_CLAIMS_TO_VERIFY:
            claims = claims[:MAX_CLAIMS_TO_VERIFY]

        # Attach source text; claims whose source has no usable text are
        # set aside rather than sent to the LLM (nothing to check them
        # against) but still show up in the final results as
        # "cannot_determine", not silently dropped.
        enriched = []
        for c in claims:
            src = sources.get(c["citation_id"], {})
            source_text = (src.get("snippet") or "").strip()
            enriched.append({**c, "source_text": source_text or None})

        verifiable = [c for c in enriched if c["source_text"]]
        missing_source = [c for c in enriched if not c["source_text"]]

        start = time.time()
        try:
            results = self._verify_claims(verifiable) if verifiable else []
        except Exception as exc:
            # This is the core fallback contract: never let a verification
            # failure touch the report CitationAgent already produced.
            logger.warning(
                f"Fact verification failed ({exc}) — falling back to "
                "CitationAgent output unchanged"
            )
            return {
                **empty_result,
                "log": state["log"] + [
                    "Fact verification unavailable — report unchanged "
                    "(CitationAgent validation still applies)"
                ],
            }
        elapsed_ms = int((time.time() - start) * 1000)

        for c in missing_source:
            results.append({
                "sentence": c["sentence"],
                "citation_id": c["citation_id"],
                "verdict": "cannot_determine",
                "confidence": 0,
                "reasoning": "No source text available to verify against.",
            })

        overall_confidence = self._aggregate_confidence(results)
        verdict_counts = self._count_verdicts(results)

        log_msg = (
            f"Fact verification: {len(results)} claims checked "
            f"({verdict_counts.get('supported', 0)} supported, "
            f"{verdict_counts.get('partially_supported', 0)} partial, "
            f"{verdict_counts.get('unsupported', 0)} unsupported, "
            f"{verdict_counts.get('cannot_determine', 0)} undetermined) "
            f"— overall citation confidence {overall_confidence}/100, {elapsed_ms}ms"
        )
        logger.info(f"[fact_verify] {log_msg}")

        return {
            "citation_verification": results,
            "citation_confidence": overall_confidence,
            "log": state["log"] + [log_msg],
        }

    # -----------------------------------------------------------------
    # Verification + caching
    # -----------------------------------------------------------------

    def _verify_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Per-claim cache lookup, then ONE batched LLM call covering every
        cache miss (not one call per claim — this is the main cost
        control for reports with many citations). Order of the returned
        list matches the input `claims` list.
        """
        cache = get_verification_cache()
        results: List[Optional[Dict[str, Any]]] = [None] * len(claims)
        to_verify: List[Dict[str, Any]] = []
        to_verify_positions: List[int] = []
        to_verify_keys: List[str] = []

        for i, c in enumerate(claims):
            key = self._cache_key(c)
            cached, hit = cache.get(key)
            if hit:
                results[i] = cached
            else:
                to_verify.append(c)
                to_verify_positions.append(i)
                to_verify_keys.append(key)

        if to_verify:
            fresh_verdicts = self._call_llm_verifier(to_verify)
            for pos, key, claim, verdict_data in zip(
                to_verify_positions, to_verify_keys, to_verify, fresh_verdicts
            ):
                verdict = verdict_data.get("verdict", "cannot_determine")
                if verdict not in VALID_VERDICTS:
                    verdict = "cannot_determine"
                record = {
                    "sentence": claim["sentence"],
                    "citation_id": claim["citation_id"],
                    "verdict": verdict,
                    "confidence": max(0, min(100, int(verdict_data.get("confidence", 50) or 0))),
                    "reasoning": verdict_data.get("reasoning", ""),
                }
                results[pos] = record
                cache.set(key, record)

        return [r for r in results if r is not None]

    def _call_llm_verifier(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Single batched LLM call. Raises on any failure (LLM error,
        malformed/short JSON response) — callers are responsible for
        catching and falling back gracefully; this method does not.
        """
        system, human = _build_verification_prompt(claims)
        llm = get_llm(temperature=0.0)
        response = llm.invoke([("system", system), ("human", human)])
        data = _parse_json_array(response.content)
        if not isinstance(data, list) or len(data) != len(claims):
            raise ValueError(
                f"Verifier returned "
                f"{len(data) if isinstance(data, list) else type(data).__name__} "
                f"results for {len(claims)} claims"
            )
        return data

    @staticmethod
    def _cache_key(claim: Dict[str, Any]) -> str:
        return f"{claim['sentence']}|{claim['source_text']}"

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------

    @staticmethod
    def _aggregate_confidence(results: List[Dict[str, Any]]) -> int:
        """
        Blends verdict severity (a hard-coded weight per verdict category)
        with the LLM's own per-claim confidence, so one low-confidence
        "supported" verdict doesn't count identically to a high-confidence
        one. Returns 0 for an empty result list rather than raising.
        """
        if not results:
            return 0
        scores = []
        for r in results:
            weight = VERDICT_WEIGHT.get(r["verdict"], 50)
            conf = max(0, min(100, r.get("confidence", 50)))
            scores.append((weight + conf) / 2)
        return int(round(sum(scores) / len(scores)))

    @staticmethod
    def _count_verdicts(results: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        return counts

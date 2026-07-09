"""
SupervisorAgent — Phase 3 Milestone 1.

Owns the "search" LangGraph node. This is the one node in the pipeline
that genuinely needs an orchestrator rather than a single agent: which
*search agent* should run this round (WebResearchAgent vs.
AcademicSearchAgent, based on detected intent) is a routing decision,
and the concurrent full-content-fetch / splice / credibility-sort work
that follows is shared by both search strategies rather than being
specific to either one of them.

Everything below is the exact same logic that lived in graph.py's
search_node (itself already restructured once, in Milestone 2, into the
gather-candidates / concurrent-fetch / splice-back passes) — the only
change in this milestone is *which* function decides web vs. academic:
that single line now delegates to self.web_agent.search(...) or
self.academic_agent.search(...) instead of calling
web_search_batch/academic_web_search_batch directly. Everything else —
MAX_SOURCES, MAX_FULL_FETCHES, dedup, timeout handling, credibility
ordering, log messages — is unchanged.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ..state import AgentState, Source
from ..tools import fetch_full_content
from ..credibility import score_url
from .base import Agent
from .intent import _detect_research_intent
from .search import WebResearchAgent, AcademicSearchAgent

logger = logging.getLogger(__name__)

# Milestone 2: concurrent full-content fetch — unchanged from graph.py.
FULL_FETCH_TIMEOUT = int(os.environ.get("FULL_FETCH_TIMEOUT", "8"))


def _fetch_full_content_batch(urls: List[str], timeout: int = FULL_FETCH_TIMEOUT) -> Dict[str, Optional[str]]:
    """
    Fetch full page content for multiple URLs concurrently. Unchanged
    from graph.py's Milestone 2 implementation — see that milestone's
    docstring/diff for the full rationale (dedup, timeout handling,
    per-future failure isolation, timing + success/failure counters).
    """
    unique_urls = list(dict.fromkeys(u for u in urls if u))
    results: Dict[str, Optional[str]] = {}
    if not unique_urls:
        return results

    start = time.time()
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=min(6, len(unique_urls))) as executor:
        future_map = {
            executor.submit(fetch_full_content, url, timeout): url
            for url in unique_urls
        }
        for future in future_map:
            url = future_map[future]
            try:
                content = future.result()
                results[url] = content
                if content:
                    success += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.error(f"[fetch] concurrent full-content fetch failed: {url!r} — {exc}")
                results[url] = None
                failed += 1

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(
        f"[fetch] full-content batch: {len(unique_urls)} urls, "
        f"{success} enriched, {failed} empty/failed, {elapsed_ms}ms"
    )
    return results


class SupervisorAgent(Agent):
    """Orchestrates the search step: routes to the right search agent
    based on intent, then runs the shared concurrent-fetch/splice/sort
    pipeline over whichever agent's results came back."""

    name = "search"

    def __init__(self):
        self.web_agent = WebResearchAgent()
        self.academic_agent = AcademicSearchAgent()

    def trace_inputs(self, state: AgentState):
        return {
            "round": state.get("round", 0),
            "intent": _detect_research_intent(state.get("question", "")),
        }

    def run(self, state: AgentState) -> dict:
        reflection = state.get("reflection")
        queries = state["plan"] if state["round"] == 0 else (
            reflection.follow_up_queries if reflection else []
        )

        intent = _detect_research_intent(state["question"])
        sources = dict(state["sources"])
        next_id = max(sources.keys(), default=0) + 1
        seen_urls = {s["url"] for s in sources.values()}

        MAX_SOURCES = 20
        MAX_FULL_FETCHES = 6

        # Delegate the actual search call to the appropriate agent.
        if intent == "academic":
            batch_results = self.academic_agent.search(queries)
        else:
            batch_results = self.web_agent.search(queries)

        # --- Pass 1: assemble candidate sources, pick full-content fetch targets ---
        fetch_targets: Dict[str, int] = {}
        pending: "Dict[int, dict]" = {}

        for query in queries:
            if len(sources) + len(pending) >= MAX_SOURCES:
                break
            results = batch_results.get(query, [])

            for result in results:
                if len(sources) + len(pending) >= MAX_SOURCES:
                    break
                url = result.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                snippet = (result.get("content", "") or result.get("snippet", "") or "")[:1500]

                if len(fetch_targets) < MAX_FULL_FETCHES and score_url(url) >= 90:
                    fetch_targets[url] = next_id

                pending[next_id] = {
                    "url": url,
                    "title": result.get("title", url),
                    "snippet": snippet,
                }
                next_id += 1

        # --- Pass 2: fetch full content for every qualifying URL concurrently ---
        fetch_start = time.time()
        enriched_by_url = _fetch_full_content_batch(list(fetch_targets.keys())) if fetch_targets else {}
        fetch_elapsed_ms = int((time.time() - fetch_start) * 1000)

        # --- Pass 3: splice enrichment back into snippets, preserving order ---
        full_fetch_count = 0
        enriched_count = 0
        for src_id, fields in pending.items():
            url = fields["url"]
            snippet = fields["snippet"]
            if url in fetch_targets:
                full_fetch_count += 1
                enriched = enriched_by_url.get(url)
                if enriched and len(enriched) > len(snippet):
                    snippet = snippet + "\n\n[Full content]:\n" + enriched[:1500]
                    enriched_count += 1
            sources[src_id] = Source(
                id=src_id,
                url=url,
                title=fields["title"],
                snippet=snippet[:3000],
            )

        sorted_sources = dict(
            sorted(
                sources.items(),
                key=lambda x: score_url(x[1].get("url", "")),
                reverse=True,
            )
        )

        log_entries = []
        if fetch_targets:
            log_entries.append(
                f"Full-content fetch: {len(fetch_targets)} urls, "
                f"{enriched_count} enriched, {fetch_elapsed_ms}ms"
            )
        log_entries.append(f"Round {state['round']+1}: gathered {len(sorted_sources)} sources")

        logger.info(
            f"Search round {state['round']+1}: {len(sorted_sources)} sources "
            f"(intent: {intent}, full-fetch attempted: {full_fetch_count}, "
            f"enriched: {enriched_count}, fetch phase: {fetch_elapsed_ms}ms)"
        )
        return {
            "sources": sorted_sources,
            "round": state["round"] + 1,
            "log": state["log"] + log_entries,
        }

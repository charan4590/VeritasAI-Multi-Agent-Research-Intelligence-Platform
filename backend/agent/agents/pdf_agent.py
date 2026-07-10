"""
PDFAgent — Phase 3 Milestone 2.

Wraps pdf_ingestion.py's existing search_pdfs() (reused exactly as-is —
no PDF retrieval logic is rebuilt here) so uploaded PDFs can be searched
alongside web/academic results, using the same per-query batch interface
WebResearchAgent/AcademicSearchAgent already expose. This is what lets
SupervisorAgent merge all three result sets with one consistent pattern.

Graceful no-PDFs behavior comes for free from search_pdfs() itself: when
no PDF has been uploaded to a session, the "pdf_docs" ChromaDB collection
doesn't exist yet, and search_pdfs() already catches that (returns []
before ever calling embed_text() or doing a vector query) — so calling
this agent costs one cheap collection-lookup attempt per query and
nothing else. See supervisor.py for how the merge is done and the
performance-impact writeup in the Milestone 2 PR notes for the measured
cost of that lookup.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from ..pdf_ingestion import search_pdfs
from ..state import AgentState
from .base import Agent

logger = logging.getLogger(__name__)

# Configurable limit for how many PDF chunks are retrieved *per query*.
# Mirrors the max_results knobs WebResearchAgent (5) / AcademicSearchAgent
# (4) already hard-code, but this one is env-configurable since PDF corpus
# size and desired precision/recall tradeoff varies a lot more by
# deployment than web search result counts do.
PDF_CHUNKS_PER_QUERY = int(os.environ.get("PDF_CHUNKS_PER_QUERY", "3"))


class PDFAgent(Agent):
    """Semantic search over uploaded PDFs. Read-only wrapper around
    pdf_ingestion.search_pdfs() — all chunking/embedding/indexing logic
    lives there, unchanged, from the existing (previously unwired) PDF
    ingestion pipeline."""

    name = "pdf_search"

    def search(
        self, queries: List[str], top_k: int = PDF_CHUNKS_PER_QUERY
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Concurrent per-query PDF search, same ThreadPoolExecutor pattern
        as web_search_batch/academic_web_search_batch in tools.py, for
        consistency and so N queries cost roughly one query's latency
        instead of N times that.
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        if not queries:
            return results
        with ThreadPoolExecutor(max_workers=min(5, len(queries))) as executor:
            future_map = {executor.submit(search_pdfs, q, top_k): q for q in queries}
            for future in future_map:
                q = future_map[future]
                try:
                    results[q] = future.result()
                except Exception as exc:
                    logger.error(f"[pdf] search failed for {q!r}: {exc}")
                    results[q] = []
        return results

    def run(self, state: AgentState) -> dict:
        return {"results": self.search(state.get("plan", []))}

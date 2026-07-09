"""
RAGAgent — Phase 3 Milestone 1.

Moved from graph.py's rag_node. Two-stage retrieval logic (bi-encoder
recall via rag.py's ChromaDB index, then cross-encoder re-rank via
reranker.py) is byte-for-byte unchanged from before this milestone.
"""

import os
import uuid
import logging

from ..state import AgentState
from ..rag import index_sources, retrieve_relevant_chunks
from ..reranker import rerank_chunks, is_reranker_available
from .base import Agent

logger = logging.getLogger(__name__)

RETRIEVAL_CANDIDATES = int(os.environ.get("RAG_CANDIDATES", "25"))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "8"))


class RAGAgent(Agent):
    """Indexes this round's sources into a per-session vector store,
    retrieves the most relevant chunks for the question, and re-ranks
    them with a cross-encoder when available."""

    name = "rag"

    def trace_inputs(self, state: AgentState):
        return {"source_count": len(state.get("sources", {}))}

    def run(self, state: AgentState) -> dict:
        session_id = state.get("rag_session_id") or str(uuid.uuid4())[:8]
        log_entries = [f"RAG: indexing {len(state['sources'])} sources..."]

        success = index_sources(state["sources"], session_id)

        if success:
            candidates = retrieve_relevant_chunks(
                state["question"], session_id, top_k=RETRIEVAL_CANDIDATES,
            )
            log_entries.append(f"RAG: retrieved {len(candidates)} candidate chunks")

            if candidates and is_reranker_available():
                chunks = rerank_chunks(state["question"], candidates, top_k=RERANK_TOP_K)
                log_entries.append(f"Re-ranked → top {len(chunks)} chunks")
            elif candidates:
                chunks = candidates[:RERANK_TOP_K]
                log_entries.append(f"Top {len(chunks)} chunks (no cross-encoder)")
            else:
                chunks = []
                log_entries.append("RAG: no chunks — using raw sources")
        else:
            chunks = []
            log_entries.append("RAG: unavailable — using raw sources")

        return {
            "rag_session_id": session_id,
            "retrieved_chunks": chunks,
            "log": state["log"] + log_entries,
        }

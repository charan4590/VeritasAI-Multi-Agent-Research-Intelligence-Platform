"""
Phase 4: Semantic Memory
=========================
Architecture decision: ChromaDB collection "agent_memory" stores
embeddings of past research questions. When a new question arrives,
we retrieve the top-2 semantically similar past sessions and inject
a brief summary as context into the planner prompt.

Why this matters: if you researched "AI agent frameworks in 2025"
last week, and now ask "which LangGraph features are new in 2025",
the agent should know it already has context about AI agent frameworks.
Without semantic memory, every run starts from zero.

Why only top-2: context window pollution is a real risk. Injecting
too many past sessions confuses the model. 2 is enough to signal
prior knowledge without overwhelming the planner.

Degradation: if ChromaDB or Ollama is unavailable, returns empty
memories and the agent runs normally with no memory context.
"""

import os
from typing import Dict, List, Optional

from .rag import _get_chroma, embed_text

MEMORY_COLLECTION = "agent_memory"
MEMORY_TOP_K = int(os.environ.get("MEMORY_TOP_K", "2"))
MEMORY_THRESHOLD = float(os.environ.get("MEMORY_THRESHOLD", "0.75"))


def _get_memory_collection():
    try:
        client = _get_chroma()
        return client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        print(f"[memory] could not open collection: {exc}")
        return None


def store_memory(session_id: int, question: str, report_summary: str, eval_scores: Optional[Dict] = None):
    """
    Store a completed research session as a memory vector.
    Only stores sessions with overall_score >= 50 to avoid
    polluting memory with bad research.
    """
    try:
        if eval_scores and eval_scores.get("overall_score", 100) < 50:
            print(f"[memory] skipping low-quality session {session_id}")
            return

        embedding = embed_text(question)
        if embedding is None:
            return

        collection = _get_memory_collection()
        if collection is None:
            return

        # Store a concise summary (first 500 chars of report)
        summary = report_summary[:500].strip()

        collection.upsert(
            ids=[str(session_id)],
            embeddings=[embedding],
            documents=[question],
            metadatas=[
                {
                    "session_id": str(session_id),
                    "question": question,
                    "summary": summary,
                    "overall_score": str(eval_scores.get("overall_score", 0) if eval_scores else 0),
                }
            ],
        )
        print(f"[memory] stored session {session_id}")
    except Exception as exc:
        print(f"[memory] store failed: {exc}")


def retrieve_memories(question: str) -> List[Dict]:
    """
    Find the most relevant past research sessions for a new question.
    Returns a list of memory dicts, empty if none found or unavailable.
    """
    try:
        collection = _get_memory_collection()
        if collection is None or collection.count() == 0:
            return []

        embedding = embed_text(question)
        if embedding is None:
            return []

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(MEMORY_TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            similarity = round(1 - dist, 3)
            # Only inject memories above threshold to avoid noise
            if similarity >= MEMORY_THRESHOLD:
                memories.append(
                    {
                        "session_id": meta.get("session_id"),
                        "question": meta.get("question", doc),
                        "summary": meta.get("summary", ""),
                        "overall_score": int(meta.get("overall_score", 0)),
                        "similarity": similarity,
                    }
                )

        return memories
    except Exception as exc:
        print(f"[memory] retrieval failed: {exc}")
        return []


def format_memory_context(memories: List[Dict]) -> str:
    """
    Format retrieved memories into a context string for the planner.
    """
    if not memories:
        return ""

    lines = ["\n## Relevant Past Research (use as context, don't repeat verbatim)\n"]
    for m in memories:
        lines.append(
            f"- Past question [{m['similarity']:.0%} similar]: \"{m['question']}\"\n"
            f"  Summary: {m['summary'][:200]}..."
        )
    return "\n".join(lines)


def delete_memory(session_id: int):
    """Remove a session's memory when the session is deleted."""
    try:
        collection = _get_memory_collection()
        if collection:
            collection.delete(ids=[str(session_id)])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Conversation context (within a session)
# ---------------------------------------------------------------------------

_conversation_context: list = []
MAX_CONTEXT_TURNS = int(os.environ.get("MAX_CONTEXT_TURNS", "5"))


def add_conversation_turn(question: str, summary: str):
    """Track questions asked in this session for context injection."""
    global _conversation_context
    _conversation_context.append(
        {
            "question": question,
            "summary": summary[:300],
        }
    )
    # Keep only recent turns
    _conversation_context = _conversation_context[-MAX_CONTEXT_TURNS:]


def get_conversation_context() -> str:
    """
    Format recent conversation turns as context for the planner.
    Example output:
      "User previously asked: 'What is LangGraph?'
       User previously asked: 'How does reflection work in agents?'"
    This lets the agent build on previous questions in the same session.
    """
    if not _conversation_context:
        return ""
    lines = ["## Current Session Context (build on these if relevant)"]
    for turn in _conversation_context:
        lines.append("- Previously asked: " + repr(turn["question"]))
        if turn["summary"]:
            lines.append(f"  Summary: {turn['summary'][:150]}...")
    return "\n".join(lines)


def clear_conversation():
    """Clear conversation context (call on new session start)."""
    global _conversation_context
    _conversation_context = []

"""
Phase 1: RAG Pipeline
======================
Architecture decision: Use Ollama's embedding endpoint (nomic-embed-text)
so we need zero extra Python ML packages. ChromaDB stores vectors locally
as a persistent SQLite+FAISS hybrid — no server needed.

Flow:
  search results → chunk (300 tokens) → embed → ChromaDB
  question → embed → semantic retrieve top-k chunks → synthesize

Why chunking matters: Tavily returns 800-char snippets. A source about
"Python async" might mention Django halfway through. Chunking isolates
the relevant part so synthesis gets precisely what it needs.

Why local embeddings: nomic-embed-text is a 250MB Ollama model, already
on the user's machine, zero API cost, zero latency variance.
"""

import os
import re
import json
import hashlib
import requests
from typing import List, Dict, Optional, Tuple

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "300"))   # tokens ≈ chars/4
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
TOP_K = int(os.environ.get("RAG_TOP_K", "8"))

# Lazy ChromaDB import so server starts even if chromadb isn't installed
_chroma_client = None
_collections: Dict[str, object] = {}


def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        db_path = os.environ.get("CHROMA_PATH", "./chroma_db")
        _chroma_client = chromadb.PersistentClient(path=db_path)
    return _chroma_client


def _get_collection(session_id: str):
    """One ChromaDB collection per research session."""
    key = f"session_{session_id}"
    if key not in _collections:
        client = _get_chroma()
        _collections[key] = client.get_or_create_collection(
            name=key,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[key]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> Optional[List[float]]:
    """
    Call Ollama's /api/embeddings endpoint.
    Returns None if Ollama is unavailable so RAG degrades gracefully.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:2000]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception as exc:
        print(f"[rag] embedding failed: {exc}")
        return None


def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Embed multiple texts in parallel using a thread pool.
    Sequential embedding of 79 chunks was the main RAG bottleneck —
    each Ollama embedding call takes 1-3s, so 79 sequential calls
    could take 2-4 minutes. Parallelizing with 8 workers cuts this
    to roughly 1/8th the time since Ollama can handle concurrent requests.
    """
    if not texts:
        return []
    from concurrent.futures import ThreadPoolExecutor
    max_workers = min(8, len(texts))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(embed_text, texts))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    #5: Semantic-aware chunking.
    Previously split on raw word counts, which fragments sentences and
    tables mid-way — especially damaging for academic content where a
    methodology description or results table getting cut mid-sentence
    loses meaning. This version splits on paragraph/sentence boundaries
    first, then packs sentences into chunks up to chunk_size words,
    falling back to word-count splitting only for pathological single
    sentences longer than chunk_size.
    """
    import re as _re

    if not text.strip():
        return [text]

    # Split into sentences using boundary-aware regex (handles abbreviations
    # reasonably well without needing a full NLP library)
    sentences = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    chunks = []
    current_chunk_sentences: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # Pathological case: single sentence longer than chunk_size —
        # fall back to word splitting for just this sentence
        if sentence_words > chunk_size:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_word_count = 0
            words = sentence.split()
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))
            continue

        if current_word_count + sentence_words > chunk_size and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            # Overlap: carry forward the last sentence for context continuity
            overlap_sentences = []
            overlap_words = 0
            for s in reversed(current_chunk_sentences):
                w = len(s.split())
                if overlap_words + w > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_words += w
            current_chunk_sentences = overlap_sentences
            current_word_count = overlap_words

        current_chunk_sentences.append(sentence)
        current_word_count += sentence_words

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks if chunks else [text]


def chunk_source(source: Dict) -> List[Dict]:
    """
    Chunk a single source dict into multiple chunk dicts.
    Each chunk carries metadata for attribution in the final report.
    """
    text = source.get("snippet", "")
    if not text:
        return []
    chunks = chunk_text(text)
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "chunk_id": f"{source['id']}_chunk_{i}",
            "source_id": source["id"],
            "url": source.get("url", ""),
            "title": source.get("title", ""),
            "text": chunk,
            "chunk_index": i,
        })
    return result


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_sources(sources: Dict, session_id: str) -> bool:
    """
    Chunk all sources and index them into ChromaDB.
    Returns True if successful, False if Ollama/ChromaDB unavailable.
    """
    try:
        collection = _get_collection(session_id)
        all_chunks = []
        for src in sources.values():
            all_chunks.extend(chunk_source(src))

        if not all_chunks:
            return False

        # Cap total chunks embedded — with 20 sources at ~5 chunks each
        # this could reach 100 chunks. Embedding is now parallelized but
        # still capping keeps worst-case latency predictable.
        MAX_CHUNKS = 60
        if len(all_chunks) > MAX_CHUNKS:
            all_chunks = all_chunks[:MAX_CHUNKS]

        texts = [c["text"] for c in all_chunks]
        embeddings = embed_texts(texts)

        # Filter out chunks where embedding failed
        valid = [
            (c, e) for c, e in zip(all_chunks, embeddings) if e is not None
        ]
        if not valid:
            return False

        collection.upsert(
            ids=[c["chunk_id"] for c, _ in valid],
            embeddings=[e for _, e in valid],
            documents=[c["text"] for c, _ in valid],
            metadatas=[{
                "source_id": str(c["source_id"]),
                "url": c["url"],
                "title": c["title"],
                "chunk_index": c["chunk_index"],
            } for c, _ in valid],
        )
        print(f"[rag] indexed {len(valid)} chunks for session {session_id}")
        return True

    except Exception as exc:
        print(f"[rag] indexing failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(question: str, session_id: str,
                              top_k: int = TOP_K) -> List[Dict]:
    """
    Semantic retrieval: embed the question and find the most relevant chunks.
    Returns a list of chunk dicts with text + source metadata.
    Falls back to empty list if unavailable.
    """
    try:
        collection = _get_collection(session_id)
        if collection.count() == 0:
            return []

        q_embedding = embed_text(question)
        if q_embedding is None:
            return []

        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            relevance = round(1 - dist, 3)  # cosine: 1=identical, 0=orthogonal
            chunks.append({
                "text": doc,
                "source_id": meta.get("source_id"),
                "url": meta.get("url"),
                "title": meta.get("title"),
                "relevance": relevance,
            })

        # Sort by relevance descending
        chunks.sort(key=lambda x: x["relevance"], reverse=True)
        return chunks

    except Exception as exc:
        print(f"[rag] retrieval failed: {exc}")
        return []


def format_retrieved_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a context block for the synthesis prompt.
    Groups chunks by source and includes relevance score.
    """
    if not chunks:
        return ""

    lines = ["## Semantically Retrieved Context (ranked by relevance)\n"]
    seen_sources = {}

    for chunk in chunks:
        sid = chunk.get("source_id", "?")
        if sid not in seen_sources:
            seen_sources[sid] = chunk.get("title", f"Source {sid}")
            lines.append(f"\n### [{sid}] {seen_sources[sid]}")
            lines.append(f"URL: {chunk.get('url', '')}")

        lines.append(
            f"\n[relevance: {chunk['relevance']:.2f}] {chunk['text']}"
        )

    return "\n".join(lines)


def cleanup_session_collection(session_id: str):
    """Remove ChromaDB collection when session is deleted."""
    try:
        client = _get_chroma()
        client.delete_collection(f"session_{session_id}")
        _collections.pop(f"session_{session_id}", None)
    except Exception:
        pass

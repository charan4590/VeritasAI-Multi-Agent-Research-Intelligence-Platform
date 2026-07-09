"""
Priority 5: PDF Ingestion
==========================
Architecture: PDFs are chunked, embedded, and stored in a dedicated
ChromaDB collection called "pdf_docs" — separate from per-session
research collections. This means uploaded PDFs persist across all
future research sessions and the agent can draw on them automatically.

Flow:
  upload PDF → extract text (pypdf) → chunk → embed (Ollama)
             → store in ChromaDB "pdf_docs" collection

Integration with research: the search_node in graph.py already
calls web_search. We add pdf_search() alongside it so the agent
also queries the PDF knowledge base for every research run.

Why pypdf: lightweight, no Java dependency (unlike PDFMiner), works
on all platforms. For scanned PDFs you'd add OCR (pytesseract) but
that's out of scope here.
"""

import os
import uuid
import sqlite3
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "research.db")
PDF_COLLECTION = "pdf_docs"


# ---------------------------------------------------------------------------
# DB: doc metadata
# ---------------------------------------------------------------------------

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_docs_table():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingested_docs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'processing',
                error TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


_init_docs_table()


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(content: bytes) -> Optional[List[Dict]]:
    """
    Extract text per page using pypdf.
    Returns list of {page: int, text: str} dicts.
    Returns None if pypdf is not installed.
    """
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
        return pages
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return None
    except Exception as exc:
        logger.error(f"PDF extraction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_pdf(content: bytes, filename: str) -> Dict:
    """
    Full ingestion pipeline for a PDF file.
    Returns {"success": True, "doc_id": ..., "chunks": N}
    """
    from .rag import chunk_text, embed_text, _get_chroma

    doc_id = str(uuid.uuid4())[:12]

    # Save metadata
    with _conn() as conn:
        conn.execute(
            "INSERT INTO ingested_docs (id, filename, status) VALUES (?, ?, 'processing')",
            (doc_id, filename),
        )
        conn.commit()

    try:
        # Step 1: Extract text
        pages = _extract_text_from_pdf(content)
        if pages is None:
            _update_doc_status(doc_id, "error", "pypdf not available")
            return {"success": False, "error": "pypdf not installed"}

        if not pages:
            _update_doc_status(doc_id, "error", "No text found in PDF")
            return {"success": False, "error": "Could not extract text from PDF"}

        # Step 2: Chunk
        all_chunks = []
        for page in pages:
            text_chunks = chunk_text(page["text"], chunk_size=250, overlap=40)
            for i, chunk_text_item in enumerate(text_chunks):
                if chunk_text_item.strip():
                    all_chunks.append({
                        "chunk_id": f"{doc_id}_p{page['page']}_c{i}",
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page["page"],
                        "text": chunk_text_item,
                    })

        if not all_chunks:
            _update_doc_status(doc_id, "error", "No chunks produced")
            return {"success": False, "error": "PDF produced no content chunks"}

        # Step 3: Embed + store in ChromaDB
        client = _get_chroma()
        collection = client.get_or_create_collection(
            PDF_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        embedded_count = 0
        # Process in batches of 10 to avoid memory issues
        batch_size = 10
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = [embed_text(c["text"]) for c in batch]
            valid = [(c, e) for c, e in zip(batch, embeddings) if e is not None]

            if valid:
                collection.upsert(
                    ids=[c["chunk_id"] for c, _ in valid],
                    embeddings=[e for _, e in valid],
                    documents=[c["text"] for c, _ in valid],
                    metadatas=[{
                        "doc_id": c["doc_id"],
                        "filename": c["filename"],
                        "page": str(c["page"]),
                    } for c, _ in valid],
                )
                embedded_count += len(valid)

        # Update metadata
        with _conn() as conn:
            conn.execute("""
                UPDATE ingested_docs
                SET status='ready', page_count=?, chunk_count=?
                WHERE id=?
            """, (len(pages), embedded_count, doc_id))
            conn.commit()

        logger.info(f"Ingested {filename}: {len(pages)} pages, {embedded_count} chunks")
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": embedded_count,
        }

    except Exception as exc:
        logger.exception(f"Ingestion failed for {filename}")
        _update_doc_status(doc_id, "error", str(exc))
        return {"success": False, "error": str(exc)}


def _update_doc_status(doc_id: str, status: str, error: Optional[str] = None):
    with _conn() as conn:
        conn.execute(
            "UPDATE ingested_docs SET status=?, error=? WHERE id=?",
            (status, error, doc_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Search ingested PDFs
# ---------------------------------------------------------------------------

def search_pdfs(query: str, top_k: int = 5) -> List[Dict]:
    """
    Semantic search across all ingested PDFs.
    Returns chunks relevant to the query, formatted like web search results
    so they can be merged with Tavily results seamlessly.
    """
    try:
        from .rag import embed_text, _get_chroma

        client = _get_chroma()
        try:
            collection = client.get_collection(PDF_COLLECTION)
        except Exception:
            return []  # No PDFs ingested yet

        if collection.count() == 0:
            return []

        embedding = embed_text(query)
        if embedding is None:
            return []

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            relevance = round(1 - dist, 3)
            if relevance > 0.5:  # Only return relevant chunks
                chunks.append({
                    "url": f"pdf://{meta.get('filename', 'unknown')}#page{meta.get('page', '?')}",
                    "title": f"{meta.get('filename', 'PDF')} (p.{meta.get('page', '?')})",
                    "content": doc,
                    "relevance": relevance,
                    "source_type": "pdf",
                })

        return chunks

    except Exception as exc:
        logger.error(f"PDF search failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Management
# ---------------------------------------------------------------------------

def list_ingested_docs() -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ingested_docs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_doc(doc_id: str):
    """Remove doc from DB and ChromaDB."""
    try:
        from .rag import _get_chroma
        client = _get_chroma()
        try:
            collection = client.get_collection(PDF_COLLECTION)
            # Delete all chunks belonging to this doc
            results = collection.get(where={"doc_id": doc_id})
            if results and results.get("ids"):
                collection.delete(ids=results["ids"])
        except Exception:
            pass
    except Exception as exc:
        logger.error(f"Failed to delete from ChromaDB: {exc}")

    with _conn() as conn:
        conn.execute("DELETE FROM ingested_docs WHERE id=?", (doc_id,))
        conn.commit()

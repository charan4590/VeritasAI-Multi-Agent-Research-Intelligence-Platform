"""
Database layer — updated for Phase 3 (evaluation scores, fact verification)
and Phase 2 (observability).
"""
import sqlite3
import json
import os
from typing import List, Dict, Optional

DB_PATH = os.environ.get("DB_PATH", "research.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn, table: str, column: str, coltype: str):
    """
    Adds `column` to `table` if it doesn't already exist. Needed because
    `CREATE TABLE IF NOT EXISTS` is a no-op against a database that
    already has the `history` table from before this migration — without
    this, save_session() would fail with "no such column" on any
    pre-Milestone-3 database. Portable across SQLite versions (doesn't
    rely on `ADD COLUMN IF NOT EXISTS`, which is a fairly recent addition).
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    with get_conn() as conn:
        # Main history table — extended with eval scores
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                report TEXT NOT NULL,
                sources TEXT DEFAULT '{}',
                confidence INTEGER DEFAULT 0,
                mode TEXT DEFAULT 'research',
                follow_ups TEXT DEFAULT '[]',
                -- Phase 3: Evaluation scores
                eval_overall INTEGER DEFAULT NULL,
                eval_relevance INTEGER DEFAULT NULL,
                eval_citations INTEGER DEFAULT NULL,
                eval_diversity INTEGER DEFAULT NULL,
                eval_hallucination INTEGER DEFAULT NULL,
                eval_grade TEXT DEFAULT NULL,
                eval_details TEXT DEFAULT '{}',
                -- Phase 1: RAG metadata
                rag_chunks_used INTEGER DEFAULT 0,
                -- Phase 2: Observability
                latency_ms INTEGER DEFAULT 0,
                -- Phase 3 Milestone 3: Fact Verification
                citation_verification TEXT DEFAULT '[]',
                citation_confidence INTEGER DEFAULT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Safe no-op on a fresh DB (columns already exist from the CREATE
        # above); adds the columns in place on a pre-Milestone-3 database.
        _ensure_column(conn, "history", "citation_verification", "TEXT DEFAULT '[]'")
        _ensure_column(conn, "history", "citation_confidence", "INTEGER DEFAULT NULL")
        conn.commit()


def save_session(
    question: str,
    report: str,
    sources: dict,
    confidence: int,
    mode: str,
    follow_ups: List[str],
    eval_scores: Optional[Dict] = None,
    rag_chunks_used: int = 0,
    latency_ms: int = 0,
    citation_verification: Optional[List[Dict]] = None,
    citation_confidence: Optional[int] = None,
) -> int:
    eval_scores = eval_scores or {}
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO history (
                question, report, sources, confidence, mode, follow_ups,
                eval_overall, eval_relevance, eval_citations,
                eval_diversity, eval_hallucination, eval_grade, eval_details,
                rag_chunks_used, latency_ms,
                citation_verification, citation_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            question, report, json.dumps(sources), confidence, mode,
            json.dumps(follow_ups),
            eval_scores.get("overall_score"),
            eval_scores.get("relevance_score"),
            eval_scores.get("citation_score"),
            eval_scores.get("diversity_score"),
            eval_scores.get("hallucination_risk_score"),
            eval_scores.get("grade"),
            json.dumps(eval_scores),
            rag_chunks_used,
            latency_ms,
            json.dumps(citation_verification or []),
            citation_confidence,
        ))
        conn.commit()
        return cur.lastrowid


def get_history(limit: int = 50) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.get("sources") or "{}")
        d["follow_ups"] = json.loads(d.get("follow_ups") or "[]")
        d["eval_details"] = json.loads(d.get("eval_details") or "{}")
        d["citation_verification"] = json.loads(d.get("citation_verification") or "[]")
        result.append(d)
    return result


def get_session(session_id: int) -> Optional[Dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM history WHERE id = ?", (session_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["sources"] = json.loads(d.get("sources") or "{}")
    d["follow_ups"] = json.loads(d.get("follow_ups") or "[]")
    d["eval_details"] = json.loads(d.get("eval_details") or "{}")
    d["citation_verification"] = json.loads(d.get("citation_verification") or "[]")
    return d


def delete_session(session_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM history WHERE id = ?", (session_id,))
        conn.commit()

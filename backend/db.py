"""
Database layer — updated for Phase 3 (evaluation scores) and Phase 2 (observability).
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
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
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
) -> int:
    eval_scores = eval_scores or {}
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO history (
                question, report, sources, confidence, mode, follow_ups,
                eval_overall, eval_relevance, eval_citations,
                eval_diversity, eval_hallucination, eval_grade, eval_details,
                rag_chunks_used, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    return d


def delete_session(session_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM history WHERE id = ?", (session_id,))
        conn.commit()

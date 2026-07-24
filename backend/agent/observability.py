"""
Phase 2: Observability
=======================
Architecture decision: custom observability stored in SQLite.
This means zero external dependencies and works offline.

Optional LangSmith integration: if LANGSMITH_API_KEY is set,
traces are also sent there for a proper GUI dashboard.

Tracks per run:
  - Total latency
  - Token estimates (input + output)
  - Cost estimates (based on model tier)
  - Per-node timing

Token estimation: LangChain/Ollama doesn't always return token counts.
We estimate: 1 token ≈ 4 characters (industry standard rough estimate).
When actual counts are available, we use those.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

DB_PATH = os.environ.get("DB_PATH", "research.db")

# Cost per 1M tokens (USD) — approximate 2025 pricing
MODEL_COSTS = {
    "ollama": {"input": 0.0, "output": 0.0},  # local = free
    "groq_llama3": {"input": 0.05, "output": 0.08},  # Groq llama3.1-8b
    "groq_llama3_70b": {"input": 0.59, "output": 0.79},
    "gemini": {"input": 0.0, "output": 0.0},  # free tier (see README) — 1500 req/day
    "default": {"input": 0.10, "output": 0.20},
}


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_observability_tables():
    """Create observability tables. Called at app startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                question TEXT,
                mode TEXT DEFAULT 'research',
                total_latency_ms INTEGER,
                total_input_tokens INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0.0,
                model_used TEXT,
                rag_enabled INTEGER DEFAULT 0,
                chunks_retrieved INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running',
                error_message TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS node_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES runs(id),
                node_name TEXT,
                start_time_ms INTEGER,
                end_time_ms INTEGER,
                latency_ms INTEGER,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                success INTEGER DEFAULT 1,
                error TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.commit()


class RunTracker:
    """
    Context manager that tracks a full agent run.

    Usage:
        tracker = RunTracker(question="...", mode="research")
        tracker.start()
        with tracker.node("planner"):
            result = planner_node(state)
        tracker.finish(session_id=42)
    """

    def __init__(self, question: str, mode: str = "research"):
        self.question = question
        self.mode = mode
        self.run_id: Optional[int] = None
        self.start_ms: int = 0
        self.node_records: List[Dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.rag_enabled = False
        self.chunks_retrieved = 0
        self.model_used = _detect_model()

    def start(self):
        self.start_ms = _now_ms()
        with _get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO runs (question, mode, model_used, status)
                   VALUES (?, ?, ?, 'running')""",
                (self.question, self.mode, self.model_used),
            )
            conn.commit()
            self.run_id = cur.lastrowid
        return self

    @contextmanager
    def node(self, name: str, metadata: Optional[Dict] = None):
        """Context manager for timing a single graph node."""
        node_start = _now_ms()
        error = None
        try:
            yield self
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            node_end = _now_ms()
            record = {
                "run_id": self.run_id,
                "node_name": name,
                "start_time_ms": node_start,
                "end_time_ms": node_end,
                "latency_ms": node_end - node_start,
                "input_tokens": 0,
                "output_tokens": 0,
                "success": 1 if error is None else 0,
                "error": error,
                "metadata": json.dumps(metadata or {}),
            }
            self.node_records.append(record)
            _save_node(record)

    def add_tokens(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def estimate_tokens_from_text(self, text: str, role: str = "output"):
        """Estimate token count from character length (4 chars ≈ 1 token)."""
        estimated = len(text) // 4
        if role == "input":
            self.total_input_tokens += estimated
        else:
            self.total_output_tokens += estimated

    def finish(self, session_id: Optional[int] = None, status: str = "done", error: Optional[str] = None):
        if not self.run_id:
            return
        total_ms = _now_ms() - self.start_ms
        cost = _estimate_cost(
            self.model_used,
            self.total_input_tokens,
            self.total_output_tokens,
        )
        with _get_conn() as conn:
            conn.execute(
                """
                UPDATE runs SET
                    session_id = ?,
                    total_latency_ms = ?,
                    total_input_tokens = ?,
                    total_output_tokens = ?,
                    estimated_cost_usd = ?,
                    rag_enabled = ?,
                    chunks_retrieved = ?,
                    status = ?,
                    error_message = ?
                WHERE id = ?
            """,
                (
                    session_id,
                    total_ms,
                    self.total_input_tokens,
                    self.total_output_tokens,
                    cost,
                    int(self.rag_enabled),
                    self.chunks_retrieved,
                    status,
                    error,
                    self.run_id,
                ),
            )
            conn.commit()
        _send_to_langsmith(self)
        return {
            "run_id": self.run_id,
            "total_latency_ms": total_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": cost,
        }


# ---------------------------------------------------------------------------
# Query functions for the API
# ---------------------------------------------------------------------------


def get_run_metrics(limit: int = 20) -> List[Dict]:
    """
    Bug fix: metrics.html's Recent Runs table and Quality Score
    Distribution chart read r.eval_grade, r.eval_overall, r.latency_ms,
    and r.rag_chunks_used off each row — none of which exist on the
    `runs` table. Those live on db.py's `history` table instead (under
    the exact same names, not by accident: history is the Phase 3/5
    source of truth for eval scores; `runs`/`node_executions` are the
    separate Phase 2 observability tables added later). Both tables live
    in the same SQLite file (same DB_PATH), linked by runs.session_id ->
    history.id, but nothing ever actually joined them, so every row
    silently fell back to "—" / "0 chunks" / "No eval scores yet"
    forever, even on runs that completed successfully with real eval
    scores sitting one join away.

    LEFT JOIN, not INNER: a rejected/aborted run (see main.py's
    ConcurrencyLimitExceeded fix) never gets a session_id and legitimately
    has no history row — it should still appear in the list, just without
    eval data, rather than disappearing entirely.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                r.id, r.session_id, r.question, r.mode, r.status,
                r.error_message, r.created_at, r.model_used,
                r.total_latency_ms, r.total_input_tokens, r.total_output_tokens,
                r.estimated_cost_usd, r.rag_enabled, r.chunks_retrieved,
                COALESCE(h.latency_ms, r.total_latency_ms) AS latency_ms,
                COALESCE(h.rag_chunks_used, r.chunks_retrieved) AS rag_chunks_used,
                h.eval_overall AS eval_overall,
                h.eval_grade AS eval_grade
            FROM runs r
            LEFT JOIN history h ON h.id = r.session_id
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_node_breakdown(run_id: int) -> List[Dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM node_executions WHERE run_id = ? ORDER BY start_time_ms",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_node_averages(limit_runs: int = 50) -> List[Dict]:
    """
    Bug fix: metrics.html's "Avg latency by pipeline node" chart never
    called this data at all — it was hardcoded percentages of the overall
    average latency (planner 10%, search 35%, ...), a placeholder labeled
    "Simulate node breakdown" in a code comment that shipped as-is. Real
    per-node timing has been recorded in node_executions since Phase 3
    Milestone 1 (via Agent.__call__ -> RunTracker.node()) but nothing ever
    queried it in aggregate. This does: average latency, execution count,
    and failure count per node name, across the most recent N runs.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT node_name,
                   AVG(latency_ms) as avg_latency_ms,
                   COUNT(*) as executions,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
            FROM node_executions
            WHERE run_id IN (SELECT id FROM runs ORDER BY created_at DESC LIMIT ?)
            GROUP BY node_name
            ORDER BY avg_latency_ms DESC
            """,
            (limit_runs,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_aggregate_stats() -> Dict:
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_runs,
                AVG(total_latency_ms) as avg_latency_ms,
                SUM(estimated_cost_usd) as total_cost_usd,
                SUM(total_input_tokens) as total_input_tokens,
                SUM(total_output_tokens) as total_output_tokens,
                AVG(chunks_retrieved) as avg_chunks_retrieved
            FROM runs WHERE status = 'done'
        """).fetchone()
    return dict(row) if row else {}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _detect_model() -> str:
    """
    Bug fix: this only ever checked whether GROQ_API_KEY was *set*, never
    which provider the UI's model dropdown actually selected (POST
    /api/provider -> llm.set_provider(), a runtime choice) or whether
    Gemini was in play at all — every Gemini run was silently mislabeled
    as Ollama here, and once Groq's key was set, this returned "groq" even
    while the user had manually switched the dropdown to Ollama.
    """
    try:
        from .llm import get_selected_provider

        selected = get_selected_provider()
    except Exception:
        selected = "auto"

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GOOGLE_API_KEY", "").strip()

    order = [selected] if selected != "auto" else []
    if groq_key:
        order.append("groq")
    if gemini_key:
        order.append("gemini")
    order.append("ollama")

    for provider in order:
        if provider == "groq" and groq_key:
            return os.environ.get("GROQ_MODEL", "groq_llama3")
        if provider == "gemini" and gemini_key:
            return f"gemini/{os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')}"
        if provider == "ollama":
            return f"ollama/{os.environ.get('OLLAMA_MODEL', 'llama3.2')}"
    return f"ollama/{os.environ.get('OLLAMA_MODEL', 'llama3.2')}"


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if "ollama" in model:
        return 0.0
    # Bug fix: MODEL_COSTS' keys ("groq_llama3", "groq_llama3_70b") never
    # matched the actual value _detect_model() returned for Groq (the raw
    # GROQ_MODEL env var, e.g. "llama-3.1-8b-instant") — the dict lookup
    # below was an exact-match `.get()`, so it silently fell through to
    # MODEL_COSTS["default"] for every single Groq run regardless of which
    # Groq model was actually used, over- or under-pricing it. Matching by
    # substring against the real model identifier fixes that.
    lower = model.lower()
    if "gemini" in lower:
        tier = MODEL_COSTS["gemini"]
    elif "70b" in lower:
        tier = MODEL_COSTS["groq_llama3_70b"]
    elif "groq" in lower or "llama" in lower or "instant" in lower or "versatile" in lower:
        tier = MODEL_COSTS["groq_llama3"]
    else:
        tier = MODEL_COSTS["default"]
    return (input_tokens * tier["input"] + output_tokens * tier["output"]) / 1_000_000


def _save_node(record: Dict):
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO node_executions
                (run_id, node_name, start_time_ms, end_time_ms, latency_ms,
                 input_tokens, output_tokens, success, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record["run_id"],
                    record["node_name"],
                    record["start_time_ms"],
                    record["end_time_ms"],
                    record["latency_ms"],
                    record["input_tokens"],
                    record["output_tokens"],
                    record["success"],
                    record["error"],
                    record["metadata"],
                ),
            )
            conn.commit()
    except Exception as exc:
        print(f"[obs] failed to save node record: {exc}")


def _send_to_langsmith(tracker: RunTracker):
    """
    Optional: forward trace to LangSmith if configured.
    Requires LANGSMITH_API_KEY env var.
    """
    key = os.environ.get("LANGSMITH_API_KEY", "")
    if not key:
        return
    try:
        import requests

        project = os.environ.get("LANGSMITH_PROJECT", "research-agent")
        requests.post(
            "https://api.smith.langchain.com/runs",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={
                "name": "research_agent_run",
                "run_type": "chain",
                "inputs": {"question": tracker.question},
                "extra": {
                    "model": tracker.model_used,
                    "mode": tracker.mode,
                    "total_tokens": tracker.total_input_tokens + tracker.total_output_tokens,
                },
                "project_name": project,
            },
            timeout=5,
        )
    except Exception:
        pass  # Observability failures must never crash the agent

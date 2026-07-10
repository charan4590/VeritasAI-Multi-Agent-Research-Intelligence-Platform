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
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_node_breakdown(run_id: int) -> List[Dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM node_executions WHERE run_id = ? ORDER BY start_time_ms",
            (run_id,),
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
    if os.environ.get("GROQ_API_KEY"):
        return os.environ.get("GROQ_MODEL", "groq_llama3")
    return f"ollama/{os.environ.get('OLLAMA_MODEL', 'llama3.2')}"


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if "ollama" in model:
        return 0.0
    tier = MODEL_COSTS.get(model, MODEL_COSTS["default"])
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

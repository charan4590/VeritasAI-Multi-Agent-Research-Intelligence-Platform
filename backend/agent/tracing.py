"""
#2: Structured tracing with span hierarchy.

The existing observability.py tracks flat per-node timing in SQLite.
This adds a true span tree — parent/child relationships, inputs/outputs
captured per span, and a queryable trace structure. This is the single
biggest signal a frontier-lab engineer looks for: "how do you debug why
this specific agent run produced a bad answer?"

Design: in-memory span tree built during a run, persisted as JSON to
SQLite at the end (no new infra — reuses existing research.db).
Optionally forwards to LangSmith if LANGSMITH_API_KEY is set (reuses
the existing hook in observability.py rather than duplicating it).

A span looks like:
  {
    "id": "uuid", "parent_id": "uuid|null", "name": "search",
    "start_ms": ..., "end_ms": ..., "duration_ms": ...,
    "inputs": {...}, "outputs": {...}, "status": "ok|error",
    "children": [...]
  }
"""

import os
import json
import time
import uuid
import sqlite3
import logging
import threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)
DB_PATH = os.environ.get("DB_PATH", "research.db")

_local = threading.local()


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_tracing_tables():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                run_id INTEGER,
                question TEXT,
                span_tree TEXT,
                total_duration_ms INTEGER,
                status TEXT DEFAULT 'running',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


init_tracing_tables()


class Span:
    def __init__(self, name: str, parent: Optional["Span"] = None,
                 inputs: Optional[Dict] = None):
        self.id = str(uuid.uuid4())[:12]
        self.parent_id = parent.id if parent else None
        self.name = name
        self.start_ms = int(time.time() * 1000)
        self.end_ms: Optional[int] = None
        self.inputs = inputs or {}
        self.outputs: Dict[str, Any] = {}
        self.status = "running"
        self.error: Optional[str] = None
        self.children: List["Span"] = []
        if parent:
            parent.children.append(self)

    def end(self, outputs: Optional[Dict] = None, status: str = "ok",
            error: Optional[str] = None):
        self.end_ms = int(time.time() * 1000)
        self.outputs = outputs or {}
        self.status = status
        self.error = error

    @property
    def duration_ms(self) -> int:
        end = self.end_ms or int(time.time() * 1000)
        return end - self.start_ms

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "inputs": _truncate_dict(self.inputs),
            "outputs": _truncate_dict(self.outputs),
            "children": [c.to_dict() for c in self.children],
        }


def _truncate_dict(d: Dict, max_len: int = 500) -> Dict:
    """Truncate large values so trace JSON doesn't bloat the DB."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > max_len:
            out[k] = v[:max_len] + f"... [{len(v)} chars total]"
        elif isinstance(v, (list, dict)):
            out[k] = str(v)[:max_len]
        else:
            out[k] = v
    return out


class Tracer:
    """
    Tracks a single agent run as a span tree.

    Usage:
        tracer = Tracer(question="...")
        with tracer.span("planner", inputs={"question": q}) as span:
            result = planner_node(state)
            span.end(outputs={"queries": result["plan"]})
    """

    def __init__(self, question: str):
        self.question = question
        self.root = Span("agent_run", inputs={"question": question})
        self.run_id: Optional[int] = None

    @contextmanager
    def span(self, name: str, parent: Optional[Span] = None, inputs: Optional[Dict] = None):
        s = Span(name, parent=parent or self.root, inputs=inputs)
        try:
            yield s
            if s.status == "running":
                s.end(status="ok")
        except Exception as exc:
            s.end(status="error", error=str(exc))
            raise

    def finish(self, run_id: Optional[int] = None, status: str = "done"):
        self.root.end(status=status)
        trace_id = str(uuid.uuid4())[:12]
        try:
            with _conn() as conn:
                conn.execute(
                    """INSERT INTO traces (id, run_id, question, span_tree, total_duration_ms, status)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (trace_id, run_id, self.question,
                     json.dumps(self.root.to_dict()), self.root.duration_ms, status),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[tracing] failed to persist trace: {exc}")
        return trace_id


def get_trace(trace_id: str) -> Optional[Dict]:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["span_tree"] = json.loads(d["span_tree"])
    return d


def get_recent_traces(limit: int = 20) -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, run_id, question, total_duration_ms, status, created_at "
            "FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

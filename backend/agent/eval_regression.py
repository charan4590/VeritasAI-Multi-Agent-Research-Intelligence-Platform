"""
#7: Eval regression tracking.

Your benchmarking module (benchmark.py) compares approaches once.
This adds a CI-style regression gate: store a baseline score per
benchmark question, and on each new run, diff against the baseline
to detect silent quality degradation from a prompt/graph change.

Usage pattern:
  1. Run benchmark suite, call save_as_baseline() to lock in current scores
  2. After making a change, run benchmark suite again
  3. Call check_regression() — flags any question where score dropped
     more than REGRESSION_THRESHOLD points
"""

import os
import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)
DB_PATH = os.environ.get("DB_PATH", "research.db")
REGRESSION_THRESHOLD = int(os.environ.get("REGRESSION_THRESHOLD", "10"))  # points


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_eval_regression_tables():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                approach TEXT,
                relevance_score INTEGER,
                citation_score INTEGER,
                diversity_score INTEGER,
                overall_score INTEGER,
                baseline_version TEXT,
                is_current INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


init_eval_regression_tables()


def save_as_baseline(results: List[Dict], version_label: Optional[str] = None) -> str:
    """
    Lock in the given benchmark results as the new baseline.
    Marks any prior baseline rows as not-current rather than deleting them,
    so historical comparisons remain possible.
    """
    version = version_label or datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    with _conn() as conn:
        conn.execute("UPDATE eval_baselines SET is_current = 0")
        for r in results:
            conn.execute("""
                INSERT INTO eval_baselines
                (question, approach, relevance_score, citation_score,
                 diversity_score, overall_score, baseline_version, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                r.get("question", ""), r.get("approach", "agent"),
                r.get("relevance_score", 0), r.get("citation_score", 0),
                r.get("diversity_score", 0),
                r.get("overall_score", r.get("relevance_score", 0)),
                version,
            ))
        conn.commit()
    logger.info(f"[eval-regression] saved baseline '{version}' with {len(results)} results")
    return version


def get_current_baseline() -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_baselines WHERE is_current = 1"
        ).fetchall()
    return [dict(r) for r in rows]


def check_regression(new_results: List[Dict]) -> Dict:
    """
    Compare new benchmark results against the current baseline.
    Returns a report of any question whose score dropped more than
    REGRESSION_THRESHOLD points — the CI-gate signal.
    """
    baseline = {b["question"]: b for b in get_current_baseline()}
    if not baseline:
        return {
            "has_baseline": False,
            "regressions": [],
            "improvements": [],
            "message": "No baseline set yet. Run save_as_baseline() first.",
        }

    regressions = []
    improvements = []

    for r in new_results:
        q = r.get("question", "")
        base = baseline.get(q)
        if not base:
            continue
        new_score = r.get("overall_score", r.get("relevance_score", 0))
        old_score = base.get("overall_score", 0)
        delta = new_score - old_score

        if delta <= -REGRESSION_THRESHOLD:
            regressions.append({
                "question": q, "old_score": old_score,
                "new_score": new_score, "delta": delta,
            })
        elif delta >= REGRESSION_THRESHOLD:
            improvements.append({
                "question": q, "old_score": old_score,
                "new_score": new_score, "delta": delta,
            })

    return {
        "has_baseline": True,
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "passed": len(regressions) == 0,
        "message": (
            f"{len(regressions)} regression(s) detected"
            if regressions else "No regressions — all scores within threshold"
        ),
    }

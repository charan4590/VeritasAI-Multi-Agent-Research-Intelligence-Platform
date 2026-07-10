"""
#8: Rate limiting + cost guardrails.

Two protections that were missing:
  1. Per-IP request rate limiting — prevents one client from hammering
     Ollama/Tavily with concurrent requests and starving others.
  2. Hard caps already exist in graph.py (MAX_SOURCES, max_rounds) but
     this adds a global concurrent-run limiter so the whole server
     doesn't get overwhelmed if multiple users research simultaneously
     on limited local hardware (Ollama on a single CPU/GPU).

Simple in-memory implementation — no Redis needed for a single-instance
deployment. For multi-instance production you'd swap this for a shared
store, but the interface stays the same.
"""

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict

logger = logging.getLogger(__name__)

# Rate limiting: requests per window per IP
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX", "10"))

# Concurrency cap: max simultaneous research runs across all users
MAX_CONCURRENT_RUNS = int(os.environ.get("MAX_CONCURRENT_RUNS", "3"))

_request_log: Dict[str, list] = defaultdict(list)
_active_runs = 0
_lock = threading.Lock()


class RateLimitExceeded(Exception):
    pass


class ConcurrencyLimitExceeded(Exception):
    pass


def check_rate_limit(client_id: str):
    """
    Sliding-window rate limit per client (IP or session id).
    Raises RateLimitExceeded if the client has exceeded the allowed
    requests within the time window.
    """
    now = time.time()
    with _lock:
        timestamps = _request_log[client_id]
        # Drop timestamps outside the window
        timestamps[:] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]

        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {RATE_LIMIT_MAX_REQUESTS} requests per "
                f"{RATE_LIMIT_WINDOW_SECONDS}s. Try again shortly."
            )
        timestamps.append(now)


class ConcurrencyGuard:
    """
    Context manager enforcing MAX_CONCURRENT_RUNS across the whole server.
    Prevents Ollama (often single-threaded on a local CPU) from being
    overwhelmed by simultaneous research runs.

    Usage:
        with ConcurrencyGuard():
            run_the_graph()
    """

    def __enter__(self):
        global _active_runs
        with _lock:
            if _active_runs >= MAX_CONCURRENT_RUNS:
                raise ConcurrencyLimitExceeded(
                    f"Server is at capacity ({MAX_CONCURRENT_RUNS} concurrent research "
                    "runs). Please wait for an existing run to finish."
                )
            _active_runs += 1
            logger.info(f"[guardrails] active runs: {_active_runs}/{MAX_CONCURRENT_RUNS}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_runs
        with _lock:
            _active_runs = max(0, _active_runs - 1)
            logger.info(f"[guardrails] active runs: {_active_runs}/{MAX_CONCURRENT_RUNS}")
        return False


def get_guardrail_status() -> Dict:
    with _lock:
        return {
            "active_runs": _active_runs,
            "max_concurrent_runs": MAX_CONCURRENT_RUNS,
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "rate_limit_max_requests": RATE_LIMIT_MAX_REQUESTS,
        }

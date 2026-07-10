"""
Milestone 3: Caching layer.
============================
Architecture: a small CacheBackend interface (get/set/delete/clear) with
two implementations —

  DiskCacheBackend   — used when the optional `diskcache` package is
                       installed. Persists to disk (SQLite-backed), so the
                       cache survives process restarts and is already
                       thread-safe *and* process-safe with no extra work.

  InMemoryCacheBackend — pure-Python fallback (dict + threading.Lock) used
                       when `diskcache` isn't installed. Process-local and
                       lost on restart, but never corrupts under concurrent
                       access from this codebase's ThreadPoolExecutor-heavy
                       call sites (batched search, batched fetch, batched
                       embedding).

Why an interface at all, for a project this size: the guardrails.py module
already documents "swap this in-memory limiter for Redis when you go
multi-instance" as the intended production path — this cache is built the
same way. A future RedisCacheBackend only needs to implement get/set/
delete/clear; nothing in tools.py or rag.py would need to change.

On top of the backend sits `Cache`, a thin per-purpose wrapper that:
  - hashes the caller's raw key (callers never worry about key length or
    characters — a URL or a full search query is safe to pass directly)
  - applies a default TTL per cache (search/fetch/embed each get their
    own instance — see get_search_cache/get_fetch_cache/get_embed_cache)
  - tracks hits/misses/sets, independent of which backend is active
  - respects the global CACHE_ENABLED kill switch

Env vars:
  CACHE_ENABLED     — "true"/"false", default "true"
  CACHE_BACKEND     — "auto" (default) / "disk" / "memory"
  CACHE_DIR         — diskcache directory, default "../data/cache"
  SEARCH_CACHE_TTL  — seconds, default 1800 (30 min)
  FETCH_CACHE_TTL   — seconds, default 86400 (24 h)
  EMBED_CACHE_TTL   — seconds, default 604800 (7 days)
"""

import os
import time
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
SEARCH_CACHE_TTL = int(os.environ.get("SEARCH_CACHE_TTL", "1800"))
FETCH_CACHE_TTL = int(os.environ.get("FETCH_CACHE_TTL", "86400"))
EMBED_CACHE_TTL = int(os.environ.get("EMBED_CACHE_TTL", "604800"))
# Phase 3 Milestone 3: (claim sentence, source text) -> verdict is a
# near-deterministic judgment (same inputs, same LLM, same question asked)
# so it gets a long TTL, same as embeddings.
VERIFICATION_CACHE_TTL = int(os.environ.get("VERIFICATION_CACHE_TTL", "604800"))

# Sentinel distinguishing "key not present / expired" from "cached value is
# legitimately None or falsy" — a plain `None` return can't tell those apart.
_MISSING = object()


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class CacheBackend(ABC):
    """Minimal interface a cache backend must implement. Swap-in point for
    Redis (or anything else) later — nothing above this layer needs to
    change if a new backend implements these four methods."""

    @abstractmethod
    def get(self, key: str) -> Any:
        """Return the stored value, or _MISSING if absent/expired."""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class DiskCacheBackend(CacheBackend):
    """Persistent, thread-safe, process-safe backend via the `diskcache`
    package. Requires `pip install diskcache` (optional dependency —
    factory below falls back to InMemoryCacheBackend if it's missing)."""

    def __init__(self, directory: str):
        import diskcache  # local import: keep this optional at module load
        os.makedirs(directory, exist_ok=True)
        self._cache = diskcache.Cache(directory)

    def get(self, key: str) -> Any:
        return self._cache.get(key, default=_MISSING)

    def set(self, key: str, value: Any, ttl: int) -> None:
        self._cache.set(key, value, expire=ttl if ttl and ttl > 0 else None)

    def delete(self, key: str) -> None:
        self._cache.delete(key)

    def clear(self) -> None:
        self._cache.clear()


class InMemoryCacheBackend(CacheBackend):
    """Pure-Python fallback. A single threading.Lock guards the whole
    store — cache operations are microsecond-fast dict access, so one
    coarse-grained lock is not a real bottleneck, and it guarantees no
    torn reads/writes under this codebase's concurrent ThreadPoolExecutor
    call sites (this is the specific concern the milestone asked to be
    ensured against)."""

    def __init__(self):
        self._store: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return _MISSING
            value, expire_at = entry
            if expire_at is not None and time.time() > expire_at:
                # Expired — evict lazily on access rather than running a
                # background sweeper thread (fine at this scale).
                del self._store[key]
                return _MISSING
            return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        expire_at = (time.time() + ttl) if ttl and ttl > 0 else None
        with self._lock:
            self._store[key] = (value, expire_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Backend factory (singleton, double-checked locking)
# ---------------------------------------------------------------------------

_backend_singleton: Optional[CacheBackend] = None
_backend_lock = threading.Lock()


def _build_backend() -> CacheBackend:
    preference = os.environ.get("CACHE_BACKEND", "auto").lower()

    if preference in ("disk", "auto"):
        try:
            cache_dir = os.environ.get("CACHE_DIR", "../data/cache")
            backend = DiskCacheBackend(cache_dir)
            logger.info(f"[cache] using diskcache backend at {cache_dir!r}")
            return backend
        except ImportError:
            if preference == "disk":
                logger.warning(
                    "[cache] CACHE_BACKEND=disk but the 'diskcache' package "
                    "is not installed (pip install diskcache) — falling "
                    "back to in-memory cache."
                )
        except Exception as exc:
            logger.warning(f"[cache] failed to initialize diskcache ({exc}) — falling back to in-memory cache.")

    logger.info("[cache] using in-memory backend")
    return InMemoryCacheBackend()


def _get_backend() -> CacheBackend:
    global _backend_singleton
    if _backend_singleton is not None:
        return _backend_singleton
    with _backend_lock:
        if _backend_singleton is None:
            _backend_singleton = _build_backend()
        return _backend_singleton


# ---------------------------------------------------------------------------
# Cache — namespaced wrapper with hashed keys + hit/miss metrics
# ---------------------------------------------------------------------------

class Cache:
    """
    Per-purpose cache handle. Not instantiated directly by callers — use
    get_search_cache() / get_fetch_cache() / get_embed_cache() below, which
    share the single backend instance but keep independent namespaces,
    TTLs, and metrics.
    """

    def __init__(self, backend: CacheBackend, namespace: str, default_ttl: int):
        self._backend = backend
        self.namespace = namespace
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._metrics_lock = threading.Lock()

    def _full_key(self, raw_key: str) -> str:
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{digest}"

    def get(self, raw_key: str) -> Tuple[Any, bool]:
        """Returns (value, hit). value is None and hit is False on a miss —
        callers should always check `hit`, not just truthiness of value,
        since a legitimately cached value could itself be falsy."""
        if not CACHE_ENABLED:
            return None, False

        value = self._backend.get(self._full_key(raw_key))
        with self._metrics_lock:
            if value is _MISSING:
                self._misses += 1
                hit = False
            else:
                self._hits += 1
                hit = True

        if hit:
            logger.debug(f"[cache:{self.namespace}] HIT")
        else:
            logger.debug(f"[cache:{self.namespace}] MISS")
        return (value if hit else None), hit

    def set(self, raw_key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not CACHE_ENABLED:
            return
        self._backend.set(self._full_key(raw_key), value, ttl if ttl is not None else self.default_ttl)
        with self._metrics_lock:
            self._sets += 1

    def delete(self, raw_key: str) -> None:
        self._backend.delete(self._full_key(raw_key))

    def stats(self) -> Dict[str, Any]:
        with self._metrics_lock:
            hits, misses, sets = self._hits, self._misses, self._sets
        total = hits + misses
        hit_rate = round((hits / total) * 100, 1) if total else 0.0
        return {
            "namespace": self.namespace,
            "hits": hits,
            "misses": misses,
            "sets": sets,
            "hit_rate_pct": hit_rate,
            "default_ttl_seconds": self.default_ttl,
        }


# ---------------------------------------------------------------------------
# Per-purpose cache instances (singletons, double-checked locking)
# ---------------------------------------------------------------------------

_instances: Dict[str, Cache] = {}
_instance_lock = threading.Lock()

_TTLS = {
    "search": SEARCH_CACHE_TTL,
    "fetch": FETCH_CACHE_TTL,
    "embed": EMBED_CACHE_TTL,
    "verification": VERIFICATION_CACHE_TTL,
}


def _get_or_create(namespace: str) -> Cache:
    if namespace in _instances:
        return _instances[namespace]
    with _instance_lock:
        if namespace not in _instances:
            _instances[namespace] = Cache(_get_backend(), namespace=namespace, default_ttl=_TTLS[namespace])
        return _instances[namespace]


def get_search_cache() -> Cache:
    return _get_or_create("search")


def get_fetch_cache() -> Cache:
    return _get_or_create("fetch")


def get_embed_cache() -> Cache:
    return _get_or_create("embed")


def get_verification_cache() -> Cache:
    return _get_or_create("verification")


def get_all_cache_stats() -> List[Dict[str, Any]]:
    """Aggregate hit/miss metrics across all active caches. No route
    currently exposes this over HTTP (out of scope for this milestone —
    tools.py/rag.py/graph.py only) but it's here so a future
    /api/cache/stats endpoint, or a test/benchmark script, can call it
    directly."""
    return [
        get_search_cache().stats(), get_fetch_cache().stats(),
        get_embed_cache().stats(), get_verification_cache().stats(),
    ]

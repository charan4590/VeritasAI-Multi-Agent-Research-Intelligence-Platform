"""
Tools — web search with academic prioritization and full content fetch.

Changes from previous version:
  A. academic_web_search(): injects site-specific operators for academic queries
     so Tavily returns arxiv/IEEE/Springer results first
  B. fetch_full_content(): for high-credibility academic URLs, fetches the
     full page text rather than relying on Tavily's 1500-char snippet
  C. web_search() unchanged API — existing callers need no changes
  D. Multi-provider search: Tavily's free tier is only 1,000 queries/month
     with no auto-reset mid-cycle, so a single exhausted account used to
     mean the whole pipeline silently returned 0 sources. web_search() now
     tries a small ordered list of providers (see _get_search_provider_order)
     and falls through to the next one on failure — same shape as the
     Groq -> Gemini -> Ollama fallback in llm.py. Currently: Tavily, then
     Serper.dev (SERPER_API_KEY, free 2,500 queries, no card required —
     https://serper.dev). Force one explicitly with SEARCH_PROVIDER=tavily
     or SEARCH_PROVIDER=serper in .env; default "auto" tries whichever keys
     are present, Tavily first.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .cache import get_fetch_cache, get_search_cache

logger = logging.getLogger(__name__)

_client = None

# Last search failure reason, surfaced by SupervisorAgent into the
# activity log when a round comes back with 0 new sources — previously
# this was only ever written to the backend logger, so the UI showed a
# bare "gathered 0 sources" with no way to tell an invalid/expired
# TAVILY_API_KEY, an exhausted quota, or a network problem apart.
_last_search_error: Optional[str] = None


def get_last_search_error() -> Optional[str]:
    return _last_search_error


def _set_last_search_error(exc: Optional[BaseException]) -> None:
    global _last_search_error
    _last_search_error = str(exc) if exc else None


# Academic domains to prioritize — searched with site: operators when intent is academic
ACADEMIC_DOMAINS = [
    "arxiv.org",
    "ieee.org",
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "acm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "semanticscholar.org",
    "researchgate.net",
]

# Domains worth fetching full content from (beyond Tavily snippet)
FULL_CONTENT_DOMAINS = {
    "arxiv.org",
    "ieee.org",
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "acm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "semanticscholar.org",
}


def _get_client():
    global _client
    if _client is None:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set. Get a free key at https://tavily.com")
        _client = TavilyClient(api_key=api_key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False,
)
def _tavily_search_with_retry(query: str, max_results: int) -> List[Dict[str, Any]]:
    response = _get_client().search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
    )
    return response.get("results", [])


# Backwards-compatible alias — some tests/callers referenced the old name.
_search_with_retry = _tavily_search_with_retry


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False,
)
def _serper_search_with_retry(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Serper.dev — a Google-SERP-backed search API used as the fallback
    provider. Free tier: 2,500 queries, no credit card required
    (https://serper.dev). Response shape is normalized to match what
    Tavily returns ("url", "title", "content") so nothing downstream
    (academic_web_search, SupervisorAgent, etc.) needs to know which
    provider actually served a given result.
    """
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not set. Get a free key at https://serper.dev")

    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in (data.get("organic") or [])[:max_results]:
        results.append(
            {
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "content": item.get("snippet", ""),
            }
        )
    return results


_SEARCH_PROVIDERS = {
    "tavily": lambda q, n: _tavily_search_with_retry(q, n),
    "serper": lambda q, n: _serper_search_with_retry(q, n),
}


def _get_search_provider_order() -> List[str]:
    """
    SEARCH_PROVIDER=auto (default): try whichever providers have a key
    configured, Tavily first, then Serper. SEARCH_PROVIDER=tavily or
    =serper: try that one first, then fall back to the other if it fails
    (missing key, quota exhausted, network error, etc.) — mirrors the
    manual-provider-with-fallback behavior in llm.get_llm().
    """
    requested = os.environ.get("SEARCH_PROVIDER", "auto").strip().lower()
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    serper_key = os.environ.get("SERPER_API_KEY", "").strip()

    auto_order = []
    if tavily_key:
        auto_order.append("tavily")
    if serper_key:
        auto_order.append("serper")

    if requested in _SEARCH_PROVIDERS:
        return [requested] + [p for p in auto_order if p != requested]
    return auto_order


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Standard web search with retry + multi-provider fallback. Returns
    empty list only if every configured provider fails.

    Milestone 3: results are cached (namespace "search", default TTL
    SEARCH_CACHE_TTL). Only a *successful* result is cached — a failure
    falls straight through without touching the cache, so a transient
    outage (or an exhausted Tavily quota that later resets) can't get
    "stuck" returning an empty list for the full TTL window.
    """
    cache = get_search_cache()
    cache_key = f"{query}|{max_results}"
    cached, hit = cache.get(cache_key)
    if hit:
        return cached

    providers = _get_search_provider_order()
    if not providers:
        msg = (
            "No search provider configured — set TAVILY_API_KEY "
            "(https://tavily.com) and/or SERPER_API_KEY (https://serper.dev) in .env"
        )
        logger.error(f"[search] {msg}")
        _set_last_search_error(msg)
        return []

    last_exc: Optional[BaseException] = None
    for provider in providers:
        try:
            results = _SEARCH_PROVIDERS[provider](query, max_results) or []
            cache.set(cache_key, results)
            _set_last_search_error(None)
            return results
        except Exception as exc:
            last_exc = exc
            logger.warning(f"[search] provider '{provider}' failed for {query!r}: {exc}")

    logger.error(f"[search] all providers ({providers}) permanently failed: {query!r} — {last_exc}")
    _set_last_search_error(last_exc)
    return []


def academic_web_search(query: str, max_results: int = 7) -> List[Dict[str, Any]]:
    """
    Academic-optimized search.

    Strategy:
      1. Run the query as-is (catches recent preprints, blog summaries)
      2. Run the same query with "site:arxiv.org OR site:ieee.org" prefix
      3. Merge, deduplicate, sort academic sources to top

    Why two passes: Tavily doesn't support multi-site operators reliably.
    Running the plain query first ensures we catch relevant recent papers
    that may be mirrored or cited on other sites.
    """
    all_results = []
    seen_urls = set()

    # Pass 1: plain query
    plain = web_search(query, max_results=max_results)
    for r in plain:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_results.append(r)

    # Pass 2: academic-targeted query
    academic_query = (
        f"{query} site:arxiv.org OR site:ieee.org OR site:springer.com OR site:pubmed.ncbi.nlm.nih.gov"
    )
    academic = web_search(academic_query, max_results=max_results)
    for r in academic:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_results.append(r)

    # Sort: academic domains first, then by score/relevance
    def _sort_key(result):
        url = result.get("url", "").lower()
        for domain in ACADEMIC_DOMAINS:
            if domain in url:
                return 0  # academic sources first
        return 1

    all_results.sort(key=_sort_key)
    return all_results[: max_results * 2]  # return more for academic queries


def academic_web_search_batch(queries: List[str], max_results: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run multiple academic searches concurrently instead of sequentially.
    5 planned queries running one-at-a-time was a major latency source —
    this cuts search time roughly to the duration of the slowest single query
    instead of the sum of all of them.
    """
    from concurrent.futures import ThreadPoolExecutor

    results: Dict[str, List[Dict[str, Any]]] = {}
    if not queries:
        return results
    with ThreadPoolExecutor(max_workers=min(5, len(queries))) as executor:
        future_map = {executor.submit(academic_web_search, q, max_results): q for q in queries}
        for future in future_map:
            q = future_map[future]
            try:
                results[q] = future.result()
            except Exception as exc:
                logger.error(f"[search] concurrent query failed: {q!r} — {exc}")
                results[q] = []
    return results


def web_search_batch(queries: List[str], max_results: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Concurrent version of plain web_search for general/technical queries."""
    from concurrent.futures import ThreadPoolExecutor

    results: Dict[str, List[Dict[str, Any]]] = {}
    if not queries:
        return results
    with ThreadPoolExecutor(max_workers=min(5, len(queries))) as executor:
        future_map = {executor.submit(web_search, q, max_results): q for q in queries}
        for future in future_map:
            q = future_map[future]
            try:
                results[q] = future.result()
            except Exception as exc:
                logger.error(f"[search] concurrent query failed: {q!r} — {exc}")
                results[q] = []
    return results


def fetch_full_content(url: str, timeout: int = 8) -> Optional[str]:
    """
    Fetch fuller text content from a URL.
    Used for high-credibility academic sources where Tavily's snippet
    misses the methodology/results sections.

    Returns extracted text or None if fetch fails.
    Only fetches from known academic domains to avoid privacy/legal issues.

    Milestone 3: cached by URL alone (namespace "fetch", default TTL
    FETCH_CACHE_TTL — 24h by default, since page content for these domains
    is effectively static hour-to-hour). `timeout` is intentionally not
    part of the cache key: it affects how long we're willing to wait for
    a given fetch, not what content comes back. Only a genuinely
    successful, long-enough extraction is cached — disallowed domains,
    too-short content, and errors all fall through without writing to the
    cache, so they're retried (not "stuck") on the next call.
    """
    cache = get_fetch_cache()
    cached, hit = cache.get(url)
    if hit:
        return cached

    try:
        from urllib.parse import urlparse

        domain = urlparse(url).netloc.replace("www.", "")

        if not any(d in domain for d in FULL_CONTENT_DOMAINS):
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Research Agent; Academic Use)",
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        text = resp.text
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 500:
            return None

        content = text[:3000]
        cache.set(url, content)
        return content

    except Exception as exc:
        logger.debug(f"[fetch] failed for {url}: {exc}")
        return None

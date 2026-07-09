"""
Tools — web search with academic prioritization and full content fetch.

Changes from previous version:
  A. academic_web_search(): injects site-specific operators for academic queries
     so Tavily returns arxiv/IEEE/Springer results first
  B. fetch_full_content(): for high-credibility academic URLs, fetches the
     full page text rather than relying on Tavily's 1500-char snippet
  C. web_search() unchanged API — existing callers need no changes
"""

import os
import re
import logging
import requests
from typing import List, Dict, Any, Optional
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)

logger = logging.getLogger(__name__)

_client = None

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
    "arxiv.org", "ieee.org", "springer.com", "nature.com",
    "sciencedirect.com", "acm.org", "pubmed.ncbi.nlm.nih.gov",
    "semanticscholar.org",
}


def _get_client():
    global _client
    if _client is None:
        from tavily import TavilyClient
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Get a free key at https://tavily.com"
            )
        _client = TavilyClient(api_key=api_key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False,
)
def _search_with_retry(query: str, max_results: int) -> List[Dict[str, Any]]:
    response = _get_client().search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
    )
    return response.get("results", [])


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Standard web search with retry. Returns empty list on failure.
    """
    try:
        return _search_with_retry(query, max_results) or []
    except Exception as exc:
        logger.error(f"[search] permanently failed: {query!r} — {exc}")
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
    academic_query = f"{query} site:arxiv.org OR site:ieee.org OR site:springer.com OR site:pubmed.ncbi.nlm.nih.gov"
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
    return all_results[:max_results * 2]  # return more for academic queries


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
        future_map = {
            executor.submit(academic_web_search, q, max_results): q
            for q in queries
        }
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
        future_map = {
            executor.submit(web_search, q, max_results): q
            for q in queries
        }
        for future in future_map:
            q = future_map[future]
            try:
                results[q] = future.result()
            except Exception as exc:
                logger.error(f"[search] concurrent query failed: {q!r} — {exc}")
                results[q] = []
    return results
    """
    Fetch fuller text content from a URL.
    Used for high-credibility academic sources where Tavily's snippet
    misses the methodology/results sections.

    Returns extracted text or None if fetch fails.
    Only fetches from known academic domains to avoid privacy/legal issues.
    """
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")

        # Only fetch from trusted academic domains
        if not any(d in domain for d in FULL_CONTENT_DOMAINS):
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Research Agent; Academic Use)",
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        # Extract text — strip HTML tags, keep content
        text = resp.text
        # Remove script/style blocks
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Return meaningful content (skip if mostly navigation/boilerplate)
        if len(text) < 500:
            return None

        # Return up to 3000 chars of main content
        return text[:3000]

    except Exception as exc:
        logger.debug(f"[fetch] failed for {url}: {exc}")
        return None


def fetch_full_content(url: str, timeout: int = 8) -> Optional[str]:
    """
    Fetch fuller text content from a URL.
    Used for high-credibility academic sources where Tavily's snippet
    misses the methodology/results sections.

    Returns extracted text or None if fetch fails.
    Only fetches from known academic domains to avoid privacy/legal issues.
    """
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

        return text[:3000]

    except Exception as exc:
        logger.debug(f"[fetch] failed for {url}: {exc}")
        return None

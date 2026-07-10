"""
Multi-agent debate mode.
Runs one search pass to gather sources, then two synthesis agents
(FOR and AGAINST) to produce a balanced debate report.
"""

import json
import re
from typing import Dict

from .llm import get_llm
from .tools import web_search


def parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def run_debate(question: str, log_cb=None) -> dict:
    """Run a two-agent debate on the question. Returns report + sources."""

    def log(msg):
        if log_cb:
            log_cb(msg)

    # Step 1 — Plan search queries
    log("Planning debate search queries...")
    llm = get_llm()
    r = llm.invoke(
        [
            (
                "system",
                "You are a research planner. Respond ONLY with JSON.\n"
                '{"queries": ["query1", "query2", "query3"]}',
            ),
            ("human", f"Research both sides of: {question}"),
        ]
    )
    data = parse_json(r.content)
    queries = data.get("queries", [question])

    # Step 2 — Search
    log(f"Searching {len(queries)} queries for debate evidence...")
    sources: Dict[int, dict] = {}
    seen_urls = set()
    next_id = 1
    for query in queries:
        for result in web_search(query):
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources[next_id] = {
                "id": next_id,
                "url": url,
                "title": result.get("title", url),
                "snippet": (result.get("content", "") or "")[:800],
            }
            next_id += 1

    log(f"Gathered {len(sources)} sources for debate")

    sources_block = "\n\n".join(
        f"[{s['id']}] {s['title']}\n{s['url']}\n{s['snippet']}" for s in sources.values()
    )

    # Step 3 — FOR agent
    log("Agent A arguing FOR the topic...")
    for_response = llm.invoke(
        [
            (
                "system",
                "You argue strongly FOR the topic in markdown. "
                "Use [n] citations from the provided sources. "
                "Use heading '## Arguments For' and short paragraphs.",
            ),
            ("human", f"Topic: {question}\n\nSources:\n{sources_block}"),
        ]
    )

    # Step 4 — AGAINST agent
    log("Agent B arguing AGAINST the topic...")
    against_response = llm.invoke(
        [
            (
                "system",
                "You argue strongly AGAINST the topic in markdown. "
                "Use [n] citations from the provided sources. "
                "Use heading '## Arguments Against' and short paragraphs.",
            ),
            ("human", f"Topic: {question}\n\nSources:\n{sources_block}"),
        ]
    )

    # Step 5 — Combine
    log("Combining debate perspectives...")
    report = (
        f"# Debate: {question}\n\n"
        f"> Two AI agents researched this topic and argued opposite sides "
        f"using the same {len(sources)} sources.\n\n"
        f"{for_response.content}\n\n"
        f"---\n\n"
        f"{against_response.content}"
    )

    # Step 6 — Validate citations
    CITATION_RE = re.compile(r"\[(\d+)\]")
    valid_ids = set(sources.keys())
    found_ids = {int(m) for m in CITATION_RE.findall(report)}
    used_ids = sorted(found_ids & valid_ids)
    for bad in found_ids - valid_ids:
        report = re.sub(rf"\[{bad}\]", "", report)

    if used_ids:
        refs = "\n".join(f"[{i}] {sources[i]['title']} — {sources[i]['url']}" for i in used_ids)
        report += f"\n\n---\n\n**Sources**\n\n{refs}"

    log(f"Debate complete — {len(used_ids)} citations validated")

    return {
        "report": report,
        "sources": {str(k): v for k, v in sources.items()},
        "citations_used": used_ids,
    }

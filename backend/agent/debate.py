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

    # Citation-density rules shared by both sides. Bug fix: debate mode's
    # confidence score is compute_confidence() from credibility.py, which
    # is avg_credibility_of_cited_sources * min(1, citations_used/3) — so
    # a debate that only ever cited 1-2 sources (easy to do when the
    # prompt just says "use [n] citations" with no minimum) was capped at
    # 33-66% of its deserved score regardless of how good those sources
    # were. This mirrors the same fix already applied to the main
    # research pipeline's report_generator.py prompts: explicitly require
    # citing several distinct sources and citing every paragraph that
    # makes a factual claim, so citations_used actually reflects how much
    # evidence was gathered instead of undershooting it.
    citation_rules = (
        "Cite [n] from the provided numbered sources. Draw on AT LEAST "
        "4-5 DIFFERENT numbered sources if that many are available — "
        "don't lean on the same 1-2 sources for the whole argument. "
        "Every paragraph making a factual claim must include at least "
        "one [n] citation; a paragraph with no citation should only be "
        "used for framing/transition, not for evidence."
    )

    # Step 3 — FOR agent
    log("Agent A arguing FOR the topic...")
    for_response = llm.invoke(
        [
            (
                "system",
                "You argue strongly FOR the topic in markdown. "
                f"{citation_rules} "
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
                f"{citation_rules} "
                "Use heading '## Arguments Against' and short paragraphs.",
            ),
            ("human", f"Topic: {question}\n\nSources:\n{sources_block}"),
        ]
    )

    # Step 5 — Verdict
    # Bug fix: debate mode used to stop at "Arguments For" / "Arguments
    # Against" with no synthesis at all — a real question like "does
    # diesel outperform petrol" got two one-sided arguments and no actual
    # answer. This step reads both sides plus the same sources and forces
    # an honest conclusion: lean one way if the evidence actually
    # supports it, or say plainly that it's genuinely mixed/context-
    # dependent and explain what would tip it — but never just restate
    # both sides without concluding anything.
    log("Weighing both sides for a final verdict...")
    verdict_response = llm.invoke(
        [
            (
                "system",
                "You are a neutral judge weighing two opposing arguments, both "
                "already grounded in the same cited sources. Write a final "
                "verdict in markdown under the heading '## Verdict'.\n"
                "RULES:\n"
                "1. Give an actual answer, not a restatement of both sides. If "
                "the evidence reasonably supports leaning one way, say so "
                "plainly in your first sentence, then explain why in 2-4 "
                "sentences.\n"
                "2. If the honest answer is genuinely 'it depends,' say that "
                "explicitly and name the specific factors that would tip it "
                "one way or the other — not as a way to avoid answering, but "
                "as the actual conclusion.\n"
                "3. Cite [n] from the sources used in the arguments above.\n"
                "4. Do not introduce new claims not covered by either side's "
                "arguments — you are weighing what's already been argued, not "
                "researching further.",
            ),
            (
                "human",
                f"Topic: {question}\n\n"
                f"FOR argument:\n{for_response.content}\n\n"
                f"AGAINST argument:\n{against_response.content}",
            ),
        ]
    )

    # Step 6 — Combine
    log("Combining debate perspectives...")
    report = (
        f"# Debate: {question}\n\n"
        f"> Two AI agents researched this topic and argued opposite sides "
        f"using the same {len(sources)} sources, then a third pass weighed "
        f"both to reach a verdict.\n\n"
        f"{verdict_response.content}\n\n"
        f"---\n\n"
        f"{for_response.content}\n\n"
        f"---\n\n"
        f"{against_response.content}"
    )

    # Step 7 — Validate citations
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

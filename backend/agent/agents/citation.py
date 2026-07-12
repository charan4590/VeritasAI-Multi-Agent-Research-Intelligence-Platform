"""
CitationAgent — Phase 3 Milestone 1, extended in Phase 3 Milestone 2 and
Phase 6 (final polish, including a real-world bug fix below).

Moved from graph.py's validate_node. Pure-Python citation validation
(strip any [n] marker that doesn't map to a real retrieved source, append
a references list built only from citations that survived) is
byte-for-byte unchanged from before this milestone. No LLM call here —
same as before.

Milestone 2: the references list tags PDF-derived sources with a
"[PDF]" marker so the report visibly distinguishes uploaded-document
evidence from web evidence. Citation numbering itself is untouched: PDF
sources share the exact same [n] sequence as web sources (see
SupervisorAgent), this only affects how a given [n] is *displayed*.

Phase 6: the footer is now a proper IEEE-style "References" section
(format_ieee_reference below), not an ad hoc "Sources" list. This is the
single formatter RevisionAgent also uses when it rebuilds the footer
post-grounding — one shared function guarantees the exact same citation
renders identically everywhere, which is what "stable numbering" actually
requires: formatting depends only on (citation_id, source), never on
which agent or which pass happens to be building the list.

Phase 6 bug fix (found via a real production run, not a synthetic test):
the academic synthesis prompt (report_generator.py) explicitly instructs
the model to write its own "## 9. References" section as part of its
rigid template. Left alone, that produces TWO disagreeing References
sections in the same report once this agent appends its own validated
one below. _strip_llm_references_section() removes the model's version
first — it's unvalidated model output (frequently containing citation
numbers/URLs the model invented), not the ground-truth list this agent
is about to build. This also prevents a second, subtler bug: a
References line like "[1] Transfer learning based..." would otherwise
be picked up by CITATION_RE as if the report were "citing" source 1
again right there, contaminating citations_used.
"""

import re

from ..state import AgentState
from .base import Agent

CITATION_RE = re.compile(r"\[(\d+)\]")

REFERENCE_SECTION_HEADER_RE = re.compile(
    r"^##\s+(?:\d+\.\s*)?(?:References?|Bibliography|Works Cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_llm_references_section(report: str) -> str:
    """Removes a model-generated References/Bibliography section (see
    module docstring). No-op if the report never had one — most reports
    (general/technical intent) don't use a template that asks for one."""
    match = REFERENCE_SECTION_HEADER_RE.search(report)
    if not match:
        return report

    before = report[: match.start()].rstrip()
    rest = report[match.end() :]
    next_header = re.search(r"\n##\s+", rest)
    if next_header:
        after = rest[next_header.start() :].lstrip("\n")
        return before + "\n\n" + after
    return before


def format_ieee_reference(citation_id: int, source: dict) -> str:
    """
    Simplified IEEE reference style: `[n] "Title," [Online]. Available:
    URL`. This is the standard simplified form used when author/
    publication-year metadata isn't available -- which Tavily search
    results and PDF-ingestion chunks never reliably provide, so a full
    `[n] A. Author, "Title," Venue, Year.` citation isn't achievable
    honestly here. Full (untruncated) title is always used -- this is
    the References section, not a compact citation badge, so there's no
    space constraint pushing toward abbreviation.
    """
    title = (source.get("title") or source.get("url", "")).strip()
    url = source.get("url", "")
    tag = "[PDF] " if source.get("source_type", "web") == "pdf" else ""
    return f'[{citation_id}] {tag}"{title}," [Online]. Available: {url}'


class CitationAgent(Agent):
    """Strips hallucinated citation markers (ids that don't correspond to
    a real retrieved source) and appends an IEEE-style References list
    built only from citations that survived validation — PDF-derived
    sources are visibly tagged in that list."""

    name = "validate"

    def trace_inputs(self, state: AgentState):
        return {"source_count": len(state.get("sources", {}))}

    def run(self, state: AgentState) -> dict:
        report = _strip_llm_references_section(state["report"])
        valid_ids = set(state["sources"].keys())
        found_ids = {int(m) for m in CITATION_RE.findall(report)}
        used_ids = sorted(found_ids & valid_ids)

        for bad in found_ids - valid_ids:
            report = re.sub(rf"\[{bad}\]", "", report)

        pdf_citations_count = sum(
            1 for i in used_ids if state["sources"][i].get("source_type", "web") == "pdf"
        )

        if used_ids:
            # Defensive dedup by URL: SupervisorAgent's seen_urls set
            # already guarantees one URL never gets two different
            # citation ids during search, so this should never actually
            # remove anything in practice -- but "deduplicate references"
            # is a correctness property worth guaranteeing explicitly at
            # the point references are rendered, not just assumed from
            # an invariant upstream that a future change could break.
            seen_urls = set()
            deduped_ids = []
            for i in used_ids:
                url = state["sources"][i].get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                deduped_ids.append(i)

            refs = "\n".join(format_ieee_reference(i, state["sources"][i]) for i in deduped_ids)
            report += f"\n\n---\n\n**References**\n\n{refs}"
        else:
            report += "\n\n---\n\n*No verifiable citations produced.*"

        hallucinated = found_ids - valid_ids
        msg = f"Validated {len(used_ids)} citations"
        if pdf_citations_count:
            msg += f" ({pdf_citations_count} from uploaded PDFs)"
        if hallucinated:
            msg += f" — removed {len(hallucinated)} hallucinated"

        return {
            "report": report,
            "citations_used": used_ids,
            "log": state["log"] + [msg],
        }

        for bad in found_ids - valid_ids:
            report = re.sub(rf"\[{bad}\]", "", report)

        pdf_citations_count = sum(
            1 for i in used_ids if state["sources"][i].get("source_type", "web") == "pdf"
        )

        if used_ids:
            # Defensive dedup by URL: SupervisorAgent's seen_urls set
            # already guarantees one URL never gets two different
            # citation ids during search, so this should never actually
            # remove anything in practice -- but "deduplicate references"
            # is a correctness property worth guaranteeing explicitly at
            # the point references are rendered, not just assumed from
            # an invariant upstream that a future change could break.
            seen_urls = set()
            deduped_ids = []
            for i in used_ids:
                url = state["sources"][i].get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                deduped_ids.append(i)

            refs = "\n".join(format_ieee_reference(i, state["sources"][i]) for i in deduped_ids)
            report += f"\n\n---\n\n**References**\n\n{refs}"
        else:
            report += "\n\n---\n\n*No verifiable citations produced.*"

        hallucinated = found_ids - valid_ids
        msg = f"Validated {len(used_ids)} citations"
        if pdf_citations_count:
            msg += f" ({pdf_citations_count} from uploaded PDFs)"
        if hallucinated:
            msg += f" — removed {len(hallucinated)} hallucinated"

        return {
            "report": report,
            "citations_used": used_ids,
            "log": state["log"] + [msg],
        }

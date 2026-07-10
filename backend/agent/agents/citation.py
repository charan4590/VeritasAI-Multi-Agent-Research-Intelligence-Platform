"""
CitationAgent — Phase 3 Milestone 1, extended in Phase 3 Milestone 2.

Moved from graph.py's validate_node. Pure-Python citation validation
(strip any [n] marker that doesn't map to a real retrieved source, append
a references list built only from citations that survived) is
byte-for-byte unchanged from before this milestone. No LLM call here —
same as before.

Milestone 2: the references list now tags PDF-derived sources with a
"[PDF]" marker so the report visibly distinguishes uploaded-document
evidence from web evidence — the only change from before. Citation
numbering itself is untouched: PDF sources share the exact same [n]
sequence as web sources (see SupervisorAgent), this only affects how a
given [n] is *displayed* in the reference list.
"""

import re

from ..state import AgentState
from .base import Agent

CITATION_RE = re.compile(r"\[(\d+)\]")


class CitationAgent(Agent):
    """Strips hallucinated citation markers (ids that don't correspond to
    a real retrieved source) and appends a references list built only
    from citations that survived validation — PDF-derived sources are
    visibly tagged in that list."""

    name = "validate"

    def trace_inputs(self, state: AgentState):
        return {"source_count": len(state.get("sources", {}))}

    def run(self, state: AgentState) -> dict:
        report = state["report"]
        valid_ids = set(state["sources"].keys())
        found_ids = {int(m) for m in CITATION_RE.findall(report)}
        used_ids = sorted(found_ids & valid_ids)

        for bad in found_ids - valid_ids:
            report = re.sub(rf"\[{bad}\]", "", report)

        pdf_citations_count = sum(
            1 for i in used_ids if state["sources"][i].get("source_type", "web") == "pdf"
        )

        if used_ids:

            def _format_ref(i: int) -> str:
                s = state["sources"][i]
                tag = "[PDF] " if s.get("source_type", "web") == "pdf" else ""
                return f"[{i}] {tag}{s['title']} — {s['url']}"

            refs = "\n".join(_format_ref(i) for i in used_ids)
            report += f"\n\n---\n\n**Sources**\n\n{refs}"
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

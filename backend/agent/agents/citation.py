"""
CitationAgent — Phase 3 Milestone 1.

Moved from graph.py's validate_node. Pure-Python citation validation
(strip any [n] marker that doesn't map to a real retrieved source, append
a references list built only from citations that survived) is
byte-for-byte unchanged from before this milestone. No LLM call here —
same as before.
"""

import re

from ..state import AgentState
from .base import Agent

CITATION_RE = re.compile(r"\[(\d+)\]")


class CitationAgent(Agent):
    """Strips hallucinated citation markers (ids that don't correspond to
    a real retrieved source) and appends a references list built only
    from citations that survived validation."""

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

        if used_ids:
            refs = "\n".join(
                f"[{i}] {state['sources'][i]['title']} — {state['sources'][i]['url']}"
                for i in used_ids
            )
            report += f"\n\n---\n\n**Sources**\n\n{refs}"
        else:
            report += "\n\n---\n\n*No verifiable citations produced.*"

        hallucinated = found_ids - valid_ids
        msg = f"Validated {len(used_ids)} citations"
        if hallucinated:
            msg += f" — removed {len(hallucinated)} hallucinated"

        return {
            "report": report,
            "citations_used": used_ids,
            "log": state["log"] + [msg],
        }

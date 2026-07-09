"""
WebResearchAgent / AcademicSearchAgent — Phase 3 Milestone 1.

These are the two search *strategies* the pipeline can use — general web
search vs. academic-site-prioritized search — each wrapping the batch
search functions tools.py already provides (and which Milestone 3 already
made cache-aware; nothing about caching changes here).

Neither of these is registered as its own LangGraph node: the graph only
ever had a single "search" node, and that topology is preserved exactly
in this milestone (see graph.py / agents/supervisor.py). SupervisorAgent
picks one of these two agents based on detected intent and owns the
shared fetch/splice/sort work that follows — that split of
responsibility (this file: "how do I search"; supervisor.py: "what do I
do with the results") is what keeps each class single-purpose.

Each agent exposes two things:
  - search(queries) -> the precise method the live pipeline actually
    calls (via SupervisorAgent), taking exactly the queries to search —
    correct for both the initial round (state["plan"]) and follow-up
    rounds (reflection.follow_up_queries), which a naive `run(state)`
    reading state["plan"] alone would get wrong on a follow-up round.
  - run(state) -> dict — satisfies the Agent interface so each class is
    independently testable/callable on its own with just a state object
    (`WebResearchAgent().run(initial_state("some question"))`), for
    ad-hoc use or a follow-up round outside the graph.
"""

from typing import Any, Dict, List

from ..state import AgentState
from ..tools import web_search_batch, academic_web_search_batch
from .base import Agent


class WebResearchAgent(Agent):
    """General/technical web search — no site restrictions."""

    name = "web_research"

    def search(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        return web_search_batch(queries, max_results=5)

    def run(self, state: AgentState) -> dict:
        return {"results": self.search(state.get("plan", []))}


class AcademicSearchAgent(Agent):
    """Academic-prioritized search: two-pass query (plain + site-restricted
    to arxiv/IEEE/Springer/PubMed) with academic domains sorted first."""

    name = "academic_search"

    def search(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        return academic_web_search_batch(queries, max_results=4)

    def run(self, state: AgentState) -> dict:
        return {"results": self.search(state.get("plan", []))}

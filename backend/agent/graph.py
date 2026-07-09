"""
Research agent graph — production version with academic research optimization.

Phase 3 Milestone 1: node business logic now lives in backend/agent/agents/
as Agent subclasses (see that package's __init__.py docstring for the
architecture). This file's job has shrunk to exactly what a "graph" file
should do: build the LangGraph StateGraph and wire edges. It no longer
contains any agent business logic itself — that was true of five of the
six nodes before this milestone lived here as plain functions; now it's
true of the file as a whole.

Pipeline (topology byte-for-byte unchanged from before this milestone):
  planner → search → reflect → rag → synthesize → validate
                        ^__________|  (loop back to search while insufficient,
                                       capped at max_rounds)

`reflect` stays a plain function here, not an Agent subclass, and is
deliberately NOT wrapped with tracker/tracer instrumentation in this
milestone — that's an exact, intentional scope boundary from the Phase 3
plan (a ReflectAgent, Fact Verification Agent, Risk Analysis Agent, and
Knowledge Graph Agent are all explicitly future milestones, not this one).
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from .state import AgentState, ReflectionDecision
from .reflection import smart_reflect
from .agents import (
    PlannerAgent, SupervisorAgent, RAGAgent, ReportGeneratorAgent, CitationAgent,
    _detect_research_intent,  # re-exported: main.py does `from agent.graph import _detect_research_intent`
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reflect — unchanged from before this milestone. Not yet converted to an
# Agent subclass; see module docstring above.
# ---------------------------------------------------------------------------

def reflect_node(state: AgentState) -> dict:
    if state["round"] >= state["max_rounds"]:
        decision = ReflectionDecision(
            sufficient=True, follow_up_queries=[],
            reasoning="Reached maximum search rounds."
        )
        return {
            "reflection": decision,
            "log": state["log"] + ["Max rounds reached — moving to RAG + synthesis"],
        }

    decision = smart_reflect(
        question=state["question"],
        sources=state["sources"],
        current_round=state["round"],
        max_rounds=state["max_rounds"],
    )

    status = "sufficient" if decision.sufficient else f"gap: {decision.reasoning}"
    logger.info(f"Reflection round {state['round']}: {status}")

    return {
        "reflection": decision,
        "log": state["log"] + [f"Reflection: {decision.reasoning}"],
    }


def route_after_reflect(state: AgentState) -> str:
    r = state.get("reflection")
    return "search" if (r and not r.sufficient) else "rag"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
# Node names and edges are identical to the pre-Milestone-1 graph — only
# the objects registered for planner/search/rag/synthesize/validate changed
# (Agent instances instead of bare functions; LangGraph calls both the
# same way, via `node(state) -> dict`, so this is a drop-in swap).

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", PlannerAgent())
    g.add_node("search", SupervisorAgent())
    g.add_node("reflect", reflect_node)
    g.add_node("rag", RAGAgent())
    g.add_node("synthesize", ReportGeneratorAgent())
    g.add_node("validate", CitationAgent())

    g.add_edge(START, "planner")
    g.add_edge("planner", "search")
    g.add_edge("search", "reflect")
    g.add_conditional_edges(
        "reflect", route_after_reflect,
        {"search": "search", "rag": "rag"},
    )
    g.add_edge("rag", "synthesize")
    g.add_edge("synthesize", "validate")
    g.add_edge("validate", END)
    return g.compile()


def initial_state(question: str, max_rounds: int = 2,
                  memories: Optional[list] = None,
                  stream_callback=None,
                  tracker=None,
                  tracer=None) -> AgentState:
    return AgentState(
        question=question, plan=[], sources={}, round=0,
        max_rounds=max_rounds, reflection=None,
        report="", citations_used=[], log=[],
        retrieved_chunks=[], rag_session_id=None,
        memories=memories or [],
        stream_callback=stream_callback,
        tracker=tracker,
        tracer=tracer,
    )

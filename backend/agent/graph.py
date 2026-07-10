"""
Research agent graph — production version with academic research optimization.

Phase 3 Milestone 1: node business logic now lives in backend/agent/agents/
as Agent subclasses (see that package's __init__.py docstring for the
architecture). This file's job has shrunk to exactly what a "graph" file
should do: build the LangGraph StateGraph and wire edges. It no longer
contains any agent business logic itself.

Pipeline:
  planner → search → reflect → rag → synthesize → validate → fact_verify
                        ^__________|  (loop back to search while insufficient,
                                       capped at max_rounds)

Phase 3 Milestone 3 adds `fact_verify` as the new final node — this is
the first topology change since Milestone 1 (everything before this was
a pure refactor that kept the graph shape identical). FactVerificationAgent
runs after validate/CitationAgent, on the already-validated report, and
never blocks or alters it on failure (see fact_verification.py's module
docstring for the full fallback contract).

`reflect` stays a plain function here, not an Agent subclass, and is
deliberately NOT wrapped with tracker/tracer instrumentation — that
remains an intentional scope boundary (no ReflectAgent yet).
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from .state import AgentState, ReflectionDecision
from .reflection import smart_reflect
from .agents import (
    PlannerAgent, SupervisorAgent, RAGAgent, ReportGeneratorAgent,
    CitationAgent, FactVerificationAgent,
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

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", PlannerAgent())
    g.add_node("search", SupervisorAgent())
    g.add_node("reflect", reflect_node)
    g.add_node("rag", RAGAgent())
    g.add_node("synthesize", ReportGeneratorAgent())
    g.add_node("validate", CitationAgent())
    g.add_node("fact_verify", FactVerificationAgent())

    g.add_edge(START, "planner")
    g.add_edge("planner", "search")
    g.add_edge("search", "reflect")
    g.add_conditional_edges(
        "reflect", route_after_reflect,
        {"search": "search", "rag": "rag"},
    )
    g.add_edge("rag", "synthesize")
    g.add_edge("synthesize", "validate")
    g.add_edge("validate", "fact_verify")
    g.add_edge("fact_verify", END)
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
        citation_verification=[],
        citation_confidence=None,
    )

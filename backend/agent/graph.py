"""
Research agent graph — production version with academic research optimization.

Phase 3 Milestone 1: node business logic now lives in backend/agent/agents/
as Agent subclasses (see that package's __init__.py docstring for the
architecture). This file's job has shrunk to exactly what a "graph" file
should do: build the LangGraph StateGraph and wire edges. It no longer
contains any agent business logic itself.

Pipeline:
  planner → search → reflect → rag → synthesize → validate → fact_verify → risk_analyze → revise
                        ^__________|  (loop back to search while insufficient,
                                       capped at max_rounds)

Phase 3 Milestone 3 added `fact_verify`; Milestone 4 added `risk_analyze`.
Phase 5 adds `revise` as the new final node — a deterministic (no LLM)
grounding pass that removes/annotates unsupported claims using
fact_verify's own verdicts, and prunes report sections the sources don't
actually support. All three post-synthesis stages run after the report is
already validated, and none of them block or alter it on failure — see
each agent's module docstring for its own fallback contract
(fact_verification.py, risk_analysis.py, revision.py).

`reflect` stays a plain function here, not an Agent subclass, and is
deliberately NOT wrapped with tracker/tracer instrumentation — that
remains an intentional scope boundary (no ReflectAgent yet).
"""

import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph

from .agents import (
    CitationAgent,
    FactVerificationAgent,
    PlannerAgent,
    RAGAgent,
    ReportGeneratorAgent,
    RevisionAgent,
    RiskAnalysisAgent,
    SupervisorAgent,
    _detect_research_intent,  # noqa: F401 -- re-exported: main.py does `from agent.graph import _detect_research_intent`
)
from .reflection import smart_reflect
from .state import AgentState, ReflectionDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reflect — unchanged from before this milestone. Not yet converted to an
# Agent subclass; see module docstring above.
# ---------------------------------------------------------------------------


def reflect_node(state: AgentState) -> dict:
    if state["round"] >= state["max_rounds"]:
        decision = ReflectionDecision(
            sufficient=True, follow_up_queries=[], reasoning="Reached maximum search rounds."
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

    # Bug fix (metrics dashboard always showing 0 tokens): `reflect` can
    # fall through to an LLM judgment call (smart_reflect's heuristics
    # don't always resolve it), but it's the one node NOT wrapped by
    # Agent.__call__ (see module docstring), so it never got the
    # automatic token-estimation hook added there. Recorded manually here
    # for the same reason base.py's own comment gives: an honest estimate
    # beats a hardcoded 0.
    tracker = state.get("tracker")
    if tracker is not None:
        input_text = state["question"] + " ".join(
            str(s.get("snippet", ""))[:500] for s in list(state["sources"].values())[:20]
        )
        tracker.estimate_tokens_from_text(input_text, role="input")
        tracker.estimate_tokens_from_text(decision.reasoning or "", role="output")

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

def _route_after_planner(state) -> str:
    """If planner rejected the query, skip straight to END."""
    plan = state.get("plan")
    report = state.get("report", "")
    if not plan and report:
        return "end"
    return "search"
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", PlannerAgent())
    g.add_node("search", SupervisorAgent())
    g.add_node("reflect", reflect_node)
    g.add_node("rag", RAGAgent())
    g.add_node("synthesize", ReportGeneratorAgent())
    g.add_node("validate", CitationAgent())
    g.add_node("fact_verify", FactVerificationAgent())
    g.add_node("risk_analyze", RiskAnalysisAgent())
    g.add_node("revise", RevisionAgent())

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", _route_after_planner, {"search": "search", "end": END})
    g.add_edge("search", "reflect")
    g.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"search": "search", "rag": "rag"},
    )
    g.add_edge("rag", "synthesize")
    g.add_edge("synthesize", "validate")
    g.add_edge("validate", "fact_verify")
    g.add_edge("fact_verify", "risk_analyze")
    g.add_edge("risk_analyze", "revise")
    g.add_edge("revise", END)
    return g.compile()


def initial_state(
    question: str,
    max_rounds: int = 2,
    memories: Optional[list] = None,
    stream_callback=None,
    tracker=None,
    tracer=None,
) -> AgentState:
    return AgentState(
        question=question,
        plan=[],
        sources={},
        round=0,
        max_rounds=max_rounds,
        reflection=None,
        report="",
        citations_used=[],
        log=[],
        retrieved_chunks=[],
        rag_session_id=None,
        memories=memories or [],
        stream_callback=stream_callback,
        tracker=tracker,
        tracer=tracer,
        citation_verification=[],
        citation_confidence=None,
        risk_score=None,
        risk_level=None,
        identified_risks=[],
        evidence_gaps=[],
        conflicting_claims=[],
        recommended_follow_up_questions=[],
        report_type=None,
        claims_removed=[],
        claims_rewritten=[],
        unsupported_claims=[],
        final_grounding_score=None,
    )

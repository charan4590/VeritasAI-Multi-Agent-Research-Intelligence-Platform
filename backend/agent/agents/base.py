"""
Phase 3 Milestone 1: Agent interface.
=======================================
Minimal common interface for every agent in the pipeline. Deliberately
thin — LangGraph already handles orchestration (StateGraph, conditional
edges, streaming); this interface exists to give each pipeline step an
explicit identity (a `name`), independent testability (call `.run(state)`
directly with a hand-built state, no graph required), and automatic
metrics/tracing (see __call__ below) without changing how LangGraph
invokes them or what it does.

Design constraint: build_graph() must be able to register `SomeAgent()`
directly as a LangGraph node exactly like it registers a bare function
today — `g.add_node("planner", PlannerAgent())`. LangGraph calls a node
as `node(state) -> dict`, so agents are made callable via __call__.

Instrumentation: __call__ is a template method that wraps every agent's
run() in the *existing* RunTracker.node() (observability.py) and
Tracer.span() (tracing.py) context managers. Both of these already
existed before this milestone and were already created once per research
run in main.py — but neither was ever actually wrapped around a node's
execution (see the Phase 1 architecture review), so /api/metrics/nodes
and /api/traces/{id} always returned empty data. Converting nodes into
Agent subclasses is what finally gives each node's execution a place to
attach that instrumentation, with zero code duplicated per agent.

tracker/tracer are read from AgentState (optional fields added in
state.py, following the exact same pattern Milestone 1 used for
stream_callback) rather than passed as constructor args, because agents
are constructed once when build_graph() runs but a tracker/tracer exists
per-request — reading them from state at call time is what makes that
work without rebuilding the graph on every request.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Dict

from ..state import AgentState


class Agent(ABC):
    """Base class for every pipeline agent."""

    name: str = "agent"

    @abstractmethod
    def run(self, state: AgentState) -> dict:
        """
        Execute this agent's work. Same contract a LangGraph node function
        already has: takes the full state, returns a partial state-update
        dict to merge in. Must not mutate `state` itself.
        """
        raise NotImplementedError

    def trace_inputs(self, state: AgentState) -> Dict[str, Any]:
        """
        Optional: subclasses override this to attach small, human-useful
        context to their trace span (e.g. round number, source count).
        Kept separate from run() on purpose — a bug in trace metadata can
        never affect actual pipeline logic, since this is only ever read
        for observability, never for control flow.
        """
        return {}

    def __call__(self, state: AgentState) -> dict:
        """
        What LangGraph actually invokes. Adds tracking/tracing around
        run() uniformly for every agent — individual agents never write
        any instrumentation code themselves.
        """
        tracker = state.get("tracker")
        tracer = state.get("tracer")

        node_cm = tracker.node(self.name) if tracker is not None else nullcontext()
        span_cm = (
            tracer.span(self.name, inputs=self.trace_inputs(state))
            if tracer is not None else nullcontext()
        )

        with node_cm:
            with span_cm as span:
                result = self.run(state)
                if span is not None:
                    # Explicit span.end(outputs=...) — this is the
                    # documented usage pattern in tracing.py's own
                    # docstring (the context manager only auto-ends with
                    # empty outputs if the caller doesn't do this).
                    span.end(outputs=_summarize_output(result), status="ok")
                return result


def _summarize_output(result: dict) -> Dict[str, Any]:
    """
    Small, generic summary of a node's return dict for the trace span —
    deliberately avoids dumping full reports/source dicts into the trace
    (those are large and already available via the normal API responses;
    the trace just needs enough to answer "what did this step produce").
    """
    summary: Dict[str, Any] = {}
    if "sources" in result:
        summary["source_count"] = len(result["sources"])
    if "plan" in result:
        summary["query_count"] = len(result["plan"])
    if "report" in result:
        summary["report_chars"] = len(result["report"])
    if "citations_used" in result:
        summary["citations_used"] = len(result["citations_used"])
    if "retrieved_chunks" in result:
        summary["chunks_retrieved"] = len(result["retrieved_chunks"])
    if "log" in result:
        summary["log_entries_added"] = len(result["log"])
    return summary

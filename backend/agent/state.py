from typing import TypedDict, Optional, List, Dict, Any


class StreamAborted(Exception):
    """
    Milestone 1: Real Token Streaming.

    Raised by a caller-supplied stream_callback (see main.py) when the SSE
    client has disconnected mid-synthesis. This is intentionally a distinct
    exception type from a generic streaming failure: synthesize_node treats
    a generic failure as "retry with a blocking llm.invoke() call", but an
    abort means the client is gone and no further LLM spend is justified —
    the run should stop immediately instead of falling back.
    """
    pass


class Source(TypedDict):
    id: int
    url: str
    title: str
    snippet: str
    # Phase 3 Milestone 2: "web" (default, set by SupervisorAgent) or
    # "pdf" (uploaded document, via PDFAgent) — lets CitationAgent and
    # ReportGeneratorAgent clearly label uploaded-document sources
    # separately from web sources. Read defensively via .get(..., "web")
    # everywhere, so this is backward compatible with any Source dict
    # that predates this field.
    source_type: str


class SearchPlan(TypedDict):
    queries: List[str]


class ReflectionDecision:
    """Structured reflection decision (plain class for Python 3.9 compat)."""
    def __init__(self, sufficient: bool, follow_up_queries: List[str], reasoning: str):
        self.sufficient = sufficient
        self.follow_up_queries = follow_up_queries
        self.reasoning = reasoning


class AgentState(TypedDict):
    # Core
    question: str
    plan: List[str]
    sources: Dict[int, Source]
    round: int
    max_rounds: int
    reflection: Optional[ReflectionDecision]
    report: str
    citations_used: List[int]
    log: List[str]
    # Phase 1: RAG
    retrieved_chunks: List[Dict[str, Any]]
    rag_session_id: Optional[str]
    # Phase 4: Memory
    memories: List[Dict[str, Any]]
    # #6: optional callback(token: str) for streaming synthesis tokens
    stream_callback: Optional[Any]
    # Phase 3 Milestone 1: optional RunTracker/Tracer instances, read by
    # Agent.__call__ (agents/base.py) to record per-agent metrics/tracing.
    # Same pattern as stream_callback above — optional, set once per
    # request in main.py, None in any test that builds state directly.
    tracker: Optional[Any]
    tracer: Optional[Any]
    # Phase 3 Milestone 3: Fact Verification
    # citation_verification: one record per (sentence, citation_id) claim
    # checked — [] if verification hasn't run yet, found nothing to check,
    # or failed and fell back gracefully (see FactVerificationAgent).
    # citation_confidence: aggregate 0-100 score, or None if unavailable
    # for any of the same reasons.
    citation_verification: List[Dict[str, Any]]
    citation_confidence: Optional[int]

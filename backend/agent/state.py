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

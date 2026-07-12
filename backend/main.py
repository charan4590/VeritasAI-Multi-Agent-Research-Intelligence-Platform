"""
Production async FastAPI backend — Python 3.9 compatible.
"""

import asyncio
import json
import logging
import os
import platform
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Phase 4: Structured logging.
# LOG_FORMAT=json emits one JSON object per line (log aggregator friendly —
# CloudWatch, Datadog, Loki, etc. all parse this natively without a custom
# grok pattern). Default stays human-readable text for local development,
# where a log aggregator isn't in the loop and plain text is easier to
# scan. Nothing about *what* gets logged changes, only how it's formatted.
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_log_handler = logging.StreamHandler()
if os.environ.get("LOG_FORMAT", "text").lower() == "json":
    _log_handler.setFormatter(_JSONFormatter())
else:
    _log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_log_handler], force=True)
logger = logging.getLogger(__name__)

from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import (
    StreamAborted,
    build_graph,
    compute_confidence,
    generate_follow_ups,
    initial_state,
    run_debate,
    score_label,
    score_url,
)
from agent.benchmark import (
    get_benchmark_questions,
    get_benchmark_results,
    run_basic_rag,
    run_direct_llm,
    save_benchmark_result,
)
from agent.credibility import compute_research_confidence
from agent.eval_regression import check_regression, get_current_baseline, save_as_baseline
from agent.evaluator import evaluate_report
from agent.graph import _detect_research_intent
from agent.guardrails import (
    ConcurrencyGuard,
    ConcurrencyLimitExceeded,
    RateLimitExceeded,
    check_rate_limit,
    get_guardrail_status,
)
from agent.memory import add_conversation_turn, delete_memory, retrieve_memories, store_memory
from agent.observability import (
    RunTracker,
    get_aggregate_stats,
    get_node_breakdown,
    get_run_metrics,
    init_observability_tables,
)
from agent.pdf_ingestion import delete_doc, ingest_pdf, list_ingested_docs
from agent.tracing import Tracer, get_recent_traces, get_trace
from db import delete_session, get_history, get_session, init_db, save_session

# ---------------------------------------------------------------------------
# Phase 4: version + response models
# ---------------------------------------------------------------------------
# Single source of truth for the version string — /api/health and
# /api/version both read this instead of each hard-coding their own
# (previously /api/health hard-coded "2.0.0" independently; that's the
# kind of drift a single constant exists to prevent).
APP_VERSION = "3.6.2"  # Bugfix: report table/list styling, Limitations depth


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=[APP_VERSION])


class VersionResponse(BaseModel):
    version: str = Field(..., examples=[APP_VERSION])
    git_commit: str = Field(
        ...,
        description="Short git SHA baked in at Docker build time via --build-arg GIT_COMMIT, or 'unknown' outside Docker.",
        examples=["a1b2c3d"],
    )
    python_version: str = Field(..., examples=[platform.python_version()])
    graph_nodes: List[str] = Field(
        ..., description="LangGraph node names in the compiled research pipeline, in registration order."
    )


class LoginRequest(BaseModel):
    password: Optional[str] = Field(None, description="Required only when AUTH_ENABLED=true.")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

init_db()
init_observability_tables()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup already happened above (module-level init_db()/build_graph()
    # calls) — this exists mainly to document and log shutdown behavior.
    #
    # Graceful shutdown: uvicorn (and gunicorn's uvicorn worker class)
    # already stop accepting new connections and drain in-flight HTTP
    # requests before the process exits on SIGTERM — that's uvicorn's
    # default behavior, nothing here needs to implement it. What *is*
    # worth knowing operationally:
    #   - SQLite connections are opened per-request via `with get_conn()`
    #     (db.py) and closed immediately after, never held open across
    #     the process lifetime — there's no connection pool to drain.
    #   - An in-flight research run's background thread (run_graph(),
    #     see _run_research_stream in this file) is NOT forcibly killed
    #     on shutdown; if the process exits mid-run, that thread simply
    #     stops existing along with the process. The SSE client sees a
    #     dropped connection, same as any other network interruption
    #     Milestone 1's StreamAborted handling already covers gracefully.
    #   - Set `--timeout-graceful-shutdown <seconds>` on uvicorn (or
    #     equivalent gunicorn setting) in production to bound how long
    #     shutdown waits for in-flight requests before forcing exit —
    #     see DEPLOY.md for recommended values per platform.
    yield
    logger.info("Shutting down — in-flight requests are being drained by uvicorn before exit.")


app = FastAPI(
    title="Research Agent — Enterprise AI Research Intelligence Platform",
    description=(
        "A supervisor-orchestrated multi-agent research pipeline: parallel "
        "web/academic/PDF retrieval, RAG, streaming synthesis, citation "
        "validation, claim-level fact verification, and heuristic risk "
        "analysis. See /docs for interactive API exploration, or "
        "SYSTEM_DESIGN.md in the repo for the full architecture."
    ),
    version=APP_VERSION,
    contact={"name": "Project README", "url": "https://github.com/"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Phase 4: request ID + timing middleware
# ---------------------------------------------------------------------------
# Every response gets an X-Request-ID (client-supplied if present, so a
# frontend or load balancer can propagate its own trace id end-to-end;
# generated otherwise) and an X-Response-Time-Ms header, and every request
# gets exactly one summary log line. This is intentionally separate from
# the per-agent RunTracker/Tracer instrumentation (Phase 3) — that measures
# pipeline internals; this measures the HTTP layer around it, including
# routes RunTracker never sees (history, metrics, health, static files).


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
        extra={"request_id": request_id},
    )
    return response


_graph = build_graph()
logger.info("Graph compiled. Nodes: %s", list(_graph.nodes.keys()))

RESEARCH_TIMEOUT = int(os.environ.get("RESEARCH_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Auth — simplified, no JWT required for local use
# ---------------------------------------------------------------------------


@app.get("/api/auth/check", tags=["meta"])
async def auth_check():
    return {"requires_password": False}


@app.post("/api/auth/login", tags=["meta"])
async def login(body: LoginRequest):
    return {"ok": True}


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Research stream
# ---------------------------------------------------------------------------


async def _run_research_stream(question: str, max_rounds: int) -> AsyncIterator[str]:
    tracker = RunTracker(question=question, mode="research")
    tracker.start()
    tracer = Tracer(question=question)  # #2: structured span tree for this run
    run_start = time.time()
    queue: asyncio.Queue = asyncio.Queue()

    # Milestone 1: Real Token Streaming.
    # disconnect_event is shared between this async generator (event loop
    # thread) and run_graph() (a worker thread from run_in_executor). It's
    # a plain threading.Event specifically because it needs to be safely
    # set/read across that thread boundary. When the SSE client goes away,
    # the `finally` block around the consumer loop below sets this; the
    # stream_token_callback passed into the graph checks it on every token
    # and raises StreamAborted to stop mid-synthesis instead of burning
    # further LLM output (and the rest of run_graph's post-processing —
    # eval, follow-ups, session save) on a request nobody will receive.
    disconnect_event = threading.Event()

    def stream_token_callback(token: str):
        if disconnect_event.is_set():
            raise StreamAborted("Client disconnected during synthesis")
        queue.put_nowait(sse({"type": "token", "node": "synthesize", "content": token}))

    def run_graph():
        # #8: concurrency guard — caps simultaneous research runs.
        # Manual enter/exit (not 'with') to avoid reindenting this large function.
        guard = ConcurrencyGuard()
        try:
            guard.__enter__()
        except ConcurrencyLimitExceeded as exc:
            queue.put_nowait(sse({"type": "error", "message": str(exc)}))
            queue.put_nowait(None)
            return

        try:
            memories = retrieve_memories(question)
            if memories:
                queue.put_nowait(
                    sse(
                        {
                            "type": "progress",
                            "node": "memory",
                            "message": f"Found {len(memories)} relevant past session(s)",
                        }
                    )
                )

            state = initial_state(
                question,
                max_rounds=max_rounds,
                memories=memories,
                stream_callback=stream_token_callback,
                # Phase 3 Milestone 1: this is what actually activates
                # RunTracker.node() / Tracer.span() per agent — both
                # objects already existed and were already created once
                # per run above, but neither was ever wired to a node's
                # execution before now (see Phase 1 architecture review;
                # /api/metrics/nodes and /api/traces/{id} always returned
                # empty data as a result). Agent.__call__ (agents/base.py)
                # reads these two fields from state and wraps every
                # agent's run() in both context managers.
                tracker=tracker,
                tracer=tracer,
            )
            last_log = 0
            latest_sources = {}
            final_state = None

            for update in _graph.stream(state, stream_mode="updates"):
                for node, partial in update.items():
                    log = partial.get("log")
                    if log:
                        for entry in log[last_log:]:
                            queue.put_nowait(sse({"type": "progress", "node": node, "message": entry}))
                        last_log = len(log)
                    if "sources" in partial:
                        latest_sources = partial["sources"]
                    # Phase 5: revise joined risk_analyze/fact_verify/
                    # validate as a terminal node after validate. Same
                    # merge rationale — and importantly, revise's
                    # updated `report`/`citations_used` correctly
                    # overwrite validate's pre-revision values here,
                    # since dict-merge order means the last node's
                    # partial wins for any field it also returns.
                    if node in ("validate", "fact_verify", "risk_analyze", "revise"):
                        final_state = {**(final_state or {}), **partial}

            if not final_state:
                queue.put_nowait(sse({"type": "error", "message": "Agent finished without a report."}))
                return

            used = final_state.get("citations_used", [])
            rag_chunks = len(state.get("retrieved_chunks", []))
            tracker.rag_enabled = rag_chunks > 0
            tracker.chunks_retrieved = rag_chunks

            sources_payload = {
                str(i): {
                    "title": latest_sources[i]["title"],
                    "url": latest_sources[i]["url"],
                    "snippet": latest_sources[i]["snippet"],
                    "credibility": score_url(latest_sources[i]["url"]),
                    "credibility_label": score_label(score_url(latest_sources[i]["url"])),
                    # Phase 3 Milestone 2: additive field ("web" for any
                    # source predating this milestone or any non-PDF
                    # source) — existing consumers of this payload ignore
                    # unknown keys, so this doesn't change response shape
                    # for anyone not looking for it.
                    "source_type": latest_sources[i].get("source_type", "web"),
                }
                for i in used
                if i in latest_sources
            }

            # Phase 6 (final polish): "Sources Retrieved During Research" —
            # every source the search stage actually gathered, not just
            # the ones that survived into a citation. sources_payload
            # above is deliberately left untouched (cited-only, same
            # shape as always) for backward compatibility; this is a new,
            # separate, additive field. cited=False here commonly means
            # "retrieved but not needed for this answer" rather than
            # "bad source" — reflection/RAG routinely gather more evidence
            # than synthesis ends up citing.
            all_sources_payload = {
                str(i): {
                    "title": s["title"],
                    "url": s["url"],
                    "credibility": score_url(s["url"]),
                    "credibility_label": score_label(score_url(s["url"])),
                    "source_type": s.get("source_type", "web"),
                    "cited": i in used,
                }
                for i, s in latest_sources.items()
            }

            _intent = _detect_research_intent(question)
            if _intent == "academic":
                confidence = compute_research_confidence(latest_sources, used, final_state.get("report", ""))
            else:
                confidence = compute_confidence(latest_sources, used)

            queue.put_nowait(
                sse({"type": "progress", "node": "eval", "message": "Evaluating report quality..."})
            )
            eval_scores = evaluate_report(
                question=question,
                report=final_state.get("report", ""),
                sources=latest_sources,
                citations_used=used,
            )
            queue.put_nowait(
                sse(
                    {
                        "type": "progress",
                        "node": "eval",
                        "message": f"Quality: {eval_scores['overall_score']}/100 — Grade {eval_scores['grade']}",
                    }
                )
            )

            queue.put_nowait(
                sse({"type": "progress", "node": "followup", "message": "Generating follow-up questions..."})
            )
            follow_ups = generate_follow_ups(question, final_state.get("report", ""))

            latency_ms = int((time.time() - run_start) * 1000)
            session_id = save_session(
                question=question,
                report=final_state.get("report", ""),
                sources=sources_payload,
                confidence=confidence,
                mode="research",
                follow_ups=follow_ups,
                eval_scores=eval_scores,
                rag_chunks_used=rag_chunks,
                latency_ms=latency_ms,
                # Phase 3 Milestone 3
                citation_verification=final_state.get("citation_verification", []),
                citation_confidence=final_state.get("citation_confidence"),
                # Phase 3 Milestone 4
                risk_score=final_state.get("risk_score"),
                risk_level=final_state.get("risk_level"),
                identified_risks=final_state.get("identified_risks", []),
                evidence_gaps=final_state.get("evidence_gaps", []),
                conflicting_claims=final_state.get("conflicting_claims", []),
                recommended_follow_up_questions=final_state.get("recommended_follow_up_questions", []),
                # Phase 5
                report_type=final_state.get("report_type"),
                claims_removed=final_state.get("claims_removed", []),
                claims_rewritten=final_state.get("claims_rewritten", []),
                unsupported_claims=final_state.get("unsupported_claims", []),
                final_grounding_score=final_state.get("final_grounding_score"),
                # Phase 6 (final polish)
                all_sources=all_sources_payload,
            )

            store_memory(
                session_id=session_id,
                question=question,
                report_summary=final_state.get("report", ""),
                eval_scores=eval_scores,
            )
            add_conversation_turn(question, final_state.get("report", ""))
            tracker.finish(session_id=session_id, status="done")

            queue.put_nowait(
                sse(
                    {
                        "type": "done",
                        "report": final_state.get("report", ""),
                        "citations_used": used,
                        "sources": sources_payload,
                        "confidence": confidence,
                        "follow_ups": follow_ups,
                        "eval_scores": eval_scores,
                        "rag_chunks": rag_chunks,
                        "latency_ms": latency_ms,
                        "session_id": session_id,
                        # Phase 3 Milestone 3: [] / None when verification found
                        # nothing to check or fell back gracefully — always
                        # present so the frontend never needs an `in` check.
                        "citation_verification": final_state.get("citation_verification", []),
                        "citation_confidence": final_state.get("citation_confidence"),
                        # Phase 3 Milestone 4: same "always present" contract.
                        "risk_score": final_state.get("risk_score"),
                        "risk_level": final_state.get("risk_level"),
                        "identified_risks": final_state.get("identified_risks", []),
                        "evidence_gaps": final_state.get("evidence_gaps", []),
                        "conflicting_claims": final_state.get("conflicting_claims", []),
                        "recommended_follow_up_questions": final_state.get(
                            "recommended_follow_up_questions", []
                        ),
                        # Phase 5: grounding summary, same "always present" contract.
                        "report_type": final_state.get("report_type"),
                        "claims_removed": final_state.get("claims_removed", []),
                        "claims_rewritten": final_state.get("claims_rewritten", []),
                        "unsupported_claims": final_state.get("unsupported_claims", []),
                        "final_grounding_score": final_state.get("final_grounding_score"),
                        # Phase 6 (final polish): every retrieved source
                        # with cited/not-cited status, for the frontend's
                        # "Sources Retrieved During Research" section.
                        "all_sources": all_sources_payload,
                    }
                )
            )

        except StreamAborted:
            # Milestone 1: client disconnected mid-synthesis. Nothing left
            # to notify (the queue has no listener anymore) — just record
            # it and stop. Deliberately skips eval/follow-ups/save_session:
            # those exist to make a delivered report useful, and there's
            # no report being delivered here.
            logger.info(f"Research stream aborted (client disconnected): {question!r}")
            tracker.finish(status="aborted", error="client disconnected")
        except Exception as exc:
            logger.exception("Research stream failed")
            msg = str(exc)
            if "Connection refused" in msg or "ConnectError" in msg:
                msg = "Cannot connect to Ollama. Make sure the Ollama app is running (check your Mac menu bar for the llama icon)."
            tracker.finish(status="error", error=msg)
            queue.put_nowait(sse({"type": "error", "message": msg}))
        finally:
            guard.__exit__(None, None, None)
            tracer.finish(status="done")
            queue.put_nowait(None)

    asyncio.get_event_loop().run_in_executor(None, run_graph)

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        # Milestone 1: fires on normal completion AND on client disconnect
        # (Starlette closes this async generator, which raises at the
        # `yield` above). Setting it after normal completion is harmless —
        # run_graph has already finished by then either way.
        disconnect_event.set()


@app.get(
    "/api/research/stream",
    tags=["research"],
    summary="Run the multi-agent research pipeline (Server-Sent Events)",
    description=(
        "Streams a live research run as Server-Sent Events: `progress` "
        "messages per pipeline stage, `token` events as the report is "
        "written (Milestone 1), and a final `done` event containing the "
        "validated report plus citation verification (Milestone 3) and "
        "risk analysis (Milestone 4). Not a JSON response — consume with "
        "an EventSource client, not a plain HTTP client expecting one body."
    ),
)
async def research_stream(
    question: str = Query(
        ...,
        min_length=3,
        description="The research question to investigate.",
        examples=["What are the latest advances in hybrid CNN-LSTM architectures for medical imaging?"],
    ),
    max_rounds: int = Query(
        2,
        ge=1,
        le=4,
        description="Max search-and-reflect rounds before forcing synthesis, regardless of reflection's sufficiency judgment.",
    ),
):
    # #8: rate limit by a simple global key (single-instance deployment).
    # For multi-user production, swap "global" for request.client.host.
    try:
        check_rate_limit("global")
    except RateLimitExceeded as exc:
        # Python deletes `exc` automatically at the end of this except
        # block (it holds a traceback reference, so CPython clears it to
        # avoid a reference cycle) — but error_stream() below is a
        # generator that isn't actually iterated until StreamingResponse
        # consumes it, which happens asynchronously *after* this function
        # has already returned and `exc` is long gone. Capturing the
        # message as a plain string here, before that deletion, is what
        # makes this actually work instead of raising NameError the first
        # time someone hits the rate limit (caught by adopting ruff in
        # Phase 4 — see CI notes).
        message = str(exc)

        async def error_stream():
            yield sse({"type": "error", "message": message})

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    return StreamingResponse(
        _run_research_stream(question, max_rounds),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Debate stream
# ---------------------------------------------------------------------------


@app.get(
    "/api/debate/stream",
    tags=["debate"],
    summary="Run a two-sided FOR/AGAINST debate pipeline (Server-Sent Events)",
    description=(
        "A separate, independent pipeline from /api/research/stream — see "
        "SYSTEM_DESIGN.md for why debate.py doesn't reuse the Phase 3 "
        "agent classes. Streams progress events, then a `done` event with "
        "both sides' arguments and citations."
    ),
)
async def debate_stream(
    question: str = Query(
        ..., min_length=3, examples=["Should social media platforms be regulated like utilities?"]
    ),
):
    async def generate():
        run_start = time.time()
        log_messages = []

        def log_cb(msg):
            log_messages.append(msg)

        try:
            result = await asyncio.to_thread(run_debate, question, log_cb)
            for msg in log_messages:
                yield sse({"type": "progress", "node": "debate", "message": msg})

            sources_raw = result.get("sources", {})
            int_sources = {int(k): v for k, v in sources_raw.items()}
            used = result.get("citations_used", [])
            eval_scores = await asyncio.to_thread(
                evaluate_report, question, result.get("report", ""), int_sources, used
            )

            for k, s in sources_raw.items():
                s["credibility"] = score_url(s.get("url", ""))
                s["credibility_label"] = score_label(s["credibility"])

            confidence = compute_confidence(int_sources, used)
            follow_ups = await asyncio.to_thread(generate_follow_ups, question, result.get("report", ""))
            latency_ms = int((time.time() - run_start) * 1000)

            session_id = save_session(
                question=question,
                report=result.get("report", ""),
                sources=sources_raw,
                confidence=confidence,
                mode="debate",
                follow_ups=follow_ups,
                eval_scores=eval_scores,
                latency_ms=latency_ms,
            )
            store_memory(
                session_id=session_id,
                question=question,
                report_summary=result.get("report", ""),
                eval_scores=eval_scores,
            )

            yield sse(
                {
                    "type": "done",
                    "report": result.get("report", ""),
                    "sources": sources_raw,
                    "citations_used": used,
                    "confidence": confidence,
                    "follow_ups": follow_ups,
                    "eval_scores": eval_scores,
                    "latency_ms": latency_ms,
                    "session_id": session_id,
                }
            )
        except Exception as exc:
            logger.exception("Debate stream failed")
            yield sse({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------


@app.post("/api/docs/upload", tags=["documents"])
async def upload_pdf(file: UploadFile = File(default=...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    result = await asyncio.to_thread(ingest_pdf, content, file.filename)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Ingestion failed"))
    return result


@app.get("/api/docs", tags=["documents"])
async def list_docs():
    return await asyncio.to_thread(list_ingested_docs)


@app.delete("/api/docs/{doc_id}", tags=["documents"])
async def delete_document(doc_id: str):
    await asyncio.to_thread(delete_doc, doc_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@app.get("/api/history", tags=["history"])
async def list_history():
    rows = await asyncio.to_thread(get_history)
    return [
        {
            "id": r["id"],
            "question": r["question"],
            "confidence": r["confidence"],
            "mode": r["mode"],
            "eval_overall": r.get("eval_overall"),
            "eval_grade": r.get("eval_grade"),
            "rag_chunks_used": r.get("rag_chunks_used", 0),
            "latency_ms": r.get("latency_ms", 0),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@app.get("/api/history/{session_id}", tags=["history"])
async def get_one(session_id: int):
    row = await asyncio.to_thread(get_session, session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row


@app.delete("/api/history/{session_id}", tags=["history"])
async def delete_one(session_id: int):
    await asyncio.to_thread(delete_session, session_id)
    await asyncio.to_thread(delete_memory, session_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@app.get("/api/metrics/runs", tags=["metrics"])
async def metrics_runs():
    return await asyncio.to_thread(get_run_metrics, 50)


@app.get("/api/metrics/nodes/{run_id}", tags=["metrics"])
async def metrics_nodes(run_id: int):
    return await asyncio.to_thread(get_node_breakdown, run_id)


@app.get("/api/metrics/summary", tags=["metrics"])
async def metrics_summary():
    return await asyncio.to_thread(get_aggregate_stats)


# ---------------------------------------------------------------------------
# #2: Tracing endpoints
# ---------------------------------------------------------------------------


@app.get("/api/traces", tags=["metrics"])
async def list_traces():
    return await asyncio.to_thread(get_recent_traces, 20)


@app.get("/api/traces/{trace_id}", tags=["metrics"])
async def get_trace_detail(trace_id: str):
    trace = await asyncio.to_thread(get_trace, trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


# ---------------------------------------------------------------------------
# #8: Guardrail status endpoint
# ---------------------------------------------------------------------------


@app.get("/api/guardrails/status", tags=["meta"])
async def guardrails_status():
    return get_guardrail_status()


# ---------------------------------------------------------------------------
# #7: Eval regression endpoints
# ---------------------------------------------------------------------------


@app.post("/api/eval/baseline", tags=["evaluation"])
async def set_baseline(body: dict):
    results = body.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="results required")
    version = await asyncio.to_thread(save_as_baseline, results, body.get("version_label"))
    return {"version": version, "saved": len(results)}


@app.post("/api/eval/check-regression", tags=["evaluation"])
async def regression_check(body: dict):
    results = body.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="results required")
    return await asyncio.to_thread(check_regression, results)


@app.get("/api/eval/baseline", tags=["evaluation"])
async def get_baseline():
    return await asyncio.to_thread(get_current_baseline)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@app.get("/api/benchmark/questions", tags=["benchmark"])
async def benchmark_questions(limit: int = Query(10, ge=1, le=30)):
    return get_benchmark_questions(limit)


@app.get("/api/benchmark/results", tags=["benchmark"])
async def benchmark_results():
    return await asyncio.to_thread(get_benchmark_results)


@app.post("/api/benchmark/run-baselines", tags=["benchmark"])
async def run_baselines(body: dict):
    question = body.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    direct = await asyncio.to_thread(run_direct_llm, question)
    save_benchmark_result(question, direct)
    basic = await asyncio.to_thread(run_basic_rag, question)
    save_benchmark_result(question, basic)
    return {"direct_llm": direct, "basic_rag": basic}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Health & version
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
async def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/version", response_model=VersionResponse, tags=["meta"])
async def get_version():
    """
    Returns the running application version, the git commit it was built
    from (Docker builds only — see Dockerfile's GIT_COMMIT build arg), the
    Python interpreter version, and the compiled LangGraph's node list —
    a quick way to confirm which pipeline stages (e.g. fact_verify,
    risk_analyze) are actually active in a given deployment.
    """
    return {
        "version": APP_VERSION,
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "python_version": platform.python_version(),
        "graph_nodes": list(_graph.nodes.keys()),
    }


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

_frontend = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")

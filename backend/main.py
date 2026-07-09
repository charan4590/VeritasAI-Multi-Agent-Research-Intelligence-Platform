"""
Production async FastAPI backend — Python 3.9 compatible.
"""

import os
import json
import time
import asyncio
import logging
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Query, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, AsyncIterator

from agent import (
    build_graph, initial_state,
    run_debate, generate_follow_ups,
    compute_confidence, score_url, score_label,
    StreamAborted,
)
from agent.credibility import compute_research_confidence
from agent.graph import _detect_research_intent
from agent.evaluator import evaluate_report
from agent.memory import retrieve_memories, store_memory, delete_memory, add_conversation_turn
from agent.observability import RunTracker, init_observability_tables, get_run_metrics, get_node_breakdown, get_aggregate_stats
from agent.pdf_ingestion import ingest_pdf, list_ingested_docs, delete_doc
from agent.benchmark import run_direct_llm, run_basic_rag, save_benchmark_result, get_benchmark_results, get_benchmark_questions
from agent.tracing import Tracer, get_trace, get_recent_traces
from agent.guardrails import check_rate_limit, ConcurrencyGuard, RateLimitExceeded, ConcurrencyLimitExceeded, get_guardrail_status
from agent.eval_regression import save_as_baseline, check_regression, get_current_baseline
from db import init_db, save_session, get_history, get_session, delete_session

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

init_db()
init_observability_tables()

app = FastAPI(title="Research Agent")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

_graph = build_graph()
logger.info("Graph compiled. Nodes: %s", list(_graph.nodes.keys()))

RESEARCH_TIMEOUT = int(os.environ.get("RESEARCH_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Auth — simplified, no JWT required for local use
# ---------------------------------------------------------------------------

@app.get("/api/auth/check")
async def auth_check():
    return {"requires_password": False}


@app.post("/api/auth/login")
async def login(body: dict):
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
            mem_span = tracer.root  # lightweight: track memory step without deep nesting
            memories = retrieve_memories(question)
            if memories:
                queue.put_nowait(sse({"type": "progress", "node": "memory",
                    "message": f"Found {len(memories)} relevant past session(s)"}))

            state = initial_state(
                question, max_rounds=max_rounds, memories=memories,
                stream_callback=stream_token_callback,
            )
            last_log = 0
            latest_sources = {}
            final_state = None

            for update in _graph.stream(state, stream_mode="updates"):
                for node, partial in update.items():
                    log = partial.get("log")
                    if log:
                        for entry in log[last_log:]:
                            queue.put_nowait(sse({"type": "progress", "node": node,
                                                  "message": entry}))
                        last_log = len(log)
                    if "sources" in partial:
                        latest_sources = partial["sources"]
                    if node == "validate":
                        final_state = partial

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
                }
                for i in used if i in latest_sources
            }

            _intent = _detect_research_intent(question)
            if _intent == "academic":
                confidence = compute_research_confidence(latest_sources, used, final_state.get("report", ""))
            else:
                confidence = compute_confidence(latest_sources, used)

            queue.put_nowait(sse({"type": "progress", "node": "eval", "message": "Evaluating report quality..."}))
            eval_scores = evaluate_report(
                question=question,
                report=final_state.get("report", ""),
                sources=latest_sources,
                citations_used=used,
            )
            queue.put_nowait(sse({"type": "progress", "node": "eval",
                "message": f"Quality: {eval_scores['overall_score']}/100 — Grade {eval_scores['grade']}"}))

            queue.put_nowait(sse({"type": "progress", "node": "followup", "message": "Generating follow-up questions..."}))
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
            )

            store_memory(session_id=session_id, question=question,
                         report_summary=final_state.get("report", ""), eval_scores=eval_scores)
            add_conversation_turn(question, final_state.get("report", ""))
            tracker.finish(session_id=session_id, status="done")

            queue.put_nowait(sse({
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
            }))

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


@app.get("/api/research/stream")
async def research_stream(
    question: str = Query(..., min_length=3),
    max_rounds: int = Query(2, ge=1, le=4),
):
    # #8: rate limit by a simple global key (single-instance deployment).
    # For multi-user production, swap "global" for request.client.host.
    try:
        check_rate_limit("global")
    except RateLimitExceeded as exc:
        async def error_stream():
            yield sse({"type": "error", "message": str(exc)})
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    return StreamingResponse(
        _run_research_stream(question, max_rounds),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Debate stream
# ---------------------------------------------------------------------------

@app.get("/api/debate/stream")
async def debate_stream(question: str = Query(..., min_length=3)):
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
            eval_scores = await asyncio.to_thread(evaluate_report, question, result.get("report", ""), int_sources, used)

            for k, s in sources_raw.items():
                s["credibility"] = score_url(s.get("url", ""))
                s["credibility_label"] = score_label(s["credibility"])

            confidence = compute_confidence(int_sources, used)
            follow_ups = await asyncio.to_thread(generate_follow_ups, question, result.get("report", ""))
            latency_ms = int((time.time() - run_start) * 1000)

            session_id = save_session(
                question=question, report=result.get("report", ""),
                sources=sources_raw, confidence=confidence, mode="debate",
                follow_ups=follow_ups, eval_scores=eval_scores, latency_ms=latency_ms,
            )
            store_memory(session_id=session_id, question=question,
                         report_summary=result.get("report", ""), eval_scores=eval_scores)

            yield sse({
                "type": "done", "report": result.get("report", ""),
                "sources": sources_raw, "citations_used": used,
                "confidence": confidence, "follow_ups": follow_ups,
                "eval_scores": eval_scores, "latency_ms": latency_ms,
                "session_id": session_id,
            })
        except Exception as exc:
            logger.exception("Debate stream failed")
            yield sse({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

@app.post("/api/docs/upload")
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


@app.get("/api/docs")
async def list_docs():
    return await asyncio.to_thread(list_ingested_docs)


@app.delete("/api/docs/{doc_id}")
async def delete_document(doc_id: str):
    await asyncio.to_thread(delete_doc, doc_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@app.get("/api/history")
async def list_history():
    rows = await asyncio.to_thread(get_history)
    return [{
        "id": r["id"], "question": r["question"],
        "confidence": r["confidence"], "mode": r["mode"],
        "eval_overall": r.get("eval_overall"),
        "eval_grade": r.get("eval_grade"),
        "rag_chunks_used": r.get("rag_chunks_used", 0),
        "latency_ms": r.get("latency_ms", 0),
        "created_at": r["created_at"],
    } for r in rows]


@app.get("/api/history/{session_id}")
async def get_one(session_id: int):
    row = await asyncio.to_thread(get_session, session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row


@app.delete("/api/history/{session_id}")
async def delete_one(session_id: int):
    await asyncio.to_thread(delete_session, session_id)
    await asyncio.to_thread(delete_memory, session_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@app.get("/api/metrics/runs")
async def metrics_runs():
    return await asyncio.to_thread(get_run_metrics, 50)


@app.get("/api/metrics/nodes/{run_id}")
async def metrics_nodes(run_id: int):
    return await asyncio.to_thread(get_node_breakdown, run_id)


@app.get("/api/metrics/summary")
async def metrics_summary():
    return await asyncio.to_thread(get_aggregate_stats)


# ---------------------------------------------------------------------------
# #2: Tracing endpoints
# ---------------------------------------------------------------------------

@app.get("/api/traces")
async def list_traces():
    return await asyncio.to_thread(get_recent_traces, 20)


@app.get("/api/traces/{trace_id}")
async def get_trace_detail(trace_id: str):
    trace = await asyncio.to_thread(get_trace, trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


# ---------------------------------------------------------------------------
# #8: Guardrail status endpoint
# ---------------------------------------------------------------------------

@app.get("/api/guardrails/status")
async def guardrails_status():
    return get_guardrail_status()


# ---------------------------------------------------------------------------
# #7: Eval regression endpoints
# ---------------------------------------------------------------------------

@app.post("/api/eval/baseline")
async def set_baseline(body: dict):
    results = body.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="results required")
    version = await asyncio.to_thread(save_as_baseline, results, body.get("version_label"))
    return {"version": version, "saved": len(results)}


@app.post("/api/eval/check-regression")
async def regression_check(body: dict):
    results = body.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="results required")
    return await asyncio.to_thread(check_regression, results)


@app.get("/api/eval/baseline")
async def get_baseline():
    return await asyncio.to_thread(get_current_baseline)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@app.get("/api/benchmark/questions")
async def benchmark_questions(limit: int = Query(10, ge=1, le=30)):
    return get_benchmark_questions(limit)


@app.get("/api/benchmark/results")
async def benchmark_results():
    return await asyncio.to_thread(get_benchmark_results)


@app.post("/api/benchmark/run-baselines")
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

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

_frontend = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")

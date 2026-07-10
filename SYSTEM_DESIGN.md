# System Design

This document explains how the Research Agent is actually built — not a
marketing overview, but the architecture, the tradeoffs, and why each
piece exists. Written for someone reviewing the codebase or interviewing
about it.

## 1. High-level architecture

```
┌─────────────┐     SSE      ┌──────────────────────────────────────────────┐
│  Frontend    │◄────────────►│                FastAPI (main.py)              │
│ (index.html) │              │  request-ID/timing middleware, 22 REST routes │
└─────────────┘              └───────────────────┬────────────────────────────┘
                                                   │ build_graph()
                                      ┌────────────▼─────────────┐
                                      │   LangGraph StateGraph    │
                                      │        (graph.py)         │
                                      └────────────┬─────────────┘
       ┌─────────┬─────────┬─────────┬─────────────┼─────────────┬──────────────┬──────────────┐
       ▼         ▼         ▼         ▼             ▼             ▼              ▼              ▼
   planner    search    reflect     rag        synthesize     validate     fact_verify    risk_analyze
  (Planner  (Supervisor            (RAG      (ReportGenerator (Citation   (FactVerif-    (RiskAnalysis
   Agent)    Agent)                 Agent)     Agent)          Agent)     icationAgent)   Agent)
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   WebResearch  Academic  PDFAgent
     Agent      SearchAgent
```

Every node except `reflect` is a class implementing a common `Agent`
interface (`agent/agents/base.py`). `reflect` stays a plain function
deliberately — see §3.

## 2. The supervisor-based multi-agent architecture

### Why "supervisor," not just "graph nodes"

LangGraph already provides orchestration (the `StateGraph`, conditional
edges, streaming). What it doesn't provide is a place to put *routing
logic that spans multiple concerns* — deciding whether to search the web
or academic sources, then running shared post-processing on whichever one
answered, is a job that belongs to neither "web search" nor "academic
search" individually. `SupervisorAgent` (`agent/agents/supervisor.py`)
owns exactly that: it's the one node that itself orchestrates other
agents rather than just doing its own work.

```python
class SupervisorAgent(Agent):
    name = "search"                      # still a single LangGraph node

    def run(self, state):
        if intent == "academic":
            batch_results = self.academic_agent.search(queries)
        else:
            batch_results = self.web_agent.search(queries)
        pdf_results = self.pdf_agent.search(queries)     # always, see §5
        # ... shared concurrent-fetch / dedup / credibility-sort pipeline
```

`WebResearchAgent`, `AcademicSearchAgent`, and `PDFAgent` are each
independently testable (`SomeAgent().search(["query"])` needs no graph,
no state, no mocking beyond the one function they call) and single-purpose
— "how do I search X." `SupervisorAgent` is the only place that needs to
know about all three.

### The `Agent` interface

```python
class Agent(ABC):
    name: str
    def run(self, state: AgentState) -> dict: ...      # subclasses implement this
    def __call__(self, state: AgentState) -> dict:      # LangGraph calls this
        with tracker.node(self.name):                    # Milestone 1 instrumentation
            with tracer.span(self.name, ...) as span:
                result = self.run(state)
                span.end(outputs=_summarize_output(result))
                return result
```

`__call__` is a template method: every agent gets `RunTracker`/`Tracer`
instrumentation automatically, with zero instrumentation code written per
agent. This is also why adding a new agent (Milestones 2–4 each added
one) is a small diff — implement `run()`, register the class as a node,
done. Observability, caching integration points, and the calling
convention are already handled by the base class.

## 3. The pipeline, stage by stage

```
planner → search → reflect → rag → synthesize → validate → fact_verify → risk_analyze → revise → END
             ▲__________________|
             (loop back to search while reflection says insufficient, capped at max_rounds)
```

1. **planner** — classifies query intent (academic/technical/general),
   asks the LLM for 3–5 targeted search queries with an intent-specific
   prompt.
2. **search** (SupervisorAgent) — see §2 and §4.
3. **reflect** — a plain function, not an `Agent` subclass (deliberate —
   see below). Runs cheap heuristic checks *before* spending an LLM call:
   source count, domain diversity, subtopic coverage, academic-section
   coverage, contradiction signals. Only falls through to an LLM judgment
   call if every heuristic passes. This ordering is the actual point: a
   query with only 2 sources never needs to ask an LLM "are 2 sources
   enough" — the heuristic already knows.
4. **rag** — indexes this round's sources into a per-session ChromaDB
   collection, retrieves the top-k chunks by semantic similarity to the
   question, re-ranks with a cross-encoder when available.
5. **synthesize** (ReportGeneratorAgent) — writes the report. Streams
   tokens to the browser as they're generated (§6) rather than blocking
   until the full report is done.
6. **validate** (CitationAgent) — pure regex, no LLM. Strips any `[n]`
   citation marker that doesn't map to a real retrieved source. This is
   citation *existence* checking, not truthfulness checking — that's the
   next stage's job.
7. **fact_verify** (FactVerificationAgent) — see §7.
8. **risk_analyze** (RiskAnalysisAgent) — see §8.

**Why `reflect` isn't an `Agent` subclass yet:** it's the one node that
still returns a typed `ReflectionDecision` object consumed by the graph's
conditional-edge routing function, not just a state-update dict like
every other node. Converting it cleanly needs slightly different
handling in the `Agent` base class than the terminal nodes needed, and
it was explicitly scoped out of Phase 3 to keep each milestone's diff
reviewable rather than bundling a routing-logic change in with the
agent-conversion refactor.

## 4. Parallel retrieval

Three independent latency sources get parallelized, all with the same
`ThreadPoolExecutor` pattern (`concurrent.futures`, not `asyncio`, since
these are blocking `requests` calls, not native async I/O):

- **N search queries** (`web_search_batch`/`academic_web_search_batch`,
  `agent/tools.py`) — run concurrently instead of serially. This was the
  single largest latency contributor before optimization (5 queries ×
  ~7s each = 35s+ serial).
- **Full-page content fetches** (`_fetch_full_content_batch`,
  `agent/agents/supervisor.py`) — up to 6 high-credibility sources fetched
  concurrently instead of one blocking `requests.get()` at a time.
  Measured ~4.8x speedup on realistic per-URL latencies.
- **PDF chunk search** (`PDFAgent.search`, `agent/agents/pdf_agent.py`) —
  same pattern, N queries against the local PDF vector index concurrently.

None of these use real thread pools "because async is hard" — they use
thread pools because the underlying work (`requests.get`, embedding calls
via HTTP to Ollama) is blocking I/O, and `ThreadPoolExecutor` is the
correct primitive for blocking I/O parallelism in a process that's
otherwise driven by `asyncio` (FastAPI) one layer up — the actual graph
execution runs in a background thread via `run_in_executor`, so nothing
here blocks the event loop either.

## 5. PDF retrieval — how uploaded documents enter the pipeline

`PDFAgent` wraps `pdf_ingestion.py`'s existing `search_pdfs()` (chunking,
embedding, and ChromaDB storage happen at upload time, in
`ingest_pdf()`), exposing the same per-query batch interface as the two
web search agents. `SupervisorAgent` queries it **unconditionally** every
round, not gated behind "does this session have PDFs" — `search_pdfs()`
already returns `[]` in under a millisecond when no PDF collection
exists yet (confirmed via benchmark: 0.26ms/query worst case), so the
conditional check would be pure overhead for the same outcome.

PDF chunks are folded into the *same* candidate-gathering loop as web
results — same `next_id` counter, same `MAX_SOURCES` cap, same dedup set
— so citation numbering is one continuous sequence regardless of source
type. A `source_type` field ("web"/"pdf") is the only thing that marks
provenance, read defensively (`.get("source_type", "web")`) everywhere,
purely for labeling (the `[PDF]` tag in the Sources footer) — never for
routing or numbering.

**Known limitation, stated plainly:** `credibility.score_url()` has no
`pdf://` scheme handling, so uploaded documents land at the generic
unscored tier (50/"Low") rather than getting credit for being
user-provided. This was a deliberate scope boundary (documented in the
Milestone 2 PR), not an oversight — PDF evidence still surfaces correctly
via RAG's *semantic* relevance ranking, which doesn't depend on
credibility score at all.

## 6. Streaming pipeline

Two independent SSE (Server-Sent Events) channels multiplexed onto one
`/api/research/stream` connection:

1. **Progress events** — one per graph node completion, sourced directly
   from each node's `log` list additions (`{"type": "progress", "node":
   ..., "message": ...}`). This existed before token streaming was added.
2. **Token events** — added in Milestone 1. `ReportGeneratorAgent` calls
   `llm.stream()` instead of `llm.invoke()` when a `stream_callback` is
   present in state, pushing each token onto the same `asyncio.Queue` the
   progress events use, as `{"type": "token", "content": ...}`.

The tricky part is disconnect handling: `main.py`'s SSE generator and the
graph execution run on different threads (the graph runs in a
`ThreadPoolExecutor` via `run_in_executor`, since it's a synchronous
LangGraph invocation). A client disconnect is detected by the async
generator's `finally` block (Starlette closes the generator on
disconnect), which sets a `threading.Event`. `stream_callback` checks
that event on every token and raises `StreamAborted` — a dedicated
exception type, not a generic failure — which `synthesize_node` treats
distinctly from a streaming hiccup: a hiccup falls back to a blocking
`llm.invoke()` to still produce a report; an abort stops immediately,
since there's no point spending further LLM output (or the trailing
eval/follow-up/save-session work) on a request nobody will receive.
Verified against a live server: abort fires within ~0.2s of actual
disconnect, not after full generation completes.

## 7. Fact verification

Layered on top of `CitationAgent`, not a replacement for it.
`CitationAgent` validates citation *existence* (does `[n]` point to a
real retrieved source) with pure regex, no LLM, always runs.
`FactVerificationAgent` runs one step later and asks a harder question:
does the source a citation *does* point to actually *support* the
specific sentence it's attached to?

Pipeline: split the validated report into sentences, keep only sentences
containing a surviving citation marker, pair each `(sentence,
citation_id)` with that source's already-collected text (no new
retrieval — reuses `state["sources"]`), then **one batched LLM call**
covering every claim (not one call per claim — the main cost control,
confirmed 10x fewer LLM calls and ~10x less latency on a 10-claim report
vs. a naive per-claim design). Each claim is cached individually
(`(sentence, source_text) -> verdict`, 7-day TTL) so only cache misses go
into that batch call.

**Failure contract:** `FactVerificationAgent.run()` never raises. Any
failure — LLM error, malformed JSON, wrong-length response — is caught
internally and degrades to `citation_verification=[]`,
`citation_confidence=None`, with a log line, leaving the report
`CitationAgent` already produced completely untouched. This matters
because `Agent.__call__`'s tracker/tracer wrapping would otherwise let an
uncaught exception propagate out of the node and crash the entire run in
`main.py`'s generic error handler — turning a perfectly good report into
a failed request over what should be a soft, recoverable degradation.

## 8. Risk analysis

The graph's final node. Distinguishes itself from fact verification by
scope: fact verification checks individual claims; risk analysis assesses
the report's *overall* reliability — contradictions, weak evidence,
missing perspectives, outdated information, source-quality concerns.

Almost everything it produces is a **pure heuristic with zero LLM
dependency** — deterministic, instant, reused directly from existing
modules rather than reimplemented:

| Signal | Source |
|---|---|
| Contradictions | `reflection._check_contradictions()` (unchanged) |
| Source diversity | `reflection._count_domains()` (unchanged) |
| Low-credibility ratio | `credibility.score_url()` per source |
| Single-source dominance | citation frequency from `citation_verification` (§7) |
| Weak/conflicting claims | `unsupported`/`partially_supported` verdicts, straight from §7 |
| Evidence volume | `len(state["retrieved_chunks"])` |
| Recency | year-mention scan over source text (new, simple regex) |

Only `recommended_follow_up_questions` needs an LLM call (phrasing
natural-language questions targeting the specific gaps found) — cached,
with a templated fallback derived directly from `evidence_gaps` if it
fails, so a clean report never spends an LLM call confirming it's clean
(`if not all_concerns: return []`, no cache lookup, no LLM call).

**Naming note:** `risk_score` here is **higher = more risk** — the
opposite polarity of `evaluator.py`'s pre-existing `hallucination_risk_score`,
where higher is better (despite its name, it's really a "citation health"
score). Both coexist in the same report; this divergence is deliberate,
documented in `risk_analysis.py`'s module docstring, not a bug.

## 8.5. Grounded generation and self-correction (Phase 5)

The graph's new final node: `revise` (`RevisionAgent`), running after
`risk_analyze`. Where `risk_analyze` *measures* reliability, `revise`
*acts* on it — removing or annotating claims fact verification already
flagged, and pruning report sections the sources never actually
supported in the first place.

**This agent makes zero LLM calls**, deliberately. Every other
enrichment agent needed an LLM for the one piece that's genuinely a
natural-language task; revision doesn't have an equivalent, and having
an LLM *rewrite* prose to fix hallucinations risks introducing new ones
in the rewrite itself. Everything here is deterministic post-processing
over data the pipeline already computed — measured at under 1ms even on
a 50-sentence, 20-citation report, effectively free next to the
seconds-scale LLM calls elsewhere in the pipeline.

What it does, in order:
1. **Claim-level revision** — using `citation_verification` (§7)
   directly: a sentence where every citation is `unsupported` is
   removed entirely; a sentence with a mix of verdicts keeps its
   supported citations and strips only the unsupported ones (the same
   surgical approach `CitationAgent` already uses for hallucinated ids);
   `partially_supported` sentences are kept with an inline qualifier
   annotation rather than deleted.
2. **Strict grounding mode** (`STRICT_GROUNDING_MODE`, default on) — any
   remaining numeric claim (a percentage, a decimal, an AUC value) gets
   checked against its own citation's source text; a number that
   doesn't appear verbatim gets an inline `[unverified]` marker. A
   separate pass applies the same check to markdown table rows, since a
   results table doesn't carry a per-row citation marker the way a
   sentence does.
3. **Report-type detection and section pruning** — this is the actual
   fix for the motivating bug: `ReportGeneratorAgent`'s academic prompt
   forces a rigid 9-section template (Introduction, Related Work,
   *Proposed Method*, *Model Architecture*, *Dataset*, *Experimental
   Results*, ...) regardless of whether the sources support any of the
   experimental sections. `revise` detects report type — Comparative
   Analysis / Research Survey / Literature Review from the question's
   own language, or "Experimental Study" only if the sources
   demonstrably contain methodology *and* metric signals (reusing
   `reflection.ACADEMIC_REQUIRED_SIGNALS` unchanged) — and removes the
   Proposed Method / Model Architecture / Dataset / Experimental Results
   sections outright when the report type doesn't warrant them and the
   question didn't explicitly ask for them. This is why a typical thin
   search snippet (no metrics keywords) produces a report where an
   entire fabricated results table is gone, not just flagged.
4. **Footer rebuild** — `CitationAgent` already appended the Sources
   footer *before* revision runs; since revision can remove citations
   from the body, `revise` recomputes which ids still appear in the
   revised body and rebuilds the footer from that set, so there are
   never orphaned reference entries pointing at citations no longer used
   anywhere in the text. Surviving citations keep their **original**
   numbers — nothing is renumbered.
5. **Grounding summary** — `report_type`, `claims_removed`,
   `claims_rewritten`, `unsupported_claims` (all as literal claim text,
   for transparency, matching `identified_risks`/`evidence_gaps`'s
   existing style), and `final_grounding_score` (0–100: the fraction of
   the original claim surface that's now cleanly grounded, with
   `partially_supported`/annotated claims counted at half weight).

Same failure contract as `fact_verification.py`/`risk_analysis.py`:
`run()` never raises; any internal failure returns the report completely
unchanged with empty grounding fields and a log line.

## 9. Caching

`agent/cache.py` — a `CacheBackend` interface (`get`/`set`/`delete`/`clear`)
with two implementations:

- **`DiskCacheBackend`** (default, if `diskcache` is installed) — SQLite-backed,
  persists across restarts, already thread-safe and process-safe.
- **`InMemoryCacheBackend`** (fallback) — a dict guarded by one
  `threading.Lock`. Coarse-grained on purpose: cache ops are
  microsecond-fast, so one lock for the whole store isn't a real
  bottleneck, and it guarantees no torn reads/writes under this
  codebase's `ThreadPoolExecutor`-heavy call sites. Verified under 50
  concurrent threads / 10,000 ops with zero corruption.

A thin `Cache` wrapper adds per-namespace SHA-256 key hashing, a default
TTL, and hit/miss/set metrics — independent of which backend is active.
Five namespaces exist today: `search`, `fetch`, `embed`, `verification`,
`risk`, each with its own TTL tuned to how stable that data actually is
(search results: 30 min; embeddings and verification/risk verdicts:
7 days — deterministic-ish given the same model/inputs).

**Swap-in point for Redis:** a future `RedisCacheBackend` only needs to
implement the same four methods — nothing in `tools.py`, `rag.py`, or any
agent needs to change. This wasn't left as an aspiration; the interface
boundary is exactly where a real migration would happen.

## 10. Observability and tracing

Two mechanisms, both pre-existing but only actually *wired up* starting
Phase 3 Milestone 1 (see `agent/observability.py` and `agent/tracing.py`):

- **`RunTracker`** — one row per research run in a `runs` table, one row
  per agent execution in `node_executions` (latency, success/failure).
  Powers `/api/metrics/runs`, `/api/metrics/nodes/{run_id}`,
  `/api/metrics/summary`.
- **`Tracer`** — a span tree per run (`Tracer.span(name, parent, inputs)`
  context manager), persisted as JSON, one root span with a child per
  agent. Powers `/api/traces`, `/api/traces/{id}`.

Both were built long before Phase 3 but never actually wrapped around a
node's execution — `Agent.__call__` (§2) is what finally activates them,
for every agent, automatically, with zero per-agent instrumentation code.

**Phase 4 adds a second, independent layer**: request-ID + timing
middleware in `main.py`, logging one summary line per HTTP request
(method, path, status, duration) with an `X-Request-ID` header echoed
back (client-supplied if present, generated otherwise). This is
deliberately separate from `RunTracker`/`Tracer` — those measure pipeline
internals; this measures the HTTP layer around it, including routes
`RunTracker` never sees (history, metrics, health, static files).
`LOG_FORMAT=json` switches log output to one JSON object per line for log
aggregators; default stays human-readable text for local development.

## 11. Database schema

SQLite, four tables, all in `db.py`/`agent/observability.py`/
`agent/tracing.py`/`agent/eval_regression.py`/`agent/benchmark.py`/`auth.py`:

- **`history`** — one row per completed research/debate run: question,
  report, sources (JSON), confidence, eval scores (5 columns + JSON
  blob), RAG/latency metadata, and — as of Milestones 3–4 —
  `citation_verification`/`citation_confidence` and
  `risk_score`/`risk_level`/`identified_risks`/`evidence_gaps`/
  `conflicting_claims`/`recommended_follow_up_questions`.
- **`runs`** / **`node_executions`** — observability (§10).
- **`traces`** — span-tree JSON per run (§10).

**Migration pattern:** every schema addition since Milestone 3 follows
the same safe pattern — `CREATE TABLE IF NOT EXISTS` (handles a fresh
database) plus an `_ensure_column()` helper that checks `PRAGMA
table_info` and runs `ALTER TABLE ADD COLUMN` only for columns that don't
already exist (handles an existing database from before that migration).
Verified against simulated pre-migration databases at every step — no
data loss, no "no such column" errors on upgrade.

## 12. Scalability considerations

Being direct about what this architecture does and doesn't handle today:

**Single-instance assumptions that would need to change for horizontal
scaling:**
- `guardrails.py`'s rate limiter and concurrency guard are in-process
  (a Python dict + `threading.Lock`), not shared across instances — the
  code's own comments already flag Redis as the intended upgrade path,
  matching the `CacheBackend` swap-in design in §9.
- SQLite is single-writer. Fine for the read-heavy, low-write-concurrency
  pattern of a research history table; would need Postgres (or SQLite in
  WAL mode as an intermediate step) before running multiple app instances
  against the same database file.
- The in-memory conversation-context list (`memory.py`) is module-level
  global state — correct for one process, would leak across users if
  naively run behind a load balancer with multiple workers sharing
  process memory (it wouldn't be shared across *processes*, but multiple
  concurrent users hitting the *same* process would see each other's
  context — a real bug to fix before real multi-tenant deployment, not
  yet addressed).

**What already scales within a single instance:** the concurrency guard
caps simultaneous research runs (default 3) specifically because each run
spawns multiple LLM calls and a thread pool of its own — this is a
deliberate backpressure mechanism, not an accidental limit. Caching (§9)
reduces the load each run places on external APIs. The multi-agent
pipeline's per-stage latency is visible via `RunTracker` (§10), so a real
bottleneck (not a guessed one) is what would drive the next optimization.

**Realistic next step for true horizontal scaling:** Redis for both the
cache backend (§9, already has the interface) and the rate
limiter/concurrency guard, Postgres for `history`/`runs`/`node_executions`/
`traces`, and moving the in-memory conversation context into whatever
session store fronts the Redis/Postgres pair. None of this requires
touching agent logic — it's entirely in the infrastructure layer the
Phase 3/4 work deliberately kept separated from the pipeline itself.

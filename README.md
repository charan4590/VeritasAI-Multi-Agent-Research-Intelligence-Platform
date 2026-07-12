# Research Agent — Enterprise AI Research Intelligence Platform

A supervisor-orchestrated multi-agent research system that plans its own
search strategy, retrieves evidence from the live web, uploaded PDFs, and
academic sources in parallel, writes a streamed report, and then checks
its own work: citation existence, per-claim fact verification, and a
heuristic reliability score — before you ever see the answer.

Built to demonstrate production engineering practice, not just an LLM
wrapped in a chat box: real observability, real caching with measured
speedups, real database migrations, real CI, real graceful failure modes
at every stage.

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](.github/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](backend/requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](#)

---

## Why this project

Most "AI agent" portfolio demos are a single LLM call in a chat box. This
one is built the way a real research-agent product would need to work:

- **It plans before it acts** — the question becomes several targeted
  search queries via an intent-aware planner, not one vague search.
- **It searches in parallel, not serially** — web, academic, and uploaded
  PDFs all queried concurrently; full-page content fetches for
  high-credibility sources run concurrently too (measured ~4.8x speedup).
- **It knows when to keep digging** — a multi-signal reflection step
  checks source count, domain diversity, subtopic coverage, and
  contradiction signals *before* spending an LLM call asking "is this
  enough."
- **Citations are checked, not trusted** — a non-LLM validation pass
  strips any citation that doesn't map to a real source; a separate LLM
  pass then checks whether the source it *does* point to actually
  supports the sentence it's attached to.
- **It grades its own reliability** — a final risk-analysis pass surfaces
  contradictions, weak evidence, single-source dominance, and outdated
  information as a score you can see, not a black box.
- **It fails gracefully, everywhere** — every agent added after the core
  pipeline (fact verification, risk analysis) has an explicit contract:
  never let a failure in an enrichment step corrupt or block the report
  that's already good.

See **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)** for the full technical
writeup, including the tradeoffs and a couple of real bugs this project's
own CI adoption caught and fixed (not hidden — documented).

## Architecture

```
planner → search → reflect → rag → synthesize → validate → fact_verify → risk_analyze
             ▲__________|
     (loop back to search while reflection says insufficient, capped at max_rounds)
```

Every stage except `reflect` is a class implementing a shared `Agent`
interface, orchestrated by a `SupervisorAgent` for the search step
specifically (routes between web/academic/PDF search, then runs shared
concurrent-fetch/dedup/credibility-sort logic). Full breakdown, including
*why* it's structured this way, in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

## Features

**Research pipeline**
- Intent-aware planning (academic / technical / general), each with a
  tailored search and synthesis strategy
- Parallel web + academic + PDF retrieval, with concurrent full-page
  content fetching for high-credibility sources
- Multi-signal reflection (source count, domain diversity, subtopic
  coverage, contradiction detection) with an LLM fallback only when
  heuristics don't already have the answer
- RAG: per-session vector index, semantic retrieval, cross-encoder
  re-ranking
- **Real-time token streaming** — the report appears as it's written, not
  after a 20–90 second wait
- **Upload your own PDFs** — searched alongside web sources, same
  citation numbering, clearly labeled in the report
- **Per-claim fact verification** — catches a real citation attached to a
  claim the source never actually made (existence ≠ truthfulness)
- **Risk analysis** — a 0–100 reliability score, identified risks,
  evidence gaps, conflicting claims, and targeted follow-up questions
- **Self-correcting revision** — removes or clearly marks unsupported
  claims (including fabricated numbers and results tables) using the
  fact-verification pass above, and prunes report sections (e.g.
  "Experimental Results") the sources never actually supported — zero
  extra LLM calls, pure deterministic post-processing
- **IEEE-style References** with full (untruncated) titles, deduplicated
  by URL, plus a separate "Sources Retrieved During Research" view
  showing every source gathered — not just the ones cited — with
  credibility, type, and cited/not-cited status
- Debate mode — a separate FOR/AGAINST two-sided research pipeline

**Engineering**
- Pluggable caching (disk-backed, falls back to in-memory automatically)
  across search, fetch, embeddings, verification, and risk analysis —
  measured cache-hit speedups from ~5x to ~10,000x depending on stage
- Full observability: per-agent latency/success tracking and a span-tree
  tracer, both wired automatically via the `Agent` base class
- Request-ID + timing middleware, structured (optional JSON) logging
- SQLite with safe, tested schema migrations (no data loss across
  upgrades) — history, observability, and evaluation-regression tables
- 117+ automated tests, CI-enforced (lint, format, test, Docker build +
  smoke test) on every push

## Tech stack

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph (`StateGraph`, conditional edges) |
| LLM | Groq (cloud, free tier) or Ollama (fully local) |
| Search | Tavily |
| Vector store | ChromaDB |
| Backend | FastAPI, Python 3.11 |
| Caching | `diskcache` (falls back to an in-process implementation) |
| Database | SQLite |
| Frontend | Single-file HTML/CSS/JS, no build step, SSE for live updates |
| CI | GitHub Actions — Ruff, Black, Pytest, Docker build + smoke test |
| Deployment | Docker / Docker Compose; guides for Render, Railway, Fly.io |

## Demo

The fastest way to see this working end to end:
```bash
make dev && make run
```
then open `http://localhost:8000` and ask something with a genuine
evidence trail, e.g. *"What are recent advances in hybrid CNN-LSTM
architectures for medical imaging?"* Watch for, in order: the report
streaming in token-by-token, citation badges appearing with small
colored verdict icons once fact-checking finishes, a Risk Analysis card
with a Low/Medium/High badge, a Grounding card showing what (if
anything) got removed or flagged, and — expand it — a "Sources Retrieved
During Research" list showing every source the agent gathered, including
ones it decided *not* to cite.

## Screenshots

*(No browser is available in the environment this project was built in,
so these are accurate descriptions of the real, implemented UI rather
than literal image files — capture your own from a local run before
sharing this repo publicly; `frontend/index.html` is a single static
file, so `File → Save As...` or your browser's screenshot tool is enough.)*

**Report view.** Citation badges (`[1]`, `[2]`, ...) render inline with a
small colored glyph next to the number — green check (supported), amber
tilde (partially supported), red X (unsupported), gray question mark
(couldn't verify) — with a hover tooltip showing the reasoning and
confidence. The References section at the bottom uses IEEE-style
formatting: `[1] "Full Untruncated Title," [Online]. Available: URL`.

**Confidence and Citation Verification bars.** Two stacked indicators
above the report: an overall confidence score (credibility-weighted), and
a citation-verification score (`N claims checked`) once fact verification
completes.

**Risk Analysis card.** A colored score bar (green/amber/red for
Low/Medium/High) with three labeled lists underneath — Identified Risks
(amber left border), Evidence Gaps (blue), Conflicting Claims (red) — and
clickable Recommended Follow-up Questions that re-run research with that
question.

**Grounding card.** A report-type badge (e.g. "Experimental Study" or
"General Research Answer"), a grounding score, and counts of claims
removed vs. annotated, with the removed claims themselves shown
strikethrough for transparency.

**Sources Retrieved During Research.** Expandable cards (native
`<details>`, no extra JS needed to toggle), one per gathered source,
sorted cited-first then by credibility — each shows a Cited/Not Cited
badge, a Web/PDF badge, a credibility badge, the full URL, and — for
cited sources — the fact-verification verdict.

## Installation

```bash
git clone <this-repo>
cd research-agent
make dev            # installs deps, creates backend/.env from the template
# edit backend/.env: TAVILY_API_KEY (required) + GROQ_API_KEY or a local Ollama
make run            # uvicorn --reload on http://localhost:8000
```

Or without `make`:
```bash
cd backend
pip install -r requirements-dev.txt
cp .env.example .env   # then fill in your keys
uvicorn main:app --reload
```

## Running with Docker

```bash
cp backend/.env.example backend/.env   # fill in your keys
docker compose up --build
open http://localhost:8000
```

Data (SQLite history, disk cache, Chroma index) persists in a named
Docker volume across restarts and rebuilds. Fully local (no LLM API key
needed) via the optional bundled Ollama service — see
[DEPLOY.md](DEPLOY.md#local-development-with-docker).

## Deployment

Guides for **Render**, **Railway**, and **Fly.io** — required environment
variables, persistence setup, and recommended plan sizing for each — are
in **[DEPLOY.md](DEPLOY.md)**.

## API examples

Interactive docs at `/docs` (Swagger UI) once running. A few highlights:

```bash
# Health + version
curl http://localhost:8000/api/health
curl http://localhost:8000/api/version

# Run research (Server-Sent Events -- consume with an EventSource client,
# not a plain HTTP client expecting a single JSON body)
curl -N "http://localhost:8000/api/research/stream?question=What+are+recent+advances+in+hybrid+CNN-LSTM+architectures&max_rounds=2"

# Past sessions
curl http://localhost:8000/api/history

# Per-run observability
curl http://localhost:8000/api/metrics/summary
curl "http://localhost:8000/api/metrics/nodes/{run_id}"
```

Example `done` SSE event (trimmed):
```json
{
  "type": "done",
  "report": "## Summary\n\n...[1]...[2]...",
  "confidence": 82,
  "citation_verification": [
    {"citation_id": 1, "verdict": "supported", "confidence": 92, "reasoning": "..."}
  ],
  "citation_confidence": 78,
  "risk_score": 35,
  "risk_level": "Medium",
  "identified_risks": ["..."],
  "recommended_follow_up_questions": ["..."],
  "report_type": "Experimental Study",
  "claims_removed": [],
  "final_grounding_score": 91,
  "sources": {"1": {"title": "...", "url": "...", "credibility": 95, "cited": true}},
  "all_sources": {
    "1": {"title": "...", "url": "...", "credibility": 95, "source_type": "web", "cited": true},
    "2": {"title": "...", "url": "...", "credibility": 50, "source_type": "web", "cited": false}
  }
}
```

## Performance benchmarks

Measured, not estimated — see each milestone's original writeup for full
methodology:

| Optimization | Result |
|---|---|
| Token streaming (perceived latency) | Report visible immediately instead of after full generation |
| Parallel full-content fetch | ~4.8x speedup (realistic per-URL latency simulation) |
| Search result caching | Repeated/overlapping queries: near-zero latency on cache hit |
| Embedding caching | ~74x speedup on repeated PDF re-ingestion |
| Fact-verification batching | 10x fewer LLM calls vs. one-call-per-claim |
| Fact-verification caching | ~10,700x speedup on an identical repeat report |
| PDF search overhead when no PDFs uploaded | 0.26ms/query (worst case, unconditional call) |
| Self-correcting revision (claim removal, grounding, section pruning) | ~0.58ms average on a 50-sentence/20-citation report — zero LLM calls |

## Roadmap

- Knowledge Graph Agent (Neo4j) — deferred from Phase 3, entities/
  relationships extracted across sources
- `ReflectAgent` — convert the remaining plain-function graph node to the
  `Agent` interface for consistency and per-agent observability
- Redis-backed cache and rate-limiter backends for true horizontal
  scaling (interface already supports this — see
  [SYSTEM_DESIGN.md §9](SYSTEM_DESIGN.md#9-caching))
- Postgres migration path for multi-instance deployment
- OCR / multimodal PDF ingestion (scanned documents, embedded figures)

## Documentation

- **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)** — full architecture, every
  design decision explained
- **[DEPLOY.md](DEPLOY.md)** — Render / Railway / Fly.io deployment guides
- **[RESUME_BULLETS.md](RESUME_BULLETS.md)** — ATS-friendly resume bullets
  for this project
- **[INTERVIEW_TALKING_POINTS.md](INTERVIEW_TALKING_POINTS.md)** — concise
  explanations of the key engineering decisions, for interview prep

## License

MIT

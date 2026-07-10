# Resume Bullets

ATS-friendly bullets for this project, grouped by what they emphasize.
Pick 3–5 that match the role you're applying for rather than using all of
them — a resume with 15 bullets from one project reads as padding, not
signal.

## If the role emphasizes AI/ML systems engineering

- Architected a supervisor-orchestrated multi-agent research pipeline
  (LangGraph, Python) with 9 specialized agents — planning, parallel
  retrieval, RAG, synthesis, citation validation, fact verification, and
  risk analysis — each independently testable and automatically
  instrumented via a shared base-class interface.
- Designed and implemented a claim-level fact-verification system that
  checks per-citation entailment (not just citation existence), using
  batched LLM calls to reduce verification cost 10x versus a
  naive per-claim approach, with per-claim caching and graceful
  degradation on LLM failure.
- Built a heuristic risk-scoring system (contradiction detection, source
  diversity, credibility analysis, evidence-volume checks) with zero LLM
  dependency for its core score, reusing and composing existing detection
  modules rather than duplicating logic — deterministic and instant even
  under LLM outages.
- Implemented real-time token streaming for LLM-generated reports over
  Server-Sent Events, including correct handling of client disconnects
  mid-stream (verified sub-200ms abort detection against a live server)
  to avoid wasted LLM spend on abandoned requests.

## If the role emphasizes backend/API engineering

- Built a FastAPI backend with 22 REST/SSE endpoints, OpenAPI
  documentation (tagged, with request/response models), request-ID and
  timing middleware, and structured JSON logging for production log
  aggregation.
- Designed a pluggable caching layer (interface-based, disk-backed with
  automatic in-memory fallback) spanning 5 cache domains, measuring
  cache-hit speedups from 5x to over 10,000x depending on the operation,
  verified thread-safe under 50 concurrent threads / 10,000 operations
  with zero data corruption.
- Implemented safe, zero-downtime SQLite schema migrations (additive
  `ALTER TABLE` with existence checks) across 4 successive schema
  versions, tested against simulated pre-migration databases to guarantee
  no data loss on upgrade.
- Parallelized sequential I/O-bound retrieval (web search, full-page
  content fetching, PDF search) using `ThreadPoolExecutor`, reducing a
  6-source fetch phase from a theoretical 48s worst case to under 8s.

## If the role emphasizes production/DevOps readiness

- Set up a 4-job GitHub Actions CI pipeline (Ruff lint, Black format
  check, Pytest, Docker build + live health-check smoke test) that caught
  and fixed 2 real latent bugs during adoption, including an unreachable
  dead-code block and a Python exception-scoping bug that would have
  crashed a production error-handling path.
- Authored a multi-stage Docker build (separate builder/runtime stages,
  non-root user, health check, build-time git-commit tagging) and a
  Docker Compose configuration with named-volume data persistence and an
  optional fully-local LLM profile.
- Wrote deployment guides for 3 platforms (Render, Railway, Fly.io)
  covering required environment variables, persistent-volume
  configuration, and platform-specific gotchas (e.g., container-internal
  networking for a bundled LLM service).

## If the role emphasizes testing/quality

- Wrote 80+ automated tests (pytest) covering graceful-fallback behavior,
  cache-hit verification, database migration safety, and multi-scenario
  agent regression testing (contradictory sources, weak evidence,
  single-source dominance, missing data) — all passing in CI.
- Diagnosed and fixed a real chunking-overlap bug (a code path silently
  ignoring a documented parameter under specific input conditions) that
  surfaced only when the existing test suite was made CI-blocking, rather
  than leaving it as a known/ignored failure.

## One-line summary (for a projects list, not a bullet list)

Enterprise-grade AI research platform: supervisor-orchestrated multi-agent
pipeline (LangGraph) with parallel retrieval, streaming synthesis,
claim-level fact verification, and heuristic risk analysis — full CI/CD,
Docker deployment, 80+ tests, and measured performance benchmarks
throughout.

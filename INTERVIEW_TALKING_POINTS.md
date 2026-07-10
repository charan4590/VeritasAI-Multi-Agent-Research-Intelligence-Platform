# Interview Talking Points

Concise explanations of the key engineering decisions in this project —
written to be *said out loud*, not read. Each one follows the same shape:
what the decision was, why, and what the honest tradeoff is. Interviewers
notice when a candidate can state the tradeoff unprompted.

## "Walk me through the architecture."

"It's a LangGraph pipeline with nine stages — planning, search, a
reflection loop, RAG, synthesis, and then three enrichment stages:
citation validation, fact verification, and risk analysis. The search
stage is the interesting one architecturally — it's owned by a
`SupervisorAgent` that routes between web search, academic search, and
PDF search based on query intent, then runs shared post-processing
(deduping, concurrent full-content fetching, credibility sorting) on
whichever source answered. Every other stage is a single-purpose agent
implementing a shared interface, so adding a new stage — which I did
three times, fact verification and risk analysis being the last two — is
a small, reviewable diff: implement one method, register it as a graph
node, done."

## "Why fact verification *and* risk analysis — aren't those redundant?"

"They check different things. Citation validation checks *existence* —
does `[3]` point to a source that's actually in the retrieved set. Fact
verification checks *entailment* — does that source's text actually
support the specific sentence it's attached to. That's the interesting
failure mode: a citation can be completely valid and still be wrong,
because the model attached a real source to a claim that source never
made. Risk analysis is a level up from both — it's not about individual
claims, it's about the report's overall reliability: are there
contradicting sources, is the evidence thin, is one source doing all the
work. I made the risk score deterministic and heuristic-only wherever
possible specifically so it can never silently fail — it reuses
contradiction detection and domain-diversity checks that already existed
in the reflection step, rather than reimplementing them."

## "How did you handle LLM failures?"

"Every enrichment stage added after the core pipeline has an explicit
contract: never let a failure corrupt or block a report that's already
good. Concretely, fact verification's `run()` method never raises — if
the LLM call fails, returns malformed JSON, or returns the wrong number
of results, it catches that internally and returns the report completely
unchanged, just with an empty verification list and a log line. That
matters because of *how* my agents are instrumented — every agent is
wrapped in a tracker/tracer context manager via a shared base class, and
if I let an exception propagate out of `run()`, that wrapping would
correctly record it as a node failure, but the exception would keep
propagating up into the main request handler and turn a perfectly good,
already-validated report into a failed HTTP request. So the fallback has
to happen inside the agent, not at the top level."

## "Tell me about a real bug you found."

"Two, actually, both surfaced by the same thing: making CI actually
blocking instead of advisory. First, adopting Ruff's pyflakes checks
found a genuinely undefined variable reference in a dead code block —
turned out to be an orphaned copy-paste leftover sitting after a
`return` statement, syntactically valid but unreachable, that had been
sitting there for a while. Second, and more interesting: the same lint
pass found a variable used in a nested closure that Python would delete
before the closure ever ran. It was in the rate-limit error-handling
path — `except RateLimitExceeded as exc:` followed by a generator
function that referenced `exc` inside a `yield`. Python automatically
clears `as`-bound exception variables at the end of the except block, but
that generator wasn't actually iterated until later, asynchronously,
after the containing function had already returned — by which point
`exc` was already gone. So anyone who actually got rate-limited would
have hit a `NameError` instead of getting the intended error message. I
reproduced it with a quick test, fixed it by capturing the message as a
plain string before the except block exits, and verified the fix. I also
found and fixed a real overlap bug in the RAG chunking function once I
made the existing test suite CI-blocking instead of just having a known
failing test — a fallback code path for pathologically long text was
silently ignoring the `overlap` parameter entirely."

## "How does caching work, and how do you know it's actually safe?"

"It's an interface — `get`/`set`/`delete`/`clear` — with two
implementations: disk-backed by default, falling back automatically to
an in-memory implementation if that dependency isn't installed. The
in-memory one is a plain dict behind a single lock, which sounds naive,
but cache operations are microsecond-fast, so one coarse-grained lock
isn't actually a bottleneck, and it's *provably* correct — I stress-tested
it with 50 concurrent threads doing 10,000 total operations and asserted
zero corrupted reads. The reason it's an interface at all, for a project
this size, is that a Redis backend is the obvious next step for running
more than one instance, and I wanted that to be a drop-in swap — nothing
above the cache boundary should need to change."

## "What would you do differently at scale?"

"A few things are explicitly single-instance right now, and I'd say that
directly rather than pretend otherwise: the rate limiter and concurrency
guard are in-process state, not shared across instances. SQLite is
single-writer, which is fine for a read-heavy history table under one
process but wouldn't hold up with multiple app instances writing
concurrently. And there's an in-memory conversation-context list that's
correct for one process but would need to move into a real session store
before this was genuinely multi-tenant. None of that requires touching
the agent logic, though — it's all in the infrastructure layer, and the
cache interface in particular was built so that swapping in Redis is
additive, not a rewrite."

## "Why does the risk score go the opposite direction from your other quality score?"

"Because I made a deliberate call that `risk_score` should read
naturally against `risk_level` — higher score, higher risk. There's
already a `hallucination_risk_score` elsewhere in the codebase from
earlier work, and despite its name, higher is *better* there — it's
really measuring citation health, inherited from a scoring convention
where every metric in that group is 'bigger is better.' I didn't want to
force my new metric into that convention just for consistency, because
it would have made `risk_level: High` correspond to a *low* number, which
is the kind of thing that causes a real bug six months later when someone
skims the code instead of reading it carefully. I documented the
divergence explicitly in the module docstring specifically so it's not a
surprise."

## "What's the single decision you'd defend hardest if someone pushed back?"

"Keeping the core risk-analysis output — score, level, identified risks,
evidence gaps, conflicting claims — completely free of LLM dependency,
and only using an LLM for the one part that's genuinely a
natural-language task, the follow-up question phrasing. It would have
been faster to just ask an LLM 'summarize the risks in this report' and
call it done. But that makes your most important reliability signal only
as reliable as the model call that produced it, which is a strange place
to introduce fragility into a feature whose entire purpose is measuring
reliability. Reusing the existing heuristic detectors instead — most of
which already existed in the reflection module — meant the score is
deterministic, instant, and testable without mocking an LLM at all."

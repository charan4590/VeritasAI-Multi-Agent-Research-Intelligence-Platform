# Research agent

A research agent that plans its own search strategy, gathers evidence from
the live web, decides for itself whether it has enough evidence, and writes
a report where every citation is checked against an actual retrieved source
before it reaches you. Built with LangGraph, the Anthropic API, and Tavily
search, with a small FastAPI backend and a single-page frontend that shows
the agent's reasoning live as it works.

## Why this project

Most "AI agent" demos are a single LLM call wrapped in a chat box. This one
is closer to how production research agents actually need to behave:

- **It plans before it acts.** The question is broken into several targeted
  search queries instead of one vague search.
- **It knows when to keep digging.** A reflection step judges whether the
  evidence collected so far actually supports an answer, and loops back to
  search again (up to a capped number of rounds) if not.
- **Citations are verified, not trusted.** The model is told which sources
  exist and asked to cite them by id. After it writes the report, a
  non-LLM validation step strips out any citation number that doesn't map
  to a real retrieved source. The model can't hallucinate a citation that
  survives into the final report.

## Architecture

```
question -> planner -> search -> reflect -+-> (back to search, if more evidence is needed)
                                            |
                                            +-> synthesize -> validate -> report
```

- **planner** asks Claude for 3-5 specific search queries.
- **search** runs those queries against Tavily and stores each unique
  result as a numbered source.
- **reflect** asks Claude to judge, given the sources gathered so far,
  whether they're sufficient to answer the question. If not, it proposes
  follow-up queries and the graph loops back to search (capped at
  `max_rounds`, default 2).
- **synthesize** asks Claude to write the report in markdown, citing only
  from the numbered source list.
- **validate** is plain Python, not an LLM call: it extracts every `[n]`
  citation marker, drops any id that doesn't correspond to a real source,
  and appends a reference list built only from citations that were
  actually used.

The backend streams progress for each of these steps to the frontend over
Server-Sent Events, so the activity log updates live instead of showing a
single spinner for 20+ seconds.

## Project structure

```
research-agent/
  backend/
    main.py              FastAPI app: SSE endpoint + serves the frontend
    agent/
      state.py            Shared state and pydantic schemas
      tools.py             Tavily search wrapper
      graph.py             The LangGraph pipeline itself
    requirements.txt
    .env.example
  frontend/
    index.html             Single-file UI (no build step)
```

## Setup

1. **Get API keys**
   - Anthropic: https://console.anthropic.com/
   - Tavily (free tier, 1000 searches/month): https://tavily.com

2. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # then edit .env and paste in your keys
   ```

4. **Run it**
   ```bash
   uvicorn main:app --reload
   ```
   Open http://localhost:8000 — the same server serves both the API and
   the frontend, so there's nothing else to start.

## Extending this project

A few directions that would each make a good follow-up portfolio entry:

- **Streaming token-by-token output** instead of waiting for the full
  synthesize step to finish, using LangGraph's `astream` with
  `stream_mode="messages"`.
- **Per-claim citation checking**: instead of just verifying that a cited
  id exists, ask a second LLM pass whether the cited source actually
  supports the specific sentence it's attached to.
- **Persistent research sessions**: save past queries and reports (e.g. to
  SQLite) so you can build a history view.
- **Swap Tavily for your own document store** to turn this into a hybrid
  web + internal-knowledge research agent.

## Notes on cost

Each question makes 2-4 Claude calls (planner, 1-2 reflect calls,
synthesize) plus a handful of Tavily searches. With Claude Sonnet this
typically costs a few cents per question. Lower `max_rounds` in the
`/api/research/stream` request (or the default in `main.py`) to cap
spend further.

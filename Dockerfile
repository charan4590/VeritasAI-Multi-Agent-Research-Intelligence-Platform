# syntax=docker/dockerfile:1
#
# Multi-stage build: a "builder" stage compiles/installs Python deps (needs
# gcc/g++ for chromadb's native deps and sentence-transformers), then the
# "runtime" stage copies only the installed packages + app code into a
# clean slim image — keeps the final image smaller and avoids shipping a
# C compiler in production.

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .
# Install into a venv rather than system site-packages -- the whole venv
# directory gets copied to the runtime stage in one COPY --from=builder,
# which is simpler and more reliable than trying to figure out exactly
# which system paths pip touched.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Baked in at build time via `docker build --build-arg GIT_COMMIT=$(git rev-parse --short HEAD)`.
# Surfaced at GET /api/version -- "unknown" if not passed (e.g. `docker build`
# with no --build-arg, which is fine, just less traceable).
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

# Run as a non-root user -- standard production hardening, not optional
# for anything internet-facing.
RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Mount point for persistent state (SQLite DB, disk cache, Chroma index).
# docker-compose.yml mounts a named volume here; DB_PATH/CACHE_DIR/
# CHROMA_PATH in .env.example already point at ../data/* relative to
# backend/'s WORKDIR below, i.e. exactly this directory.
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser
WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# --timeout-graceful-shutdown: how long uvicorn waits for in-flight
# requests to finish after SIGTERM before forcing exit. See main.py's
# lifespan() docstring for the full graceful-shutdown explanation.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]

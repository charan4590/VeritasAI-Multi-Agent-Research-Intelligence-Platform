FROM python:3.11-slim

# System dependencies for chromadb + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

EXPOSE 8000

# Healthcheck so Docker knows when app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#!/usr/bin/env bash
# Quick health check against a running instance -- useful after
# `make run` / `docker compose up`, or wired into a deployment platform's
# own health-check mechanism (most already hit /api/health directly and
# don't need this, but it's handy for a manual sanity check that also
# reports version/graph-node info, not just "200 OK").
#
# Usage: ./scripts/healthcheck.sh [port] [host]

set -euo pipefail

PORT="${1:-8000}"
HOST="${2:-localhost}"
BASE_URL="http://${HOST}:${PORT}"

echo "Checking ${BASE_URL}/api/health ..."

HEALTH_RESPONSE=$(curl -sf -w '\n%{http_code}' "${BASE_URL}/api/health" 2>/dev/null) || {
  echo "FAIL: could not reach ${BASE_URL}/api/health -- is the app running?"
  exit 1
}

HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
BODY=$(echo "$HEALTH_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" != "200" ]; then
  echo "FAIL: /api/health returned HTTP ${HTTP_CODE}"
  echo "$BODY"
  exit 1
fi

echo "OK: $BODY"

echo "Checking ${BASE_URL}/api/version ..."
VERSION_RESPONSE=$(curl -sf "${BASE_URL}/api/version" 2>/dev/null) || {
  echo "WARN: /api/version unreachable (non-fatal -- /api/health already passed)"
  exit 0
}
echo "OK: $VERSION_RESPONSE"

echo ""
echo "Instance is healthy."

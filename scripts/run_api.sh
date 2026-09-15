#!/usr/bin/env bash
# Runs the FastAPI scoring service in the foreground.
# Requires infrastructure running (./scripts/start.sh) and a registered
# model (python -m src.ml.train).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  source .venv/bin/activate
fi

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"

exec uvicorn src.api.main:app --host "$API_HOST" --port "$API_PORT"

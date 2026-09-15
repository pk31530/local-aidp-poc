#!/usr/bin/env bash
# Runs the Streamlit dashboard in the foreground.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  source .venv/bin/activate
fi

DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"

exec streamlit run src/dashboard/app.py \
  --server.address "$DASHBOARD_HOST" \
  --server.port "$DASHBOARD_PORT" \
  --server.headless true

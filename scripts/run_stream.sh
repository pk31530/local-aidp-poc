#!/usr/bin/env bash
# Starts the live transaction generator (producer). Pass through any
# stream_transactions.py flags, e.g.:
#   ./scripts/run_stream.sh --rate 10
#   ./scripts/run_stream.sh --rate 10 --duration 30
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  source .venv/bin/activate
fi

python -m src.generator.stream_transactions "$@"

#!/usr/bin/env bash
# Starts the streaming consumer (Redpanda -> feature engineering -> model ->
# decision -> PostgreSQL -> MinIO). Pass through any consumer.py flags.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  source .venv/bin/activate
fi

python -m src.ingestion.consumer "$@"

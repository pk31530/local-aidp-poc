#!/usr/bin/env bash
# Generates synthetic customers/transactions and loads customer_profiles
# into Postgres. Requires infrastructure to be running (./scripts/start.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  source .venv/bin/activate
fi

python -m src.generator.seed "$@"

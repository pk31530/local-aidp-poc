#!/usr/bin/env bash
# First-time setup: creates .env from .env.example if missing, starts
# infrastructure, and waits for it to become healthy.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example (local development credentials only)."
fi

./scripts/start.sh

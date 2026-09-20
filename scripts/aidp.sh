#!/usr/bin/env bash
# Unified aidp operator CLI wrapper. Forwards all arguments, e.g.:
#   ./scripts/aidp.sh config validate --json
#   ./scripts/aidp.sh run list --limit 10 --json
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  source .venv/bin/activate
fi

python -m src.cli "$@"

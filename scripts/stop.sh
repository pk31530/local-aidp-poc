#!/usr/bin/env bash
# Stops and removes POC containers. Named volumes (postgres_data, minio_data,
# redpanda_data) are NOT removed, so data survives — this is not a reset.
# Use scripts/reset_demo.sh to clear demo data specifically.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose down

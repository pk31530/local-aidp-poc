#!/usr/bin/env bash
# Starts (or resumes) all infrastructure containers and waits until the
# compose-level healthchecks report healthy.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "No .env found. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi

docker compose up -d

echo "Waiting for services to become healthy..."
for i in $(seq 1 60); do
  unhealthy=$(docker compose ps --format '{{.Service}} {{.Health}}' 2>/dev/null | awk '$2 != "" && $2 != "healthy" {print $1}')
  if [ -z "$unhealthy" ]; then
    echo "All services healthy."
    break
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    echo "Timed out waiting for: $unhealthy" >&2
  fi
done

docker compose ps

cat <<'EOF'

Local URLs:
  MinIO Console:    http://127.0.0.1:9001
  Redpanda Console: http://127.0.0.1:8080
  MLflow:           http://127.0.0.1:5001
EOF

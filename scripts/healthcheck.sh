#!/usr/bin/env bash
# Checks the runtime health of every POC service and prints [OK]/[FAIL] lines.
# Safe to run at any time; does not modify state. FastAPI/Streamlit checks will
# show [FAIL] until Phase 6/Phase 8 are built and started — that is expected
# before then, not a bug.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

POSTGRES_USER="${POSTGRES_USER:-aidp}"
POSTGRES_DB="${POSTGRES_DB:-aidp}"
MLFLOW_PORT=5001
API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"

overall_status=0

check_ok() { echo "[OK] $1"; }
check_fail() { echo "[FAIL] $1 - $2"; overall_status=1; }

# --- PostgreSQL ---
if docker compose exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
  check_ok "PostgreSQL"
else
  check_fail "PostgreSQL" "connection refused or container not running"
fi

# --- MinIO ---
if curl -fsS "http://127.0.0.1:9000/minio/health/live" >/dev/null 2>&1; then
  check_ok "MinIO"
else
  check_fail "MinIO" "connection refused or container not running"
fi

# --- Redpanda ---
if docker compose exec -T redpanda rpk cluster health 2>/dev/null | grep -qE 'Healthy:.+true'; then
  check_ok "Redpanda"
else
  check_fail "Redpanda" "cluster not healthy or container not running"
fi

# --- MLflow ---
if curl -fsS "http://127.0.0.1:${MLFLOW_PORT}/health" >/dev/null 2>&1; then
  check_ok "MLflow"
else
  check_fail "MLflow" "connection refused or container not running"
fi

# --- FastAPI ---
if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
  check_ok "FastAPI"
else
  check_fail "FastAPI" "connection refused (not started, or Phase 6 not yet built)"
fi

# --- Streamlit ---
if curl -fsS "http://127.0.0.1:${DASHBOARD_PORT}/_stcore/health" >/dev/null 2>&1; then
  check_ok "Streamlit"
else
  check_fail "Streamlit" "connection refused (not started, or Phase 8 not yet built)"
fi

exit $overall_status

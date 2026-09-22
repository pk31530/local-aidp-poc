#!/usr/bin/env bash
#
# End-to-end Debit Card fraud-intelligence demonstration, executed against a
# fully ISOLATED demo stack (Postgres + MLflow + MinIO), never against the
# shared `aidp` / `aidp_test` databases or the shared MLflow/MinIO servers.
#
#   Stages (A-O of docs/DEBIT_CARD_FULL_LIFECYCLE_DEMO.md):
#     preflight -> generate -> generate_retry -> verify -> split_preview
#     -> train -> evaluate_candidate -> diagnostic -> approval (HALT)
#     -> promote -> score -> score_retry -> labels -> labels_retry
#     -> live_evaluate -> audit
#
#   Usage:
#     scripts/demo_debit_card_full_lifecycle.sh                       # A..H, halts for approval
#     scripts/demo_debit_card_full_lifecycle.sh --preflight-only
#     scripts/demo_debit_card_full_lifecycle.sh --resume-from promote --approve-promotion
#     scripts/demo_debit_card_full_lifecycle.sh --cleanup
#
# Isolation is derived from source, not guessed:
#   * src/common/config.py -> Settings(BaseSettings) with pydantic-settings:
#     real process environment variables take PRECEDENCE over .env, so
#     exporting POSTGRES_*/MINIO_*/MLFLOW_* here retargets every CLI process
#     without touching .env at all.
#   * src/dashboard/data.py::_dashboard_database() reads
#     settings.postgres_test_db -- so POSTGRES_TEST_DB is also pinned to
#     aidp_demo, removing the last code path that could reach aidp_test.
#   * docker-compose.demo.yml is a separate compose project (aidp-demo) with
#     its own container names, network, volumes and host ports, because
#     docker-compose.yml hardcodes container_name and several host ports.
#
# This script never writes to `aidp`, `aidp_test`, the shared MLflow backend
# or the shared MinIO. --cleanup is scoped to the aidp-demo compose project
# only, and is never run automatically.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Repository resolution and identity
# ---------------------------------------------------------------------------

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SCRIPT_SOURCE" ]; do
  SCRIPT_LINK_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
  SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
  case "$SCRIPT_SOURCE" in
    /*) : ;;
    *) SCRIPT_SOURCE="$SCRIPT_LINK_DIR/$SCRIPT_SOURCE" ;;
  esac
done
SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
REPO_ROOT="$(cd -P "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# 1. Demo parameters (fixed by the demonstration brief)
# ---------------------------------------------------------------------------

# Exported so the embedded Python helpers read exactly these values rather
# than re-deriving or hardcoding them.
export DEMO_CHANNEL="debit_card"
export DEMO_COUNT="5000"
export DEMO_SEED="99"
export DEMO_REFERENCE_DATE="2026-09-22"
export DEMO_DATABASE="aidp_demo"
export DEMO_ANALYST_CAPACITY="100"
export DEMO_RECALL_TARGET="0.8"
export DEMO_PROMOTED_BY="prabhat-kumar"

# Databases this script must never use or contact.
export FORBIDDEN_DATABASES="aidp aidp_test"

# Isolated infrastructure identity.
DEMO_COMPOSE_PROJECT="aidp-demo"
DEMO_COMPOSE_FILE="$REPO_ROOT/docker-compose.demo.yml"
DEMO_ENV_EXAMPLE="$REPO_ROOT/.env.demo.example"
DEMO_ENV_LOCAL="$REPO_ROOT/.env.demo.local"
DEMO_CONTAINER_PREFIX="aidp-demo-"

DEMO_WORK_DIR="$REPO_ROOT/.demo"
DEMO_STATE_FILE="$DEMO_WORK_DIR/state.json"
# Stable (not per-run) path: stage G writes it and stage J -- typically a
# SEPARATE invocation after manual approval -- reads it back to prove the
# persisted scores match the pre-promotion diagnostic.
DEMO_DIAGNOSTIC_SCORES_FILE="$DEMO_WORK_DIR/diagnostic_scores.json"

# Ordered stage list. `approval` is a halt point, not a workload.
DEMO_STAGES="preflight generate generate_retry verify split_preview train evaluate_candidate diagnostic approval promote score score_retry labels labels_retry live_evaluate audit"

# ---------------------------------------------------------------------------
# 2. Argument parsing
# ---------------------------------------------------------------------------

OPT_PREFLIGHT_ONLY=0
OPT_RESUME_FROM=""
OPT_APPROVE_PROMOTION=0
OPT_CLEANUP=0
OPT_ASSUME_YES=0

usage() {
  cat <<'USAGE'
Usage: scripts/demo_debit_card_full_lifecycle.sh [options]

  --preflight-only            Run stage A (infrastructure preflight) and stop.
  --resume-from <stage>       Start from <stage> instead of `preflight`.
  --approve-promotion         Explicit manual approval; required for `promote`.
  --cleanup                   Tear down ONLY the isolated aidp-demo stack
                              (containers, network and its own volumes).
  --yes                       Do not prompt for confirmation during --cleanup.
  -h, --help                  Show this help.

Stages, in order:
  preflight generate generate_retry verify split_preview train
  evaluate_candidate diagnostic approval promote score score_retry
  labels labels_retry live_evaluate audit
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --preflight-only) OPT_PREFLIGHT_ONLY=1; shift ;;
    --resume-from)
      [ $# -ge 2 ] || { echo "Error: --resume-from requires a stage name" >&2; exit 2; }
      OPT_RESUME_FROM="$2"; shift 2 ;;
    --resume-from=*) OPT_RESUME_FROM="${1#--resume-from=}"; shift ;;
    --approve-promotion) OPT_APPROVE_PROMOTION=1; shift ;;
    --cleanup) OPT_CLEANUP=1; shift ;;
    --yes|-y) OPT_ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -n "$OPT_RESUME_FROM" ]; then
  _stage_known=0
  for _s in $DEMO_STAGES; do
    [ "$_s" = "$OPT_RESUME_FROM" ] && _stage_known=1
  done
  if [ "$_stage_known" -ne 1 ]; then
    echo "Error: unknown stage '$OPT_RESUME_FROM'. Valid stages: $DEMO_STAGES" >&2
    exit 2
  fi
fi

if [ "$OPT_PREFLIGHT_ONLY" -eq 1 ] && [ -n "$OPT_RESUME_FROM" ] && [ "$OPT_RESUME_FROM" != "preflight" ]; then
  echo "Error: --preflight-only cannot be combined with --resume-from $OPT_RESUME_FROM" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# 3. Logging (stdout and stderr kept in SEPARATE timestamped files)
# ---------------------------------------------------------------------------

DEMO_RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
DEMO_LOG_DIR="$DEMO_WORK_DIR/logs/$DEMO_RUN_TS"
DEMO_REPORT_DIR="$DEMO_WORK_DIR/reports/$DEMO_RUN_TS"
mkdir -p "$DEMO_LOG_DIR" "$DEMO_REPORT_DIR"

DEMO_STDOUT_LOG="$DEMO_LOG_DIR/demo.stdout.log"
DEMO_STDERR_LOG="$DEMO_LOG_DIR/demo.stderr.log"
DEMO_COMMAND_LOG="$DEMO_LOG_DIR/commands.log"
: >"$DEMO_STDOUT_LOG"
: >"$DEMO_STDERR_LOG"
: >"$DEMO_COMMAND_LOG"

exec > >(tee -a "$DEMO_STDOUT_LOG") 2> >(tee -a "$DEMO_STDERR_LOG" >&2)

info()  { printf '[INFO] %s\n' "$*"; }
warn()  { printf '[WARN] %s\n' "$*" >&2; }
pass()  { printf '[PASS] %s\n' "$*"; }
note()  { printf '       %s\n' "$*"; }

die() {
  printf '[FAIL] %s\n' "$*" >&2
  printf '\n[RESULT] DEMO FAILED at stage: %s\n' "${CURRENT_STAGE:-<none>}" >&2
  printf '         stdout log: %s\n' "$DEMO_STDOUT_LOG" >&2
  printf '         stderr log: %s\n' "$DEMO_STDERR_LOG" >&2
  exit 1
}

CURRENT_STAGE=""
stage_banner() {
  CURRENT_STAGE="$1"
  printf '\n'
  printf '===============================================================================\n'
  printf ' STAGE %s -- %s\n' "$1" "$2"
  printf '===============================================================================\n'
}

# ---------------------------------------------------------------------------
# 4. Repository / toolchain guards
# ---------------------------------------------------------------------------

require_correct_repository() {
  local marker
  for marker in \
    "src/cli/__main__.py" \
    "src/fraud_intel/cli_data_access.py" \
    "src/fraud_intel/registry.py" \
    "config/fraud_intel/rules_debit_card.yaml" \
    "infrastructure/postgres/lib/schema.sql" \
    "docker-compose.demo.yml" \
    ".env.demo.example"
  do
    [ -f "$REPO_ROOT/$marker" ] || die "not the local-aidp-poc repository: missing $marker under $REPO_ROOT"
  done

  command -v git >/dev/null 2>&1 || die "git is required but not installed"
  local toplevel
  toplevel="$(cd "$REPO_ROOT" && git rev-parse --show-toplevel 2>/dev/null || true)"
  [ -n "$toplevel" ] || die "$REPO_ROOT is not inside a git work tree"
  [ "$(cd "$toplevel" && pwd -P)" = "$REPO_ROOT" ] || \
    die "script must run from the repository root ($toplevel != $REPO_ROOT)"

  command -v docker >/dev/null 2>&1 || die "docker is required but not installed"
  docker compose version >/dev/null 2>&1 || die "docker compose v2 is required but not available"
  pass "repository identity confirmed: $REPO_ROOT"
}

report_git_state() {
  local branch dirty
  branch="$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD)"
  dirty="$(cd "$REPO_ROOT" && git status --porcelain)"
  GIT_BRANCH="$branch"
  if [ -z "$dirty" ]; then
    GIT_WORKTREE_STATE="clean"
  else
    GIT_WORKTREE_STATE="dirty"
  fi
  info "git branch: $branch"
  info "git working tree: $GIT_WORKTREE_STATE"
  if [ "$GIT_WORKTREE_STATE" = "dirty" ]; then
    printf '%s\n' "$dirty" | sed 's/^/       /'
    warn "working tree is dirty -- the demo still runs, but recorded provenance (git_sha) will not describe a committed state"
  fi
  printf '%s\n' "$dirty" >"$DEMO_REPORT_DIR/git_status.txt"
}

require_venv() {
  if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" = "$REPO_ROOT/.venv" ]; then
    info "virtualenv already active: $VIRTUAL_ENV"
  elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.venv/bin/activate"
    info "activated virtualenv: $REPO_ROOT/.venv"
  else
    die "no virtualenv at $REPO_ROOT/.venv -- create it and install requirements.txt first"
  fi

  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  [ -x "$PYTHON_BIN" ] || die "expected interpreter $PYTHON_BIN is missing or not executable"
  pass "python: $("$PYTHON_BIN" -V 2>&1) ($PYTHON_BIN)"
}

# ---------------------------------------------------------------------------
# 5. Isolated environment bootstrap
# ---------------------------------------------------------------------------

ensure_demo_credentials() {
  # Credentials are NEVER stored in the committed .env.demo.example. They are
  # generated once, locally, into the git-ignored .env.demo.local.
  if [ -f "$DEMO_ENV_LOCAL" ]; then
    info "using existing local demo credentials: $DEMO_ENV_LOCAL"
    return 0
  fi
  info "generating demo-only credentials into $DEMO_ENV_LOCAL (git-ignored, mode 600)"
  local pg_pw minio_pw
  pg_pw="$("$PYTHON_BIN" -c 'import secrets;print(secrets.token_hex(24))')"
  minio_pw="$("$PYTHON_BIN" -c 'import secrets;print(secrets.token_hex(24))')"
  ( umask 077; cat >"$DEMO_ENV_LOCAL" <<LOCALENV
# Locally generated credentials for the ISOLATED aidp-demo stack ONLY.
# Generated by scripts/demo_debit_card_full_lifecycle.sh. Never commit this
# file (.gitignore covers it). Deleting it while the demo Postgres/MinIO
# volumes still exist will break authentication -- run --cleanup first.
POSTGRES_PASSWORD=$pg_pw
MINIO_SECRET_KEY=$minio_pw
LOCALENV
  )
  chmod 600 "$DEMO_ENV_LOCAL"
  pass "demo credentials created"
}

load_demo_env() {
  [ -f "$DEMO_ENV_EXAMPLE" ] || die "missing $DEMO_ENV_EXAMPLE"
  if [ ! -f "$DEMO_ENV_LOCAL" ] && [ "$OPT_CLEANUP" -ne 1 ]; then
    die "missing $DEMO_ENV_LOCAL"
  fi

  # Exported into this process only. .env on disk is never read, written or
  # modified by this script; pydantic-settings gives these precedence.
  set -a
  # shellcheck disable=SC1090
  . "$DEMO_ENV_EXAMPLE"
  if [ -f "$DEMO_ENV_LOCAL" ]; then
    # shellcheck disable=SC1090
    . "$DEMO_ENV_LOCAL"
  fi
  set +a

  DEMO_MLFLOW_URL="$MLFLOW_TRACKING_URI"
  DEMO_MINIO_API_URL="http://$MINIO_ENDPOINT"
  DEMO_MINIO_CONSOLE_URL="http://$MINIO_CONSOLE_ENDPOINT"
  pass "isolated environment loaded (.env untouched)"
}

assert_env_isolation() {
  # --- the demo must target aidp_demo, and only aidp_demo ---
  [ "$POSTGRES_DB" = "$DEMO_DATABASE" ] || die "POSTGRES_DB is '$POSTGRES_DB', expected '$DEMO_DATABASE'"
  [ "$POSTGRES_TEST_DB" = "$DEMO_DATABASE" ] || die "POSTGRES_TEST_DB is '$POSTGRES_TEST_DB', expected '$DEMO_DATABASE' (src/dashboard/data.py resolves the dashboard database from this setting)"

  local forbidden
  for forbidden in $FORBIDDEN_DATABASES; do
    [ "$POSTGRES_DB" != "$forbidden" ] || die "POSTGRES_DB resolves to the forbidden database '$forbidden'"
    [ "$POSTGRES_TEST_DB" != "$forbidden" ] || die "POSTGRES_TEST_DB resolves to the forbidden database '$forbidden'"
  done

  # --- the demo must not reuse the shared stack's endpoints ---
  # The shared stack's values are read (read-only) purely to prove we differ.
  local shared_env=""
  if [ -f "$REPO_ROOT/.env" ]; then
    shared_env="$REPO_ROOT/.env"
  elif [ -f "$REPO_ROOT/.env.example" ]; then
    shared_env="$REPO_ROOT/.env.example"
  fi
  if [ -n "$shared_env" ]; then
    local shared_pg_port shared_minio shared_mlflow
    shared_pg_port="$(grep -E '^POSTGRES_PORT=' "$shared_env" | tail -1 | cut -d= -f2- || true)"
    shared_minio="$(grep -E '^MINIO_ENDPOINT=' "$shared_env" | tail -1 | cut -d= -f2- || true)"
    shared_mlflow="$(grep -E '^MLFLOW_TRACKING_URI=' "$shared_env" | tail -1 | cut -d= -f2- || true)"
    [ -z "$shared_pg_port" ] || [ "$POSTGRES_PORT" != "$shared_pg_port" ] || \
      die "demo POSTGRES_PORT ($POSTGRES_PORT) equals the shared stack's ($shared_pg_port)"
    [ -z "$shared_minio" ] || [ "$MINIO_ENDPOINT" != "$shared_minio" ] || \
      die "demo MINIO_ENDPOINT ($MINIO_ENDPOINT) equals the shared stack's ($shared_minio)"
    [ -z "$shared_mlflow" ] || [ "$MLFLOW_TRACKING_URI" != "$shared_mlflow" ] || \
      die "demo MLFLOW_TRACKING_URI ($MLFLOW_TRACKING_URI) equals the shared stack's ($shared_mlflow)"
    note "shared stack endpoints read read-only from $shared_env and confirmed different"
  fi

  pass "environment isolation asserted (db=$POSTGRES_DB, pg=$POSTGRES_HOST:$POSTGRES_PORT, minio=$MINIO_ENDPOINT, mlflow=$MLFLOW_TRACKING_URI)"
}

# ---------------------------------------------------------------------------
# 6. Command runners
# ---------------------------------------------------------------------------

record_command() {
  printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$DEMO_COMMAND_LOG"
}

# Monotonic prefix so a tag used more than once (label_digest, evidence_digest,
# the repeated population snapshots) never overwrites an earlier invocation's
# stdout/stderr evidence.
#
# The counter is deliberately published through the global NEXT_LOG_BASE rather
# than printed on stdout: a `base="$(next_log_base "$tag")"` call would run this
# function in a COMMAND-SUBSTITUTION SUBSHELL, so the DEMO_STEP_SEQ increment
# would be discarded and every basename would come back as 001_<tag> -- exactly
# the collision this counter exists to prevent. Callers must therefore use:
#
#     next_log_base "$tag"
#     base="$NEXT_LOG_BASE"
#
# and must never declare NEXT_LOG_BASE `local`, which would make bash's dynamic
# scoping capture the assignment in the caller's own frame.
DEMO_STEP_SEQ=0
NEXT_LOG_BASE=""
next_log_base() {
  DEMO_STEP_SEQ=$((DEMO_STEP_SEQ + 1))
  NEXT_LOG_BASE="$(printf '%s/%03d_%s' "$DEMO_LOG_DIR" "$DEMO_STEP_SEQ" "$1")"
}

# demo_compose <docker compose args...>
# ALWAYS scoped to the isolated project/file/env files. There is no code path
# in this script that invokes `docker compose` without these flags.
demo_compose() {
  record_command "docker compose -p $DEMO_COMPOSE_PROJECT -f $DEMO_COMPOSE_FILE $*"
  local env_args
  env_args="--env-file $DEMO_ENV_EXAMPLE"
  if [ -f "$DEMO_ENV_LOCAL" ]; then
    env_args="$env_args --env-file $DEMO_ENV_LOCAL"
  else
    # Only reachable during --cleanup after the credentials file was removed.
    # Interpolation still needs the names to be defined; the values are never
    # used to authenticate against anything that is about to be deleted.
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-unused-for-teardown}"
    export MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-unused-for-teardown}"
  fi
  # shellcheck disable=SC2086
  ( cd "$REPO_ROOT" && docker compose \
      -p "$DEMO_COMPOSE_PROJECT" \
      -f "$DEMO_COMPOSE_FILE" \
      $env_args \
      "$@" )
}

# run_cli <tag> <cli args...> -- always appends nothing; caller supplies --json.
# Captures stdout/stderr into SEPARATE per-stage log files and exports
# CLI_JSON_FILE for the verifier that follows.
run_cli() {
  local tag="$1"; shift
  local base
  next_log_base "$tag"
  base="$NEXT_LOG_BASE"
  local out="${base}.stdout.log"
  local err="${base}.stderr.log"
  local rc=0
  record_command "python -m src.cli $*"
  ( cd "$REPO_ROOT" && "$PYTHON_BIN" -m src.cli "$@" ) >"$out" 2>"$err" || rc=$?
  CLI_JSON_FILE="$out"
  CLI_STDERR_FILE="$err"
  return "$rc"
}

# run_python <tag> -- reads a python program on stdin, runs it with the repo
# root on sys.path, stdout/stderr to separate per-stage log files.
run_python() {
  local tag="$1"
  local base
  next_log_base "$tag"
  base="$NEXT_LOG_BASE"
  local out="${base}.stdout.log"
  local err="${base}.stderr.log"
  local rc=0
  record_command "python - (<$tag>)"
  ( cd "$REPO_ROOT" && "$PYTHON_BIN" - ) >"$out" 2>"$err" || rc=$?
  PY_JSON_FILE="$out"
  PY_STDERR_FILE="$err"
  return "$rc"
}

# assert_json <file> -- stdlib json only; no jq anywhere in this script.
assert_json() {
  local file="$1"
  [ -s "$file" ] || die "expected JSON output in $file but the file is empty"
  ( cd "$REPO_ROOT" && "$PYTHON_BIN" -c 'import json,sys; json.load(open(sys.argv[1]))' "$file" ) \
    >/dev/null 2>&1 || die "invalid JSON in $file"
}

# json_get <file> <dotted.path> -- prints a scalar, or compact JSON for
# containers. Fails loudly on a missing key rather than printing nothing.
json_get() {
  ( cd "$REPO_ROOT" && "$PYTHON_BIN" - "$1" "$2" <<'PYJSON'
import json, sys
path = sys.argv[2]
cur = json.load(open(sys.argv[1]))
for part in path.split("."):
    if isinstance(cur, list):
        cur = cur[int(part)]
    else:
        if part not in cur:
            sys.stderr.write("missing key %r at path %r\n" % (part, path))
            raise SystemExit(3)
        cur = cur[part]
if isinstance(cur, (dict, list)):
    print(json.dumps(cur, sort_keys=True))
elif cur is None:
    print("null")
elif isinstance(cur, bool):
    print("true" if cur else "false")
else:
    print(cur)
PYJSON
  )
}

expect_eq() {
  # expect_eq <label> <actual> <expected>
  if [ "$2" = "$3" ]; then
    pass "$1 = $2"
  else
    die "$1 = '$2', expected '$3'"
  fi
}

require_cli_ok() {
  # require_cli_ok <rc> <label>
  local rc="$1" label="$2"
  if [ "$rc" -ne 0 ]; then
    printf '[FAIL] %s exited %s\n' "$label" "$rc" >&2
    printf '--- stdout (%s) ---\n' "$CLI_JSON_FILE" >&2
    sed 's/^/       /' "$CLI_JSON_FILE" >&2 || true
    printf '--- stderr (%s) ---\n' "$CLI_STDERR_FILE" >&2
    tail -40 "$CLI_STDERR_FILE" >&2 || true
    die "$label failed"
  fi
  assert_json "$CLI_JSON_FILE"
  # emit_error() writes {"error": ..., "message": ...} on stdout in --json
  # mode; a zero exit with an error object would be a contract violation, so
  # check for it explicitly rather than trusting the exit code alone.
  if ( cd "$REPO_ROOT" && "$PYTHON_BIN" -c \
        'import json,sys; sys.exit(0 if "error" in json.load(open(sys.argv[1])) else 1)' \
        "$CLI_JSON_FILE" ) >/dev/null 2>&1; then
    printf '[FAIL] %s returned an error object:\n' "$label" >&2
    sed 's/^/       /' "$CLI_JSON_FILE" >&2
    die "$label failed"
  fi
}

require_python_ok() {
  # require_python_ok <rc> <label>
  local rc="$1" label="$2"
  if [ "$rc" -ne 0 ]; then
    printf '[FAIL] %s exited %s\n' "$label" "$rc" >&2
    printf '--- stderr (%s) ---\n' "$PY_STDERR_FILE" >&2
    tail -60 "$PY_STDERR_FILE" >&2 || true
    die "$label failed"
  fi
  assert_json "$PY_JSON_FILE"
}

# ---------------------------------------------------------------------------
# 7. Runtime state (never hardcode identifiers produced at runtime)
# ---------------------------------------------------------------------------

state_put() {
  # state_put <key> <value>
  ( cd "$REPO_ROOT" && STATE_FILE="$DEMO_STATE_FILE" SK="$1" SV="$2" "$PYTHON_BIN" - <<'PYSTATE'
import json, os, pathlib
path = pathlib.Path(os.environ["STATE_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)
doc = {}
if path.exists():
    try:
        doc = json.loads(path.read_text())
    except ValueError:
        doc = {}
doc[os.environ["SK"]] = os.environ["SV"]
path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
PYSTATE
  )
}

state_get() {
  # state_get <key>; prints the value or an empty string
  ( cd "$REPO_ROOT" && STATE_FILE="$DEMO_STATE_FILE" SK="$1" "$PYTHON_BIN" - <<'PYSTATE'
import json, os, pathlib
path = pathlib.Path(os.environ["STATE_FILE"])
if not path.exists():
    raise SystemExit(0)
try:
    doc = json.loads(path.read_text())
except ValueError:
    raise SystemExit(0)
value = doc.get(os.environ["SK"])
if value is not None:
    print(value)
PYSTATE
  )
}

# Deterministic generation identity, recomputed from the REAL production
# function (src.fraud_intel.cli_data_access._generation_identity) rather than
# hardcoded or remembered -- this is what makes --resume-from safe even when
# the state file is absent.
derive_generation_identity() {
  local rc=0
  run_python "derive_generation_identity" <<'PYGEN' || rc=$?
import json
from datetime import date
from src.fraud_intel.cli_data_access import _generation_identity
import os

run_id, dataset_version = _generation_identity(
    channel=os.environ["DEMO_CHANNEL"],
    count=int(os.environ["DEMO_COUNT"]),
    seed=int(os.environ["DEMO_SEED"]),
    reference_date=date.fromisoformat(os.environ["DEMO_REFERENCE_DATE"]),
)
print(json.dumps({"generation_run_id": run_id, "dataset_version": dataset_version}))
PYGEN
  require_python_ok "$rc" "derive_generation_identity"
  EXPECTED_GENERATION_RUN_ID="$(json_get "$PY_JSON_FILE" generation_run_id)"
  EXPECTED_DATASET_VERSION="$(json_get "$PY_JSON_FILE" dataset_version)"
}

resolve_generation_identity() {
  GENERATION_RUN_ID="$(state_get generation_run_id)"
  DATASET_VERSION="$(state_get dataset_version)"
  if [ -z "$GENERATION_RUN_ID" ] || [ -z "$DATASET_VERSION" ]; then
    info "no recorded generation identity -- re-deriving from the production generation-identity function"
    derive_generation_identity
    GENERATION_RUN_ID="$EXPECTED_GENERATION_RUN_ID"
    DATASET_VERSION="$EXPECTED_DATASET_VERSION"
  fi
  [ -n "$GENERATION_RUN_ID" ] || die "could not resolve generation_run_id"
  info "generation_run_id=$GENERATION_RUN_ID dataset_version=$DATASET_VERSION"
}

resolve_bundle_identity() {
  BUNDLE_ID="$(state_get bundle_id)"
  BUNDLE_VERSION="$(state_get bundle_version)"
  if [ -z "$BUNDLE_ID" ] || [ -z "$BUNDLE_VERSION" ]; then
    info "no recorded bundle identity -- reading it back from channel_model_bundles in $DEMO_DATABASE"
    local rc=0
    run_python "resolve_bundle_identity" <<'PYBUNDLE' || rc=$?
import json, os
import psycopg2.extras
from src.common.db import get_connection

channel = os.environ["DEMO_CHANNEL"]
conn = get_connection(os.environ["DEMO_DATABASE"])
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT bundle_id, bundle_version, status, training_run_id "
                "FROM channel_model_bundles WHERE channel = %s ORDER BY bundle_version",
                (channel,),
            )
            rows = [dict(r) for r in cur.fetchall()]
finally:
    conn.close()

if len(rows) != 1:
    raise SystemExit(
        "expected exactly 1 bundle row for channel %r in this demo database, found %d -- "
        "refusing to guess which bundle the demo meant" % (channel, len(rows))
    )
print(json.dumps(rows[0], default=str))
PYBUNDLE
    require_python_ok "$rc" "resolve_bundle_identity"
    BUNDLE_ID="$(json_get "$PY_JSON_FILE" bundle_id)"
    BUNDLE_VERSION="$(json_get "$PY_JSON_FILE" bundle_version)"
    state_put bundle_id "$BUNDLE_ID"
    state_put bundle_version "$BUNDLE_VERSION"
  fi
  [ -n "$BUNDLE_ID" ] || die "could not resolve bundle_id"
  info "bundle_id=$BUNDLE_ID bundle_version=$BUNDLE_VERSION"
}

# ---------------------------------------------------------------------------
# 8. Database-target assertions (run before EVERY write-producing stage)
# ---------------------------------------------------------------------------

assert_demo_database() {
  # assert_demo_database <tag>
  local rc=0
  run_python "assert_db_$1" <<'PYDB' || rc=$?
import json, os
from src.common.config import get_settings
from src.common.db import get_connection

settings = get_settings()
demo_db = os.environ["DEMO_DATABASE"]
forbidden = set(os.environ["FORBIDDEN_DATABASES"].split())

problems = []
if settings.postgres_db != demo_db:
    problems.append("settings.postgres_db=%r" % settings.postgres_db)
if settings.postgres_test_db != demo_db:
    problems.append("settings.postgres_test_db=%r" % settings.postgres_test_db)
if str(settings.postgres_port) != os.environ["POSTGRES_PORT"]:
    problems.append("settings.postgres_port=%r" % settings.postgres_port)

# The application's OWN default connection factory -- get_connection() with
# no argument resolves through settings.postgres_db, exactly as every
# production code path does.
conn = get_connection()
try:
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user, inet_server_port()")
            current_db, current_user, server_port = cur.fetchone()
            cur.execute("SELECT datname FROM pg_database ORDER BY datname")
            datnames = [r[0] for r in cur.fetchall()]
finally:
    conn.close()

if current_db != demo_db:
    problems.append("current_database()=%r" % current_db)
present_forbidden = sorted(forbidden.intersection(datnames))
if present_forbidden:
    problems.append("forbidden database(s) present in this cluster: %r" % present_forbidden)

print(json.dumps({
    "current_database": current_db,
    "current_user": current_user,
    "server_port": server_port,
    "databases": datnames,
    "settings_postgres_db": settings.postgres_db,
    "settings_postgres_test_db": settings.postgres_test_db,
    "forbidden_databases_present": present_forbidden,
    "problems": problems,
    "ok": not problems,
}))
if problems:
    raise SystemExit("database isolation assertion failed: %s" % "; ".join(problems))
PYDB
  require_python_ok "$rc" "assert_demo_database($1)"
  local current_db
  current_db="$(json_get "$PY_JSON_FILE" current_database)"
  expect_eq "current_database() before '$1'" "$current_db" "$DEMO_DATABASE"
  note "cluster databases: $(json_get "$PY_JSON_FILE" databases)"
}

prove_no_forbidden_contact() {
  # Structural proof, in four independent parts:
  #   1. the effective Settings resolve only to aidp_demo;
  #   2. the ISOLATED cluster contains no database named aidp or aidp_test,
  #      so nothing running against it can reach one;
  #   3. every --database argument this run issued was aidp_demo;
  #   4. the demo endpoints differ from the shared stack's (asserted in
  #      assert_env_isolation()).
  assert_demo_database "no_forbidden_contact"

  local bad
  bad="$(grep -o -- '--database [A-Za-z0-9_]*' "$DEMO_COMMAND_LOG" 2>/dev/null | awk '{print $2}' | sort -u | grep -v "^${DEMO_DATABASE}$" || true)"
  if [ -n "$bad" ]; then
    die "command log contains a non-demo --database target: $bad ($DEMO_COMMAND_LOG)"
  fi

  local forbidden
  for forbidden in $FORBIDDEN_DATABASES; do
    if grep -qE -- "--database[= ]${forbidden}\b" "$DEMO_COMMAND_LOG" 2>/dev/null; then
      die "command log references forbidden database '$forbidden'"
    fi
  done
  pass "aidp and aidp_test were not contacted (settings, cluster catalog and command log all agree)"
}

# ---------------------------------------------------------------------------
# 9. Isolated infrastructure lifecycle
# ---------------------------------------------------------------------------

start_demo_infrastructure() {
  info "starting the isolated '$DEMO_COMPOSE_PROJECT' stack (postgres, minio, mlflow) -- the shared aidp-poc stack is not touched"
  demo_compose up -d >"$DEMO_LOG_DIR/compose_up.stdout.log" 2>"$DEMO_LOG_DIR/compose_up.stderr.log" || {
    tail -40 "$DEMO_LOG_DIR/compose_up.stderr.log" >&2 || true
    die "docker compose up failed for project $DEMO_COMPOSE_PROJECT"
  }

  info "waiting for demo services to report healthy..."
  local i unhealthy
  for i in $(seq 1 90); do
    unhealthy="$(demo_compose ps --format '{{.Service}} {{.Health}}' 2>/dev/null \
      | awk '$2 != "" && $2 != "healthy" {print $1}' || true)"
    if [ -z "$unhealthy" ]; then
      pass "all demo services healthy"
      demo_compose ps >"$DEMO_REPORT_DIR/compose_ps.txt" 2>&1 || true
      return 0
    fi
    sleep 2
  done
  demo_compose ps >&2 || true
  die "timed out waiting for demo services to become healthy: $unhealthy"
}

assert_only_demo_containers() {
  local names
  names="$(demo_compose ps --format '{{.Name}}' 2>/dev/null || true)"
  [ -n "$names" ] || die "no containers found for project $DEMO_COMPOSE_PROJECT"
  local n
  for n in $names; do
    case "$n" in
      "${DEMO_CONTAINER_PREFIX}"*) : ;;
      *) die "project $DEMO_COMPOSE_PROJECT unexpectedly owns container '$n' (expected prefix '${DEMO_CONTAINER_PREFIX}')" ;;
    esac
  done
  pass "isolated stack owns only ${DEMO_CONTAINER_PREFIX}* containers: $(printf '%s ' $names)"
}

check_http_health() {
  # check_http_health <label> <url>
  command -v curl >/dev/null 2>&1 || die "curl is required for health checks"
  if curl -fsS --max-time 10 "$2" >/dev/null 2>&1; then
    pass "$1 healthy ($2)"
  else
    die "$1 not healthy at $2"
  fi
}

# ---------------------------------------------------------------------------
# 10. STAGE A -- infrastructure preflight
# ---------------------------------------------------------------------------

stage_preflight() {
  stage_banner "A" "Infrastructure preflight"

  require_correct_repository
  report_git_state
  require_venv
  ensure_demo_credentials
  load_demo_env
  assert_env_isolation

  start_demo_infrastructure
  assert_only_demo_containers
  check_http_health "MinIO" "$DEMO_MINIO_API_URL/minio/health/live"
  check_http_health "MLflow" "$DEMO_MLFLOW_URL/health"

  assert_demo_database "preflight"

  if [ "$DEMO_REQUIRE_EMPTY" = "1" ]; then
    info "verifying the demo database is empty for channel '$DEMO_CHANNEL' and that MLflow holds no versions for it"
  else
    info "resume mode: recording the current channel population instead of asserting emptiness"
  fi
  local rc=0
  run_python "preflight_empty" <<'PYPRE' || rc=$?
import json, os
import psycopg2.extras

from src.common.db import get_connection
from src.fraud_intel.models.mlflow_naming import MODEL_COMPONENTS, registered_model_name
from src.fraud_intel.models.training import _experiment_name

channel = os.environ["DEMO_CHANNEL"]
database = os.environ["DEMO_DATABASE"]

counts = {}
conn = get_connection(database)
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT count(*) AS n FROM channel_events WHERE channel = %s", (channel,))
            counts["channel_events"] = cur.fetchone()["n"]
            cur.execute(
                "SELECT count(*) AS n FROM source_alerts sa JOIN channel_events ce ON ce.event_id = sa.event_id "
                "WHERE ce.channel = %s", (channel,))
            counts["source_alerts"] = cur.fetchone()["n"]
            cur.execute(
                "SELECT count(*) AS n FROM synthetic_event_labels sel JOIN channel_events ce "
                "ON ce.event_id = sel.event_id WHERE ce.channel = %s", (channel,))
            counts["synthetic_event_labels"] = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM channel_model_bundles WHERE channel = %s", (channel,))
            counts["channel_model_bundles"] = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM fraud_alerts WHERE channel = %s", (channel,))
            counts["fraud_alerts"] = cur.fetchone()["n"]
            cur.execute(
                "SELECT count(*) AS n FROM alert_evidence ae JOIN fraud_alerts fa ON fa.alert_id = ae.alert_id "
                "WHERE fa.channel = %s", (channel,))
            counts["alert_evidence"] = cur.fetchone()["n"]
            cur.execute(
                "SELECT count(*) AS n FROM label_assessments la JOIN fraud_alerts fa ON fa.alert_id = la.alert_id "
                "WHERE fa.channel = %s", (channel,))
            counts["label_assessments"] = cur.fetchone()["n"]
            cur.execute(
                "SELECT count(*) AS n FROM analyst_dispositions ad JOIN fraud_alerts fa ON fa.alert_id = ad.alert_id "
                "WHERE fa.channel = %s", (channel,))
            counts["analyst_dispositions"] = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM pipeline_runs")
            counts["pipeline_runs"] = cur.fetchone()["n"]
finally:
    conn.close()

non_empty = sorted(k for k, v in counts.items() if v)

# MLflow: the ISOLATED tracking server must hold no experiment and no
# registered-model version for this channel yet. Metadata reads only --
# nothing is created, logged, registered or aliased here.
import mlflow
from mlflow.exceptions import MlflowException
from src.common.mlflow_setup import configure_mlflow

configure_mlflow()
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(_experiment_name(channel))
mlflow_state = {
    "tracking_uri": mlflow.get_tracking_uri(),
    "experiment": None if experiment is None else experiment.experiment_id,
    "registered_models": {},
}
for component in MODEL_COMPONENTS:
    name = registered_model_name(channel, component)
    try:
        versions = client.search_model_versions("name='%s'" % name)
        mlflow_state["registered_models"][name] = len(versions)
    except MlflowException:
        mlflow_state["registered_models"][name] = 0

existing_versions = sorted(n for n, v in mlflow_state["registered_models"].items() if v)

print(json.dumps({
    "counts": counts,
    "non_empty": non_empty,
    "mlflow": mlflow_state,
    "existing_model_versions": existing_versions,
}, sort_keys=True))

# In resume mode the channel is expected to be populated already, so the
# counts are reported as evidence rather than asserted to be zero.
if os.environ.get("DEMO_REQUIRE_EMPTY", "1") == "1":
    problems = []
    if non_empty:
        problems.append("non-empty tables for channel %r: %s" % (channel, non_empty))
    if mlflow_state["experiment"] is not None:
        problems.append("MLflow experiment %r already exists" % _experiment_name(channel))
    if existing_versions:
        problems.append("registered model versions already exist: %s" % existing_versions)
    if problems:
        raise SystemExit("preflight emptiness check failed: %s" % "; ".join(problems))
PYPRE
  require_python_ok "$rc" "preflight population check"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/preflight_population.json"
  if [ "$DEMO_REQUIRE_EMPTY" = "1" ]; then
    pass "channel '$DEMO_CHANNEL' has zero events, source alerts, labels, bundles, alerts, evidence, assessments and MLflow versions"
  else
    pass "current channel population recorded (resume mode)"
  fi
  note "table counts: $(json_get "$PY_JSON_FILE" counts)"
  note "MLflow tracking URI: $(json_get "$PY_JSON_FILE" mlflow.tracking_uri)"

  info "computing the EXPECTED population from the real deterministic generator (in memory, no database write)"
  rc=0
  run_python "preflight_generator_expectation" <<'PYEXP' || rc=$?
import json, os
from collections import Counter
from datetime import date

from src.fraud_intel.cli_data_access import _GENERATORS, _generation_identity
from src.fraud_intel.generator.customers import generate_customers
from src.fraud_intel.scoring.dispatch import _MAX_PENDING_ALERTS_PER_RUN

channel = os.environ["DEMO_CHANNEL"]
count = int(os.environ["DEMO_COUNT"])
seed = int(os.environ["DEMO_SEED"])
reference_date = date.fromisoformat(os.environ["DEMO_REFERENCE_DATE"])

# Mirrors src.fraud_intel.cli_data_access.generate_and_write() exactly.
generation_run_id, dataset_version = _generation_identity(
    channel=channel, count=count, seed=seed, reference_date=reference_date
)
customers = generate_customers(n=max(count // 5, 1), seed=seed, reference_date=reference_date)
results = _GENERATORS[channel](
    seed=seed, n=count, reference_date=reference_date, customers=customers,
    generation_run_id=generation_run_id, dataset_version=dataset_version,
)

source_alert_count = sum(1 for _, sa, _ in results if sa is not None)
fraud_count = sum(1 for _, _, label in results if label.synthetic_scenario_label)
scenario_types = Counter(label.scenario_type for _, _, label in results)
alerted_fraud = sum(1 for _, sa, label in results if sa is not None and label.synthetic_scenario_label)
alerted_legit = source_alert_count - alerted_fraud
timestamps = sorted(event.event_timestamp for event, _, _ in results)

payload = {
    "generation_run_id": generation_run_id,
    "dataset_version": dataset_version,
    "event_count": len(results),
    "source_alert_count": source_alert_count,
    "label_count": len(results),
    "fraud_count": fraud_count,
    "legitimate_count": len(results) - fraud_count,
    "alerted_fraud_count": alerted_fraud,
    "alerted_legitimate_count": alerted_legit,
    "scenario_types": dict(sorted(scenario_types.items())),
    "distinct_event_ids": len({str(e.event_id) for e, _, _ in results}),
    "distinct_source_alert_ids": len({str(sa.source_alert_id) for _, sa, _ in results if sa is not None}),
    "distinct_customers": len({e.customer_id for e, _, _ in results}),
    "event_timestamp_min": timestamps[0].isoformat(),
    "event_timestamp_max": timestamps[-1].isoformat(),
    "max_pending_alerts_per_scoring_run": _MAX_PENDING_ALERTS_PER_RUN,
}
print(json.dumps(payload, sort_keys=True))

problems = []
if len(results) != count:
    problems.append("generator produced %d events for --count %d" % (len(results), count))
if payload["distinct_event_ids"] != len(results):
    problems.append("generator produced duplicate event_ids")
if payload["distinct_source_alert_ids"] != source_alert_count:
    problems.append("generator produced duplicate source_alert_ids")
if source_alert_count > _MAX_PENDING_ALERTS_PER_RUN:
    problems.append(
        "the deterministic source-alert population is %d, which exceeds "
        "src.fraud_intel.scoring.dispatch._MAX_PENDING_ALERTS_PER_RUN (%d). A single "
        "`fraud-intel score` run would process only the first %d alerts, so the demo's "
        "'pending_count == source-alert population' and 'retry pending_count == 0' gates "
        "cannot hold. Reduce --count, or raise that constant in source (a code change "
        "this demo deliberately does not make)."
        % (source_alert_count, _MAX_PENDING_ALERTS_PER_RUN, _MAX_PENDING_ALERTS_PER_RUN)
    )
if problems:
    raise SystemExit("generator expectation check failed: %s" % "; ".join(problems))
PYEXP
  require_python_ok "$rc" "generator expectation"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/expected_population.json"

  EXPECTED_SOURCE_ALERT_COUNT="$(json_get "$PY_JSON_FILE" source_alert_count)"
  EXPECTED_FRAUD_COUNT="$(json_get "$PY_JSON_FILE" fraud_count)"
  EXPECTED_ALERTED_FRAUD="$(json_get "$PY_JSON_FILE" alerted_fraud_count)"
  EXPECTED_ALERTED_LEGIT="$(json_get "$PY_JSON_FILE" alerted_legitimate_count)"
  EXPECTED_GENERATION_RUN_ID="$(json_get "$PY_JSON_FILE" generation_run_id)"
  EXPECTED_DATASET_VERSION="$(json_get "$PY_JSON_FILE" dataset_version)"

  export EXPECTED_SOURCE_ALERT_COUNT EXPECTED_FRAUD_COUNT EXPECTED_ALERTED_FRAUD \
         EXPECTED_ALERTED_LEGIT EXPECTED_GENERATION_RUN_ID EXPECTED_DATASET_VERSION

  state_put expected_source_alert_count "$EXPECTED_SOURCE_ALERT_COUNT"
  state_put expected_fraud_count "$EXPECTED_FRAUD_COUNT"
  state_put expected_alerted_fraud_count "$EXPECTED_ALERTED_FRAUD"
  state_put expected_alerted_legitimate_count "$EXPECTED_ALERTED_LEGIT"

  pass "deterministic generator expectation computed (nothing persisted)"
  note "events=$DEMO_COUNT source_alerts=$EXPECTED_SOURCE_ALERT_COUNT fraud=$EXPECTED_FRAUD_COUNT (alerted fraud=$EXPECTED_ALERTED_FRAUD, alerted legitimate=$EXPECTED_ALERTED_LEGIT)"
  note "expected generation_run_id=$EXPECTED_GENERATION_RUN_ID dataset_version=$EXPECTED_DATASET_VERSION"
  note "scoring batch cap (_MAX_PENDING_ALERTS_PER_RUN) = $(json_get "$PY_JSON_FILE" max_pending_alerts_per_scoring_run)"

  prove_no_forbidden_contact
  pass "STAGE A complete -- infrastructure preflight"
}

# ---------------------------------------------------------------------------
# 11. Population snapshot (shared by stages B/C/verify and the final audit)
# ---------------------------------------------------------------------------

population_snapshot() {
  # population_snapshot <tag> -> writes $DEMO_REPORT_DIR/<tag>.json
  local tag="$1"
  local rc=0
  run_python "snapshot_$tag" <<'PYSNAP' || rc=$?
import json, os
import psycopg2.extras
from src.common.db import get_connection

channel = os.environ["DEMO_CHANNEL"]
database = os.environ["DEMO_DATABASE"]
generation_run_id = os.environ.get("GENERATION_RUN_ID") or None

out = {}
conn = get_connection(database)
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT event_id) AS distinct_ids, "
                "count(DISTINCT generation_run_id) AS run_ids, count(DISTINCT dataset_version) AS dsvs, "
                "min(event_timestamp) AS ts_min, max(event_timestamp) AS ts_max "
                "FROM channel_events WHERE channel = %s", (channel,))
            row = dict(cur.fetchone())
            out["channel_events"] = {
                "count": row["total"], "distinct_event_ids": row["distinct_ids"],
                "distinct_generation_run_ids": row["run_ids"], "distinct_dataset_versions": row["dsvs"],
                "event_timestamp_min": str(row["ts_min"]), "event_timestamp_max": str(row["ts_max"]),
            }

            cur.execute(
                "SELECT DISTINCT generation_run_id, dataset_version FROM channel_events WHERE channel = %s "
                "ORDER BY generation_run_id", (channel,))
            out["generation_identities"] = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT sa.source_alert_id) AS distinct_ids, "
                "count(DISTINCT (sa.source_system, sa.source_alert_id)) AS distinct_keys "
                "FROM source_alerts sa JOIN channel_events ce ON ce.event_id = sa.event_id "
                "WHERE ce.channel = %s", (channel,))
            row = dict(cur.fetchone())
            out["source_alerts"] = {
                "count": row["total"], "distinct_source_alert_ids": row["distinct_ids"],
                "distinct_source_keys": row["distinct_keys"],
            }

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT sel.event_id) AS distinct_ids, "
                "count(*) FILTER (WHERE sel.synthetic_scenario_label) AS fraud, "
                "count(*) FILTER (WHERE NOT sel.synthetic_scenario_label) AS legitimate "
                "FROM synthetic_event_labels sel JOIN channel_events ce ON ce.event_id = sel.event_id "
                "WHERE ce.channel = %s", (channel,))
            row = dict(cur.fetchone())
            out["synthetic_event_labels"] = {
                "count": row["total"], "distinct_event_ids": row["distinct_ids"],
                "fraud": row["fraud"], "legitimate": row["legitimate"],
            }

            cur.execute(
                "SELECT count(*) AS total, "
                "count(*) FILTER (WHERE sel.synthetic_scenario_label) AS fraud, "
                "count(*) FILTER (WHERE NOT sel.synthetic_scenario_label) AS legitimate "
                "FROM source_alerts sa JOIN channel_events ce ON ce.event_id = sa.event_id "
                "JOIN synthetic_event_labels sel ON sel.event_id = ce.event_id WHERE ce.channel = %s", (channel,))
            row = dict(cur.fetchone())
            out["source_alerted_ground_truth"] = {
                "count": row["total"], "fraud": row["fraud"], "legitimate": row["legitimate"],
            }

            cur.execute(
                "SELECT status, count(*) AS n FROM channel_model_bundles WHERE channel = %s GROUP BY status "
                "ORDER BY status", (channel,))
            out["bundles_by_status"] = {r["status"]: r["n"] for r in cur.fetchall()}

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT (source_system, source_alert_id)) AS distinct_source_keys "
                "FROM fraud_alerts WHERE channel = %s", (channel,))
            out["fraud_alerts"] = dict(cur.fetchone())

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT ae.evidence_id) AS distinct_ids, "
                "count(*) FILTER (WHERE ae.degraded) AS degraded "
                "FROM alert_evidence ae JOIN fraud_alerts fa ON fa.alert_id = ae.alert_id "
                "WHERE fa.channel = %s", (channel,))
            out["alert_evidence"] = dict(cur.fetchone())

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT la.alert_id) AS distinct_alerts "
                "FROM label_assessments la JOIN fraud_alerts fa ON fa.alert_id = la.alert_id "
                "WHERE fa.channel = %s", (channel,))
            out["label_assessments"] = dict(cur.fetchone())

            cur.execute(
                "SELECT pipeline_name, status, count(*) AS n FROM pipeline_runs "
                "GROUP BY pipeline_name, status ORDER BY pipeline_name, status")
            out["pipeline_runs"] = [dict(r) for r in cur.fetchall()]

            if generation_run_id:
                cur.execute(
                    "SELECT count(*) AS n FROM channel_events WHERE channel = %s AND generation_run_id = %s",
                    (channel, generation_run_id))
                out["events_in_generation_run"] = cur.fetchone()["n"]
finally:
    conn.close()

print(json.dumps(out, sort_keys=True, default=str))
PYSNAP
  require_python_ok "$rc" "population_snapshot($tag)"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/${tag}.json"
  SNAPSHOT_FILE="$DEMO_REPORT_DIR/${tag}.json"
}

compare_snapshots() {
  # compare_snapshots <before.json> <after.json> <label>
  local rc=0
  ( cd "$REPO_ROOT" && "$PYTHON_BIN" - "$1" "$2" <<'PYCMP'
import json, sys
before = json.load(open(sys.argv[1]))
after = json.load(open(sys.argv[2]))
if before != after:
    diffs = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            diffs.append("%s: %r -> %r" % (key, before.get(key), after.get(key)))
    sys.stderr.write("snapshot changed:\n  " + "\n  ".join(diffs) + "\n")
    raise SystemExit(1)
PYCMP
  ) || rc=$?
  if [ "$rc" -ne 0 ]; then
    die "$3: database state changed when it must not have"
  fi
  pass "$3: database state unchanged"
}

# ---------------------------------------------------------------------------
# 12. STAGE B -- first generation
# ---------------------------------------------------------------------------

stage_generate() {
  stage_banner "B" "First generation (real CLI, writes to $DEMO_DATABASE)"
  assert_demo_database "generate"

  local rc=0
  run_cli "generate" fraud-intel generate \
    --channel "$DEMO_CHANNEL" \
    --count "$DEMO_COUNT" \
    --seed "$DEMO_SEED" \
    --reference-date "$DEMO_REFERENCE_DATE" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel generate"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/generate.json"

  expect_eq "generate.channel"              "$(json_get "$CLI_JSON_FILE" channel)"              "$DEMO_CHANNEL"
  expect_eq "generate.requested_count"      "$(json_get "$CLI_JSON_FILE" requested_count)"      "$DEMO_COUNT"
  expect_eq "generate.inserted_event_count" "$(json_get "$CLI_JSON_FILE" inserted_event_count)" "$DEMO_COUNT"
  expect_eq "generate.existing_event_count" "$(json_get "$CLI_JSON_FILE" existing_event_count)" "0"
  expect_eq "generate.label_count"          "$(json_get "$CLI_JSON_FILE" label_count)"          "$DEMO_COUNT"
  expect_eq "generate.source_alert_count"   "$(json_get "$CLI_JSON_FILE" source_alert_count)"   "$EXPECTED_SOURCE_ALERT_COUNT"
  expect_eq "generate.reference_date"       "$(json_get "$CLI_JSON_FILE" reference_date)"       "$DEMO_REFERENCE_DATE"
  expect_eq "generate.seed"                 "$(json_get "$CLI_JSON_FILE" seed)"                 "$DEMO_SEED"
  expect_eq "generate.generation_run_id"    "$(json_get "$CLI_JSON_FILE" generation_run_id)"    "$EXPECTED_GENERATION_RUN_ID"
  expect_eq "generate.dataset_version"      "$(json_get "$CLI_JSON_FILE" dataset_version)"      "$EXPECTED_DATASET_VERSION"

  GENERATION_RUN_ID="$(json_get "$CLI_JSON_FILE" generation_run_id)"
  DATASET_VERSION="$(json_get "$CLI_JSON_FILE" dataset_version)"
  export GENERATION_RUN_ID DATASET_VERSION
  state_put generation_run_id "$GENERATION_RUN_ID"
  state_put dataset_version "$DATASET_VERSION"

  population_snapshot "population_after_generate"
  verify_population "$SNAPSHOT_FILE" "after first generation"
  pass "STAGE B complete -- $DEMO_COUNT events, $EXPECTED_SOURCE_ALERT_COUNT source alerts, $DEMO_COUNT labels persisted under one generation identity"
}

verify_population() {
  # verify_population <snapshot.json> <label>
  local snap="$1" label="$2"
  local rc=0
  ( cd "$REPO_ROOT" && SNAP="$snap" "$PYTHON_BIN" - <<'PYVER'
import json, os, sys

snap = json.load(open(os.environ["SNAP"]))
count = int(os.environ["DEMO_COUNT"])
expected_alerts = int(os.environ["EXPECTED_SOURCE_ALERT_COUNT"])
expected_fraud = int(os.environ["EXPECTED_FRAUD_COUNT"])
expected_alerted_fraud = int(os.environ["EXPECTED_ALERTED_FRAUD"])
expected_alerted_legit = int(os.environ["EXPECTED_ALERTED_LEGIT"])
run_id = os.environ["GENERATION_RUN_ID"]
dsv = os.environ["DATASET_VERSION"]

problems = []
ce = snap["channel_events"]
if ce["count"] != count:
    problems.append("channel_events=%d, expected %d" % (ce["count"], count))
if ce["distinct_event_ids"] != count:
    problems.append("duplicate channel_events rows (%d rows, %d distinct event_ids)" % (ce["count"], ce["distinct_event_ids"]))
if ce["distinct_generation_run_ids"] != 1:
    problems.append("expected exactly 1 generation_run_id, found %d" % ce["distinct_generation_run_ids"])
if ce["distinct_dataset_versions"] != 1:
    problems.append("expected exactly 1 dataset_version, found %d" % ce["distinct_dataset_versions"])

identities = snap["generation_identities"]
if identities != [{"generation_run_id": run_id, "dataset_version": dsv}]:
    problems.append("persisted generation identity %r != captured (%r, %r)" % (identities, run_id, dsv))

sa = snap["source_alerts"]
if sa["count"] != expected_alerts:
    problems.append("source_alerts=%d, expected %d (deterministic generator)" % (sa["count"], expected_alerts))
if sa["distinct_source_alert_ids"] != sa["count"] or sa["distinct_source_keys"] != sa["count"]:
    problems.append("duplicate source_alerts rows")

sel = snap["synthetic_event_labels"]
if sel["count"] != count:
    problems.append("synthetic_event_labels=%d, expected %d" % (sel["count"], count))
if sel["distinct_event_ids"] != sel["count"]:
    problems.append("duplicate synthetic_event_labels rows")
if sel["fraud"] != expected_fraud:
    problems.append("fraud labels=%d, expected %d" % (sel["fraud"], expected_fraud))
if sel["fraud"] + sel["legitimate"] != count:
    problems.append("fraud+legitimate labels do not reconcile to %d" % count)

gt = snap["source_alerted_ground_truth"]
if gt["count"] != expected_alerts:
    problems.append("source-alerted rows joined to labels=%d, expected %d" % (gt["count"], expected_alerts))
if gt["fraud"] != expected_alerted_fraud or gt["legitimate"] != expected_alerted_legit:
    problems.append(
        "source-alerted ground truth (fraud=%d, legitimate=%d) != generator expectation (fraud=%d, legitimate=%d)"
        % (gt["fraud"], gt["legitimate"], expected_alerted_fraud, expected_alerted_legit))

if snap.get("events_in_generation_run") not in (None, count):
    problems.append("events under generation_run_id=%d, expected %d" % (snap["events_in_generation_run"], count))

if problems:
    sys.stderr.write("population verification failed:\n  " + "\n  ".join(problems) + "\n")
    raise SystemExit(1)
PYVER
  ) || rc=$?
  [ "$rc" -eq 0 ] || die "population verification failed ($label)"
  pass "population reconciles ($label): events/source-alerts/labels, one generation identity, zero duplicates"
}

# ---------------------------------------------------------------------------
# 13. STAGE C -- generation retry (idempotency)
# ---------------------------------------------------------------------------

stage_generate_retry() {
  stage_banner "C" "Generation retry (identical command must be a no-op)"
  assert_demo_database "generate_retry"
  resolve_generation_identity
  export GENERATION_RUN_ID DATASET_VERSION

  population_snapshot "population_before_generate_retry"
  local before="$SNAPSHOT_FILE"

  local rc=0
  run_cli "generate_retry" fraud-intel generate \
    --channel "$DEMO_CHANNEL" \
    --count "$DEMO_COUNT" \
    --seed "$DEMO_SEED" \
    --reference-date "$DEMO_REFERENCE_DATE" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel generate (retry)"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/generate_retry.json"

  expect_eq "retry.inserted_event_count" "$(json_get "$CLI_JSON_FILE" inserted_event_count)" "0"
  expect_eq "retry.existing_event_count" "$(json_get "$CLI_JSON_FILE" existing_event_count)" "$DEMO_COUNT"
  expect_eq "retry.generation_run_id"    "$(json_get "$CLI_JSON_FILE" generation_run_id)"    "$GENERATION_RUN_ID"
  expect_eq "retry.dataset_version"      "$(json_get "$CLI_JSON_FILE" dataset_version)"      "$DATASET_VERSION"
  expect_eq "retry.source_alert_count"   "$(json_get "$CLI_JSON_FILE" source_alert_count)"   "$EXPECTED_SOURCE_ALERT_COUNT"

  population_snapshot "population_after_generate_retry"
  compare_snapshots "$before" "$SNAPSHOT_FILE" "generation retry"
  pass "STAGE C complete -- retry inserted nothing and changed nothing"
}

# ---------------------------------------------------------------------------
# 14. Population verification (explicit workflow step)
# ---------------------------------------------------------------------------

stage_verify() {
  stage_banner "B/C" "Population verification"
  resolve_generation_identity
  export GENERATION_RUN_ID DATASET_VERSION
  population_snapshot "population_verified"
  verify_population "$SNAPSHOT_FILE" "population verification"
  note "$(json_get "$SNAPSHOT_FILE" channel_events)"
  note "$(json_get "$SNAPSHOT_FILE" source_alerts)"
  note "$(json_get "$SNAPSHOT_FILE" synthetic_event_labels)"
  prove_no_forbidden_contact
  pass "STAGE B/C complete -- population verified"
}

# ---------------------------------------------------------------------------
# 15. STAGE D -- chronological split preview (read-only)
# ---------------------------------------------------------------------------

stage_split_preview() {
  stage_banner "D" "Chronological split preview (read-only; real loaders and selectors)"
  resolve_generation_identity
  export GENERATION_RUN_ID DATASET_VERSION

  local rc=0
  run_python "split_preview" <<'PYSPLIT' || rc=$?
import json, os, sys
from datetime import timedelta

import polars as pl

from src.common.splits import assign_chronological_split, realized_split_fractions
from src.fraud_intel.cli_data_access import load_channel_population, load_cross_channel_customer_pool
from src.fraud_intel.config import ChannelTrainingRunConfig
from src.fraud_intel.models.training import (
    InsufficientTrainingDataError,
    _build_supervised_population,
    _class_counts,
    _dataset_version,
    _validate_partition,
)
from src.fraud_intel.registry import get_channel_adapter

channel = os.environ["DEMO_CHANNEL"]
database = os.environ["DEMO_DATABASE"]
generation_run_id = os.environ["GENERATION_RUN_ID"]

# The REAL production loaders -- identical calls to src/cli/__main__.py's
# _handle_fraud_intel_train().
events, source_alerts, labels = load_channel_population(channel, database, generation_run_id=generation_run_id)
customer_ids = {e.customer_id for e in events}
cross_channel_events, cross_channel_source_alerts = load_cross_channel_customer_pool(customer_ids, database)

adapter = get_channel_adapter(channel)
config = ChannelTrainingRunConfig(channel=channel)

# The REAL supervised-population builder, which itself uses the shared
# historical feature selectors (src.fraud_intel.features.history).
rows = _build_supervised_population(
    adapter=adapter,
    channel_events=events,
    source_alerts=source_alerts,
    synthetic_labels=labels,
    history_events=cross_channel_events,
    history_source_alerts=cross_channel_source_alerts,
)

report = {
    "channel": channel,
    "generation_run_id": generation_run_id,
    "loaded_events": len(events),
    "loaded_source_alerts": len(source_alerts),
    "loaded_synthetic_labels": len(labels),
    "cross_channel_history_events": len(cross_channel_events),
    "cross_channel_history_source_alerts": len(cross_channel_source_alerts),
    "supervised_population": len(rows),
    "supervised_population_hash": _dataset_version(rows) if rows else None,
    "feature_schema_version": adapter.feature_schema_version,
    "feature_columns": list(adapter.feature_columns),
    "split_config": {
        "train_frac": config.train_frac,
        "calib_frac": config.calib_frac,
        "test_frac": config.test_frac,
        "purge_gap_seconds": config.purge_gap_seconds,
        "min_train_rows": config.min_train_rows,
        "min_calib_rows": config.min_calib_rows,
        "min_test_rows": config.min_test_rows,
    },
}

if not rows:
    print(json.dumps(report, sort_keys=True, default=str))
    raise SystemExit("no source-alerted, eligible %s rows -- training would fail" % channel)

population_fraud = sum(1 for r in rows if r["label"])
report["supervised_fraud"] = population_fraud
report["supervised_legitimate"] = len(rows) - population_fraud

df = pl.DataFrame([
    {
        "event_id": r["event_id"],
        "event_timestamp": r["event_timestamp"],
        "label": r["label"],
        **{c: r[c] for c in adapter.feature_columns},
    }
    for r in rows
])

split_df = assign_chronological_split(
    df,
    timestamp_col="event_timestamp",
    id_col="event_id",
    train_frac=config.train_frac,
    calib_frac=config.calib_frac,
    test_frac=config.test_frac,
    purge_gap=timedelta(seconds=config.purge_gap_seconds),
)
report["realized_split_fractions"] = realized_split_fractions(split_df)

partitions = {}
for name in ("train", "calibration", "test"):
    part = split_df.filter(pl.col("split") == name)
    timestamps = part["event_timestamp"].to_list()
    partitions[name] = {
        "rows": part.height,
        "class_counts": _class_counts(part),
        "event_timestamp_min": str(min(timestamps)) if timestamps else None,
        "event_timestamp_max": str(max(timestamps)) if timestamps else None,
    }
report["partitions"] = partitions

assigned_ids = set(split_df["event_id"].to_list())
all_ids = set(df["event_id"].to_list())
report["purged_rows"] = len(all_ids - assigned_ids)

# Partition disjointness (row identity) -- must be total.
id_sets = {
    name: set(split_df.filter(pl.col("split") == name)["event_id"].to_list())
    for name in ("train", "calibration", "test")
}
overlaps = {}
for a, b in (("train", "calibration"), ("calibration", "test"), ("train", "test")):
    overlaps["%s_vs_%s" % (a, b)] = len(id_sets[a] & id_sets[b])
report["row_id_overlap_counts"] = overlaps

# Equal-timestamp boundary overlap: a timestamp value that appears in more
# than one partition would mean a timestamp GROUP was split across the
# boundary (assign_chronological_split snaps to group starts precisely to
# prevent this).
ts_sets = {
    name: set(split_df.filter(pl.col("split") == name)["event_timestamp"].to_list())
    for name in ("train", "calibration", "test")
}
boundary = {}
for a, b in (("train", "calibration"), ("calibration", "test"), ("train", "test")):
    boundary["%s_vs_%s" % (a, b)] = len(ts_sets[a] & ts_sets[b])
report["equal_timestamp_boundary_overlap_counts"] = boundary

# Chronological ordering: each partition must start no earlier than the
# previous one ends.
def _mx(name):
    return partitions[name]["event_timestamp_max"]
def _mn(name):
    return partitions[name]["event_timestamp_min"]
report["chronological_order_ok"] = bool(
    _mx("train") <= _mn("calibration") and _mx("calibration") <= _mn("test")
)

# The REAL training gates -- same function train_channel_configured() calls.
gate_failures = []
for name, min_rows in (
    ("train", config.min_train_rows),
    ("calibration", config.min_calib_rows),
    ("test", config.min_test_rows),
):
    try:
        _validate_partition(split_df.filter(pl.col("split") == name), name, min_rows=min_rows)
    except InsufficientTrainingDataError as exc:
        gate_failures.append(str(exc))
report["training_gate_failures"] = gate_failures

print(json.dumps(report, sort_keys=True, default=str))

problems = list(gate_failures)
if any(v for v in overlaps.values()):
    problems.append("partitions share row ids: %r" % overlaps)
if any(v for v in boundary.values()):
    problems.append("equal-timestamp groups straddle a split boundary: %r" % boundary)
if not report["chronological_order_ok"]:
    problems.append("partitions are not in chronological order")
if problems:
    sys.stderr.write("split preview gate failure:\n  " + "\n  ".join(problems) + "\n")
    raise SystemExit(1)
PYSPLIT
  require_python_ok "$rc" "split preview"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/split_preview.json"

  note "supervised population: $(json_get "$PY_JSON_FILE" supervised_population) (fraud=$(json_get "$PY_JSON_FILE" supervised_fraud), legitimate=$(json_get "$PY_JSON_FILE" supervised_legitimate))"
  note "partitions: $(json_get "$PY_JSON_FILE" partitions)"
  note "realized fractions: $(json_get "$PY_JSON_FILE" realized_split_fractions)"
  note "purged rows: $(json_get "$PY_JSON_FILE" purged_rows)"
  note "row-id overlap: $(json_get "$PY_JSON_FILE" row_id_overlap_counts)"
  note "equal-timestamp boundary overlap: $(json_get "$PY_JSON_FILE" equal_timestamp_boundary_overlap_counts)"
  note "gate config: $(json_get "$PY_JSON_FILE" split_config)"
  pass "STAGE D complete -- chronological split preview passed every training gate"
}

# ---------------------------------------------------------------------------
# 16. STAGE E -- training
# ---------------------------------------------------------------------------

stage_train() {
  stage_banner "E" "Training (real CLI; writes a CANDIDATE bundle + isolated MLflow runs)"
  assert_demo_database "train"
  resolve_generation_identity
  export GENERATION_RUN_ID DATASET_VERSION

  local rc=0
  run_cli "train" fraud-intel train \
    --channel "$DEMO_CHANNEL" \
    --generation-run-id "$GENERATION_RUN_ID" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel train"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/train.json"

  TRAINING_RUN_ID="$(json_get "$CLI_JSON_FILE" run_id)"
  BUNDLE_ID="$(json_get "$CLI_JSON_FILE" bundle_id)"
  BUNDLE_VERSION="$(json_get "$CLI_JSON_FILE" bundle_version)"
  export TRAINING_RUN_ID BUNDLE_ID BUNDLE_VERSION
  state_put training_run_id "$TRAINING_RUN_ID"
  state_put bundle_id "$BUNDLE_ID"
  state_put bundle_version "$BUNDLE_VERSION"

  expect_eq "train.source_generation_run_id" "$(json_get "$CLI_JSON_FILE" source_generation_run_id)" "$GENERATION_RUN_ID"
  expect_eq "train.source_dataset_version"   "$(json_get "$CLI_JSON_FILE" source_dataset_version)"   "$DATASET_VERSION"
  info "captured bundle_id=$BUNDLE_ID bundle_version=$BUNDLE_VERSION training_run_id=$TRAINING_RUN_ID"

  rc=0
  run_python "train_verify" <<'PYTRAIN' || rc=$?
import json, os, sys
import psycopg2.extras

import mlflow

from src.common.db import get_connection
from src.common.mlflow_setup import configure_mlflow
from src.fraud_intel.evaluation.cold_start import load_and_validate_cold_start_report
from src.fraud_intel.models.bundle import REQUIRED_OPERATIONAL_COMPONENTS, ChannelModelBundleRecord
from src.fraud_intel.models.mlflow_naming import MODEL_COMPONENTS, registered_model_name
from src.fraud_intel.models.preprocessing import ChannelPreprocessor
from src.fraud_intel.models.anomaly import AnomalyNormalization
from src.fraud_intel.models.training import _experiment_name

channel = os.environ["DEMO_CHANNEL"]
database = os.environ["DEMO_DATABASE"]
generation_run_id = os.environ["GENERATION_RUN_ID"]
dataset_version = os.environ["DATASET_VERSION"]
bundle_version = int(os.environ["BUNDLE_VERSION"])
training_run_id = int(os.environ["TRAINING_RUN_ID"])

problems = []
report = {}

conn = get_connection(database)
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM channel_model_bundles WHERE channel = %s ORDER BY bundle_version", (channel,))
            bundle_rows = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT run_id, pipeline_name, status, dataset_version, model_version, records_processed, "
                "records_rejected, artifacts FROM pipeline_runs ORDER BY run_id")
            runs = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT count(*) AS n FROM fraud_alerts WHERE channel = %s", (channel,))
            alerts_n = cur.fetchone()["n"]
            cur.execute(
                "SELECT count(*) AS n FROM alert_evidence ae JOIN fraud_alerts fa ON fa.alert_id = ae.alert_id "
                "WHERE fa.channel = %s", (channel,))
            evidence_n = cur.fetchone()["n"]
            cur.execute(
                "SELECT count(*) AS n FROM label_assessments la JOIN fraud_alerts fa ON fa.alert_id = la.alert_id "
                "WHERE fa.channel = %s", (channel,))
            assessments_n = cur.fetchone()["n"]
finally:
    conn.close()

report["bundle_count"] = len(bundle_rows)
if len(bundle_rows) != 1:
    problems.append("expected exactly 1 bundle row, found %d" % len(bundle_rows))
    print(json.dumps({"report": report, "problems": problems}, sort_keys=True, default=str))
    raise SystemExit(1)

bundle = ChannelModelBundleRecord(**bundle_rows[0])
report["bundle"] = {
    "bundle_id": bundle.bundle_id, "bundle_version": bundle.bundle_version, "status": bundle.status,
    "training_run_id": bundle.training_run_id, "dataset_version": bundle.dataset_version,
    "promoted_by": bundle.promoted_by, "promoted_at": str(bundle.promoted_at),
}
if bundle.status != "CANDIDATE":
    problems.append("bundle status is %r, expected CANDIDATE" % bundle.status)
if bundle.bundle_version != bundle_version:
    problems.append("bundle_version %r != captured %r" % (bundle.bundle_version, bundle_version))
if bundle.training_run_id != training_run_id:
    problems.append("bundle.training_run_id %r != captured %r" % (bundle.training_run_id, training_run_id))

missing = [f for f in REQUIRED_OPERATIONAL_COMPONENTS if getattr(bundle, f) is None]
report["missing_operational_components"] = missing
if missing:
    problems.append("bundle is missing required operational components: %s" % missing)

# Provenance: generation identity, supervised-population hash.
cold_report = load_and_validate_cold_start_report(bundle)
report["provenance"] = {
    "source_generation_run_id": cold_report.source_generation_run_id,
    "source_dataset_version": cold_report.source_dataset_version,
    "supervised_population_hash": cold_report.supervised_population_hash,
    "bundle_dataset_version": bundle.dataset_version,
    "feature_schema_version": cold_report.feature_schema_version,
}
if cold_report.source_generation_run_id != generation_run_id:
    problems.append("report.source_generation_run_id %r != %r" % (cold_report.source_generation_run_id, generation_run_id))
if cold_report.source_dataset_version != dataset_version:
    problems.append("report.source_dataset_version %r != %r" % (cold_report.source_dataset_version, dataset_version))
if cold_report.supervised_population_hash != bundle.dataset_version:
    problems.append("supervised_population_hash does not match channel_model_bundles.dataset_version")

# Lifecycle: exactly one train/SUCCESS row and no scoring/labels/promotion.
by_name = {}
for r in runs:
    by_name.setdefault(r["pipeline_name"], []).append(r)
report["pipeline_runs"] = {k: [{"run_id": r["run_id"], "status": r["status"]} for r in v] for k, v in by_name.items()}
train_runs = by_name.get("train", [])
if len(train_runs) != 1 or train_runs[0]["status"] != "SUCCESS" or train_runs[0]["run_id"] != training_run_id:
    problems.append("expected exactly one train/SUCCESS run with run_id=%d, found %r" % (training_run_id, train_runs))
for forbidden_name in ("fraud_score", "label_eligibility", "model_promotion"):
    if by_name.get(forbidden_name):
        problems.append("training must not create a %s run, found %r" % (forbidden_name, by_name[forbidden_name]))

report["side_effects"] = {"fraud_alerts": alerts_n, "alert_evidence": evidence_n, "label_assessments": assessments_n}
if alerts_n or evidence_n or assessments_n:
    problems.append("training created alert/evidence/assessment rows: %r" % report["side_effects"])

# MLflow: exactly three component runs and three READY registered versions.
configure_mlflow()
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(_experiment_name(channel))
if experiment is None:
    problems.append("MLflow experiment %r was not created" % _experiment_name(channel))
    mlflow_runs = []
else:
    mlflow_runs = client.search_runs([experiment.experiment_id])
report["mlflow_run_count"] = len(mlflow_runs)
report["mlflow_run_names"] = sorted(r.data.tags.get("mlflow.runName", "") for r in mlflow_runs)
if len(mlflow_runs) != 3:
    problems.append("expected 3 MLflow component runs, found %d" % len(mlflow_runs))

expected_run_ids = {
    "gbm": cold_report.gbm_mlflow_run_id,
    "lr-shadow": cold_report.lr_mlflow_run_id,
    "anomaly": cold_report.anomaly_mlflow_run_id,
}
expected_versions = {
    "gbm": bundle.gbm_model_version,
    "lr-shadow": bundle.lr_model_version,
    "anomaly": bundle.anomaly_model_version,
}
report["registered_versions"] = {}
for component in MODEL_COMPONENTS:
    name = registered_model_name(channel, component)
    mv = client.get_model_version(name, expected_versions[component])
    report["registered_versions"][name] = {"version": mv.version, "status": mv.status, "run_id": mv.run_id}
    if mv.status != "READY":
        problems.append("%s version %s status is %r, expected READY" % (name, mv.version, mv.status))
    if mv.run_id != expected_run_ids[component]:
        problems.append("%s version %s points at run %r, expected %r" % (name, mv.version, mv.run_id, expected_run_ids[component]))

# Artifacts must actually load back.
gbm_run_id = cold_report.gbm_mlflow_run_id
preproc_raw = mlflow.artifacts.load_dict("runs:/%s/preprocessor.json" % gbm_run_id)
preprocessor = ChannelPreprocessor.from_json_dict(preproc_raw)
report["preprocessing_artifact_version"] = preprocessor.preprocessing_artifact_version
if preprocessor.preprocessing_artifact_version != bundle.preprocessing_artifact_version:
    problems.append("reloaded preprocessor version %r != bundle %r" % (
        preprocessor.preprocessing_artifact_version, bundle.preprocessing_artifact_version))

anom_raw = mlflow.artifacts.load_dict("runs:/%s/anomaly_normalization.json" % cold_report.anomaly_mlflow_run_id)
normalization = AnomalyNormalization.from_json_dict(anom_raw)
report["anomaly_artifact_version"] = getattr(normalization, "anomaly_artifact_version", None)

report["gbm_evaluation"] = cold_report.gbm_evaluation
report["lr_shadow_evaluation"] = cold_report.lr_shadow_evaluation
report["split_class_counts"] = cold_report.split_class_counts
report["realized_split_fractions"] = cold_report.realized_split_fractions

print(json.dumps({"report": report, "problems": problems}, sort_keys=True, default=str))
if problems:
    sys.stderr.write("training verification failed:\n  " + "\n  ".join(problems) + "\n")
    raise SystemExit(1)
PYTRAIN
  require_python_ok "$rc" "training verification"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/train_verify.json"

  note "bundle: $(json_get "$PY_JSON_FILE" report.bundle)"
  note "provenance: $(json_get "$PY_JSON_FILE" report.provenance)"
  note "MLflow runs: $(json_get "$PY_JSON_FILE" report.mlflow_run_count) -- $(json_get "$PY_JSON_FILE" report.mlflow_run_names)"
  note "registered versions: $(json_get "$PY_JSON_FILE" report.registered_versions)"
  note "split class counts: $(json_get "$PY_JSON_FILE" report.split_class_counts)"
  note "no scoring, labels or promotion side effects: $(json_get "$PY_JSON_FILE" report.side_effects)"
  prove_no_forbidden_contact
  pass "STAGE E complete -- one CANDIDATE bundle, complete components, 3 MLflow runs, 3 READY versions, artifacts reload"
}

# ---------------------------------------------------------------------------
# 17. STAGE F -- candidate evaluation (read-only, cold start)
# ---------------------------------------------------------------------------

stage_evaluate_candidate() {
  stage_banner "F" "Candidate evaluation (real CLI; read-only)"
  resolve_generation_identity
  resolve_bundle_identity
  export GENERATION_RUN_ID DATASET_VERSION BUNDLE_ID BUNDLE_VERSION

  population_snapshot "population_before_candidate_evaluation"
  local before="$SNAPSHOT_FILE"

  local rc=0
  run_cli "evaluate_candidate" fraud-intel evaluate \
    --channel "$DEMO_CHANNEL" \
    --generation-run-id "$GENERATION_RUN_ID" \
    --candidate-bundle-version "$BUNDLE_VERSION" \
    --capacity-mode count \
    --capacity-value "$DEMO_ANALYST_CAPACITY" \
    --recall-target "$DEMO_RECALL_TARGET" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel evaluate (candidate)"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/evaluate_candidate.json"

  expect_eq "evaluate.evaluation_mode"  "$(json_get "$CLI_JSON_FILE" evaluation_mode)"  "candidate_training_holdout"
  expect_eq "evaluate.generation_run_id" "$(json_get "$CLI_JSON_FILE" generation_run_id)" "$GENERATION_RUN_ID"
  expect_eq "evaluate.source_dataset_version" "$(json_get "$CLI_JSON_FILE" source_dataset_version)" "$DATASET_VERSION"
  expect_eq "evaluate.candidate_bundle.bundle_version" "$(json_get "$CLI_JSON_FILE" candidate_bundle.bundle_version)" "$BUNDLE_VERSION"

  local gate_passed
  gate_passed="$(json_get "$CLI_JSON_FILE" promotion_gate_result.passed)"
  if [ "$gate_passed" != "true" ]; then
    printf '[FAIL] promotion gate did not pass:\n' >&2
    json_get "$CLI_JSON_FILE" promotion_gate_result.reasons >&2
    die "candidate promotion gate failed -- promotion is blocked"
  fi
  pass "promotion gate passed"

  note "held-out GBM metrics:  $(json_get "$CLI_JSON_FILE" candidate_evaluation.gbm_evaluation)"
  note "LR shadow metrics:     $(json_get "$CLI_JSON_FILE" candidate_evaluation.lr_shadow_evaluation)"
  note "split class counts:    $(json_get "$CLI_JSON_FILE" promotion_gate_result.split_class_counts)"
  note "realized fractions:    $(json_get "$CLI_JSON_FILE" candidate_evaluation.realized_split_fractions)"
  note "test fraud prevalence: $(json_get "$CLI_JSON_FILE" promotion_gate_result.test_fraud_prevalence)"
  note "gbm pr_auc:            $(json_get "$CLI_JSON_FILE" promotion_gate_result.gbm_pr_auc)"
  note "rules-only baseline:   $(json_get "$CLI_JSON_FILE" rules_only_baseline)"
  note "disclaimer:            $(json_get "$CLI_JSON_FILE" disclaimer)"

  population_snapshot "population_after_candidate_evaluation"
  compare_snapshots "$before" "$SNAPSHOT_FILE" "candidate evaluation is read-only"
  pass "STAGE F complete -- held-out evidence reported, promotion gate passed, nothing written"
}

# ---------------------------------------------------------------------------
# 18. STAGE G -- non-persistent full-candidate diagnostic
# ---------------------------------------------------------------------------

stage_diagnostic() {
  stage_banner "G" "Full-candidate diagnostic (pure scoring path; NOTHING persisted)"
  resolve_generation_identity
  resolve_bundle_identity
  export GENERATION_RUN_ID DATASET_VERSION BUNDLE_ID BUNDLE_VERSION

  population_snapshot "population_before_diagnostic"
  local before="$SNAPSHOT_FILE"

  local rc=0
  DIAGNOSTIC_SCORES_FILE="$DEMO_DIAGNOSTIC_SCORES_FILE" \
  run_python "diagnostic" <<'PYDIAG' || rc=$?
import json, os, sys, uuid
from collections import Counter

import psycopg2.extras

from src.common.db import get_connection
from src.control_plane.provenance import get_git_sha
from src.fraud_intel.ensemble.policy import load_ensemble_policy
from src.fraud_intel.models.promotion import create_default_bundle_promotion_store
from src.fraud_intel.rules.provider import _load_rule_set_config
from src.fraud_intel.scoring.dispatch import (
    _MAX_PENDING_ALERTS_PER_RUN,
    _PostgresScoringDataAccess,
    _bundle_config_hash,
    create_default_bundle_artifact_loader,
    load_and_validate_pinned_policies,
)
from src.fraud_intel.scoring.orchestrator import score_source_alert

channel = os.environ["DEMO_CHANNEL"]
database = os.environ["DEMO_DATABASE"]
generation_run_id = os.environ["GENERATION_RUN_ID"]
bundle_version = int(os.environ["BUNDLE_VERSION"])
expected_alerts = int(os.environ["EXPECTED_SOURCE_ALERT_COUNT"])
capacity = int(os.environ["DEMO_ANALYST_CAPACITY"])
scores_path = os.environ["DIAGNOSTIC_SCORES_FILE"]

problems = []

store = create_default_bundle_promotion_store(database)
bundle = store.get_bundle(channel, bundle_version)
if bundle.status != "CANDIDATE":
    problems.append("bundle %s status is %r, expected CANDIDATE" % (bundle.bundle_id, bundle.status))

rule_provider, graph_policy, ensemble_policy = load_and_validate_pinned_policies(bundle)
loaded_bundle = create_default_bundle_artifact_loader().load(bundle)
config_hash = _bundle_config_hash(bundle)
git_sha = get_git_sha()

# The REAL production pending-selection query. No OPERATIONAL bundle exists
# and no alert_evidence row exists for this bundle_id, so this returns the
# whole generated source-alert population.
data_access = _PostgresScoringDataAccess(database)
source_dataset_version = data_access.validate_generation_run(channel, generation_run_id)
items = data_access.list_pending(
    channel, generation_run_id=generation_run_id, operational_bundle_id=bundle.bundle_id
)
if len(items) != expected_alerts:
    problems.append(
        "selected %d source alerts, expected the full generated population of %d "
        "(_MAX_PENDING_ALERTS_PER_RUN=%d)" % (len(items), expected_alerts, _MAX_PENDING_ALERTS_PER_RUN)
    )

# Generated ground truth, joined per source alert (never fed into scoring).
event_ids = [str(item.event.event_id) for item in items]
conn = get_connection(database)
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT event_id, synthetic_scenario_label, scenario_type FROM synthetic_event_labels "
                "WHERE event_id = ANY(%s::uuid[])", (event_ids,))
            truth = {str(r["event_id"]): r for r in cur.fetchall()}
            cur.execute("SELECT count(*) AS n FROM fraud_alerts WHERE channel = %s", (channel,))
            alerts_before = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM alert_evidence")
            evidence_before = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM pipeline_runs")
            runs_before = cur.fetchone()["n"]
finally:
    conn.close()

rule_set = _load_rule_set_config(channel)
all_rule_ids = [r.id for r in rule_set.rules]
rule_categories = {r.id: r.category for r in rule_set.rules}

band_counts = Counter()
cross_tab = {}
rule_firings = Counter()
component_errors = Counter()
component_ok = Counter()
degraded_count = 0
high_mandatory = 0
high_threshold = 0
high_floored = 0
scoring_errors = []
rows = []

for item in items:
    try:
        scored = score_source_alert(
            event=item.event,
            source_alert=item.source_alert,
            context=item.context,
            bundle=loaded_bundle,
            rule_provider=rule_provider,
            ensemble_policy=ensemble_policy,
            graph_policy=graph_policy,
            resolved_fraud_evidence=item.resolved_fraud_evidence,
            score_execution_id=uuid.uuid4(),
            config_hash=config_hash,
            git_sha=git_sha,
        )
    except Exception as exc:  # never abort the batch on one alert
        scoring_errors.append({
            "source_alert_id": str(item.source_alert.source_alert_id),
            "error_type": type(exc).__name__,
        })
        continue

    is_fraud = bool(truth.get(str(item.event.event_id), {}).get("synthetic_scenario_label"))
    band_counts[scored.priority_band] += 1
    key = "%s/%s" % (scored.priority_band, "fraud" if is_fraud else "legitimate")
    cross_tab[key] = cross_tab.get(key, 0) + 1

    for rule_id in scored.rule_result.fired_rule_ids:
        rule_firings[rule_id] += 1

    for name, status in scored.component_statuses.items():
        if status.status == "OK":
            component_ok[name] += 1
        else:
            component_errors["%s:%s" % (name, status.error_code)] += 1
    if scored.degraded:
        degraded_count += 1

    mandatory_hit = any(
        scored.rule_result.rule_categories.get(rid) == "MANDATORY_REVIEW"
        for rid in scored.rule_result.fired_rule_ids
    )
    if scored.priority_band == "HIGH":
        if mandatory_hit:
            high_mandatory += 1
        elif scored.operational_priority_score >= ensemble_policy.high_threshold:
            high_threshold += 1
        else:
            high_floored += 1

    rows.append({
        "source_alert_id": str(scored.source_alert_id),
        "event_id": str(scored.event_id),
        "event_timestamp": item.event.event_timestamp.isoformat(),
        "operational_priority_score": scored.operational_priority_score,
        "priority_band": scored.priority_band,
        "degraded": scored.degraded,
        "is_fraud": is_fraud,
        "fired_rule_ids": list(scored.rule_result.fired_rule_ids),
    })

if scoring_errors:
    problems.append("%d alert(s) failed to score: %r" % (len(scoring_errors), scoring_errors[:5]))

scores = [r["operational_priority_score"] for r in rows]
total_fraud = sum(1 for r in rows if r["is_fraud"])

# Same deterministic ranking rule the evaluation modules use:
# score desc, event_timestamp asc, source_alert_id asc.
ranked = sorted(rows, key=lambda r: (-r["operational_priority_score"], r["event_timestamp"], r["source_alert_id"]))
top = ranked[:capacity]
fraud_in_top = sum(1 for r in top if r["is_fraud"])

report = {
    "channel": channel,
    "generation_run_id": generation_run_id,
    "source_dataset_version": source_dataset_version,
    "candidate_bundle_id": bundle.bundle_id,
    "candidate_bundle_version": bundle.bundle_version,
    "scored_population": len(rows),
    "expected_population": expected_alerts,
    "priority_band_counts": {b: band_counts.get(b, 0) for b in ("LOW", "MEDIUM", "HIGH")},
    "score_min": min(scores) if scores else None,
    "score_mean": (sum(scores) / len(scores)) if scores else None,
    "score_max": max(scores) if scores else None,
    "band_by_ground_truth": dict(sorted(cross_tab.items())),
    "ground_truth_totals": {"fraud": total_fraud, "legitimate": len(rows) - total_fraud},
    "rule_firing_counts": {rid: rule_firings.get(rid, 0) for rid in all_rule_ids},
    "rule_categories": rule_categories,
    "component_ok_counts": dict(component_ok),
    "component_error_counts": dict(component_errors),
    "degraded_alert_count": degraded_count,
    "scoring_errors": scoring_errors,
    "capacity": capacity,
    "fraud_captured_in_top_capacity": fraud_in_top,
    "precision_at_capacity": (fraud_in_top / len(top)) if top else None,
    "recall_at_capacity": (fraud_in_top / total_fraud) if total_fraud else None,
    "high_band_breakdown": {
        "mandatory_review": high_mandatory,
        "threshold_exceeded": high_threshold,
        "component_failure_floor": high_floored,
    },
    "ensemble_policy": {
        "policy_version": ensemble_policy.policy_version,
        "high_threshold": ensemble_policy.high_threshold,
        "medium_threshold": ensemble_policy.medium_threshold,
        "calibration_status": ensemble_policy.calibration_status,
        "promotion_note": ensemble_policy.promotion_note,
    },
    "warning": (
        "config/fraud_intel/ensemble_policy_%s.yaml declares calibration_status=%s -- the ensemble "
        "weights and LOW/MEDIUM/HIGH thresholds behind every number above are an UNTUNED placeholder "
        "copied from online_banking. These are POC demonstration figures on synthetic data, not "
        "production or regulatory evidence." % (channel, ensemble_policy.calibration_status)
    ),
    "persistence": "none -- score_source_alert() is pure; no fraud_alerts, alert_evidence or pipeline_runs row was created",
}

# Prove non-persistence from the database itself.
conn = get_connection(database)
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT count(*) AS n FROM fraud_alerts WHERE channel = %s", (channel,))
            alerts_after = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM alert_evidence")
            evidence_after = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM pipeline_runs")
            runs_after = cur.fetchone()["n"]
finally:
    conn.close()
report["persistence_check"] = {
    "fraud_alerts": [alerts_before, alerts_after],
    "alert_evidence": [evidence_before, evidence_after],
    "pipeline_runs": [runs_before, runs_after],
}
if (alerts_before, evidence_before, runs_before) != (alerts_after, evidence_after, runs_after):
    problems.append("the diagnostic wrote to the database: %r" % report["persistence_check"])

with open(scores_path, "w") as fh:
    json.dump({"rows": rows}, fh, sort_keys=True)

print(json.dumps(report, sort_keys=True, default=str))
if problems:
    sys.stderr.write("diagnostic failed:\n  " + "\n  ".join(problems) + "\n")
    raise SystemExit(1)
PYDIAG
  require_python_ok "$rc" "full-candidate diagnostic"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/diagnostic.json"
  cp "$DEMO_DIAGNOSTIC_SCORES_FILE" "$DEMO_REPORT_DIR/diagnostic_scores.json"

  printf '\n--- CANDIDATE DIAGNOSTIC REPORT ---------------------------------------------\n'
  note "scored population:      $(json_get "$PY_JSON_FILE" scored_population) of $(json_get "$PY_JSON_FILE" expected_population)"
  note "priority bands:         $(json_get "$PY_JSON_FILE" priority_band_counts)"
  note "score min/mean/max:     $(json_get "$PY_JSON_FILE" score_min) / $(json_get "$PY_JSON_FILE" score_mean) / $(json_get "$PY_JSON_FILE" score_max)"
  note "band x ground truth:    $(json_get "$PY_JSON_FILE" band_by_ground_truth)"
  note "ground-truth totals:    $(json_get "$PY_JSON_FILE" ground_truth_totals)"
  note "rule firings:           $(json_get "$PY_JSON_FILE" rule_firing_counts)"
  note "rule categories:        $(json_get "$PY_JSON_FILE" rule_categories)"
  note "component OK counts:    $(json_get "$PY_JSON_FILE" component_ok_counts)"
  note "component errors:       $(json_get "$PY_JSON_FILE" component_error_counts)"
  note "degraded alerts:        $(json_get "$PY_JSON_FILE" degraded_alert_count)"
  note "fraud in top $DEMO_ANALYST_CAPACITY:        $(json_get "$PY_JSON_FILE" fraud_captured_in_top_capacity)"
  note "precision@$DEMO_ANALYST_CAPACITY:           $(json_get "$PY_JSON_FILE" precision_at_capacity)"
  note "recall@$DEMO_ANALYST_CAPACITY:              $(json_get "$PY_JSON_FILE" recall_at_capacity)"
  note "HIGH breakdown:         $(json_get "$PY_JSON_FILE" high_band_breakdown)"
  note "ensemble policy:        $(json_get "$PY_JSON_FILE" ensemble_policy)"
  printf '\n[WARNING] %s\n\n' "$(json_get "$PY_JSON_FILE" warning)" >&2
  printf -- '-----------------------------------------------------------------------------\n'

  population_snapshot "population_after_diagnostic"
  compare_snapshots "$before" "$SNAPSHOT_FILE" "diagnostic is non-persistent"
  pass "STAGE G complete -- full candidate population scored in memory, nothing persisted"
}

# ---------------------------------------------------------------------------
# 19. STAGE H -- manual approval pause
# ---------------------------------------------------------------------------

stage_approval() {
  stage_banner "H" "Manual approval pause"
  if [ "$OPT_APPROVE_PROMOTION" -eq 1 ]; then
    pass "explicit promotion approval supplied (--approve-promotion) by operator '$DEMO_PROMOTED_BY'"
    return 0
  fi

  cat <<APPROVAL

  The candidate bundle has NOT been promoted. Nothing further will run.

  Review the candidate report before approving:
    $DEMO_REPORT_DIR/diagnostic.json
    $DEMO_REPORT_DIR/evaluate_candidate.json
    $DEMO_REPORT_DIR/split_preview.json
    $DEMO_REPORT_DIR/train_verify.json

  Captured runtime identifiers (state: $DEMO_STATE_FILE):
    generation_run_id = ${GENERATION_RUN_ID:-<unresolved>}
    dataset_version   = ${DATASET_VERSION:-<unresolved>}
    bundle_id         = ${BUNDLE_ID:-<unresolved>}
    bundle_version    = ${BUNDLE_VERSION:-<unresolved>}

  To promote and continue, run the second, explicit invocation:

    scripts/demo_debit_card_full_lifecycle.sh --resume-from promote --approve-promotion

APPROVAL
  printf '[HALT] Stopped for manual approval. Promotion requires --approve-promotion.\n'
  printf '[RESULT] Stages A-G PASSED. Awaiting approval.\n'
  exit 0
}

# ---------------------------------------------------------------------------
# 20. STAGE I -- promotion
# ---------------------------------------------------------------------------

bundle_components_snapshot() {
  # bundle_components_snapshot <tag>
  local tag="$1"
  local rc=0
  run_python "bundle_components_$tag" <<'PYBC' || rc=$?
import json, os
import psycopg2.extras
from src.common.db import get_connection

channel = os.environ["DEMO_CHANNEL"]
conn = get_connection(os.environ["DEMO_DATABASE"])
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT bundle_id, bundle_version, gbm_model_version, lr_model_version, anomaly_model_version, "
                "preprocessing_artifact_version, feature_schema_version, rule_set_version, graph_policy_version, "
                "ensemble_policy_version, reason_code_version, training_run_id, dataset_version "
                "FROM channel_model_bundles WHERE channel = %s ORDER BY bundle_version", (channel,))
            rows = [dict(r) for r in cur.fetchall()]
finally:
    conn.close()
print(json.dumps(rows, sort_keys=True, default=str))
PYBC
  require_python_ok "$rc" "bundle_components_snapshot($tag)"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/bundle_components_${tag}.json"
  BUNDLE_COMPONENTS_FILE="$DEMO_REPORT_DIR/bundle_components_${tag}.json"
}

stage_promote() {
  stage_banner "I" "Promotion (requires explicit --approve-promotion)"
  [ "$OPT_APPROVE_PROMOTION" -eq 1 ] || \
    die "promotion requires --approve-promotion; re-run: scripts/demo_debit_card_full_lifecycle.sh --resume-from promote --approve-promotion"

  assert_demo_database "promote"
  resolve_generation_identity
  resolve_bundle_identity
  export GENERATION_RUN_ID DATASET_VERSION BUNDLE_ID BUNDLE_VERSION

  bundle_components_snapshot "before_promote"
  local components_before="$BUNDLE_COMPONENTS_FILE"

  local rc=0
  run_cli "promote" fraud-intel promote \
    --channel "$DEMO_CHANNEL" \
    --bundle-version "$BUNDLE_VERSION" \
    --promoted-by "$DEMO_PROMOTED_BY" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel promote"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/promote.json"

  expect_eq "promote.status"         "$(json_get "$CLI_JSON_FILE" status)"         "OPERATIONAL"
  expect_eq "promote.bundle_version" "$(json_get "$CLI_JSON_FILE" bundle_version)" "$BUNDLE_VERSION"
  expect_eq "promote.bundle_id"      "$(json_get "$CLI_JSON_FILE" bundle_id)"      "$BUNDLE_ID"
  expect_eq "promote.promoted_by"    "$(json_get "$CLI_JSON_FILE" promoted_by)"    "$DEMO_PROMOTED_BY"

  bundle_components_snapshot "after_promote"
  compare_snapshots "$components_before" "$BUNDLE_COMPONENTS_FILE" "promotion leaves model versions and pinned policies unchanged"

  rc=0
  run_python "promote_verify" <<'PYPROM' || rc=$?
import json, os, sys
import psycopg2.extras
from src.common.db import get_connection

channel = os.environ["DEMO_CHANNEL"]
bundle_id = int(os.environ["BUNDLE_ID"])
promoted_by = os.environ["DEMO_PROMOTED_BY"]

problems = []
conn = get_connection(os.environ["DEMO_DATABASE"])
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT bundle_id, bundle_version, status, promoted_by, promoted_at "
                "FROM channel_model_bundles WHERE channel = %s ORDER BY bundle_version", (channel,))
            bundles = [dict(r) for r in cur.fetchall()]
            cur.execute(
                "SELECT run_id, status, artifacts FROM pipeline_runs WHERE pipeline_name = 'model_promotion' "
                "ORDER BY run_id")
            promotions = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT count(*) AS n FROM pipeline_runs WHERE pipeline_name = 'fraud_score'")
            scoring_runs = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM fraud_alerts WHERE channel = %s", (channel,))
            alerts = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM alert_evidence")
            evidence = cur.fetchone()["n"]
finally:
    conn.close()

operational = [b for b in bundles if b["status"] == "OPERATIONAL"]
if len(operational) != 1:
    problems.append("expected exactly 1 OPERATIONAL bundle, found %d" % len(operational))
elif operational[0]["bundle_id"] != bundle_id:
    problems.append("OPERATIONAL bundle_id %r != promoted %r" % (operational[0]["bundle_id"], bundle_id))
elif operational[0]["promoted_by"] != promoted_by:
    problems.append("promoted_by %r != %r" % (operational[0]["promoted_by"], promoted_by))

if len(promotions) != 1 or promotions[0]["status"] != "SUCCESS":
    problems.append("expected exactly one model_promotion/SUCCESS run, found %r" % promotions)

if scoring_runs or alerts or evidence:
    problems.append(
        "promotion must not score: fraud_score runs=%d, fraud_alerts=%d, alert_evidence=%d"
        % (scoring_runs, alerts, evidence))

print(json.dumps({
    "bundles": bundles,
    "model_promotion_runs": promotions,
    "fraud_score_runs": scoring_runs,
    "fraud_alerts": alerts,
    "alert_evidence": evidence,
    "problems": problems,
}, sort_keys=True, default=str))
if problems:
    sys.stderr.write("promotion verification failed:\n  " + "\n  ".join(problems) + "\n")
    raise SystemExit(1)
PYPROM
  require_python_ok "$rc" "promotion verification"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/promote_verify.json"
  note "bundles: $(json_get "$PY_JSON_FILE" bundles)"
  note "model_promotion runs: $(json_get "$PY_JSON_FILE" model_promotion_runs)"
  prove_no_forbidden_contact
  pass "STAGE I complete -- exactly one OPERATIONAL bundle promoted by '$DEMO_PROMOTED_BY', no scoring occurred"
}

# ---------------------------------------------------------------------------
# 21. STAGE J -- first scoring
# ---------------------------------------------------------------------------

stage_score() {
  stage_banner "J" "First scoring (generation-scoped, real CLI)"
  assert_demo_database "score"
  resolve_generation_identity
  resolve_bundle_identity
  export GENERATION_RUN_ID DATASET_VERSION BUNDLE_ID BUNDLE_VERSION

  local rc=0
  run_cli "score" fraud-intel score \
    --channel "$DEMO_CHANNEL" \
    --generation-run-id "$GENERATION_RUN_ID" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel score"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/score.json"

  local pending processed rejected
  pending="$(json_get "$CLI_JSON_FILE" pending_count)"
  processed="$(json_get "$CLI_JSON_FILE" records_processed)"
  rejected="$(json_get "$CLI_JSON_FILE" records_rejected)"
  expect_eq "score.pending_count"     "$pending"   "$EXPECTED_SOURCE_ALERT_COUNT"
  expect_eq "score.records_processed" "$processed" "$pending"
  expect_eq "score.records_rejected"  "$rejected"  "0"
  expect_eq "score.bundle_id"         "$(json_get "$CLI_JSON_FILE" bundle_id)" "$BUNDLE_ID"
  expect_eq "score.source_dataset_version" "$(json_get "$CLI_JSON_FILE" source_dataset_version)" "$DATASET_VERSION"

  [ -f "$DEMO_DIAGNOSTIC_SCORES_FILE" ] || \
    die "missing $DEMO_DIAGNOSTIC_SCORES_FILE -- stage G (diagnostic) must run before scoring can be compared against it; re-run with --resume-from diagnostic"

  rc=0
  DIAGNOSTIC_SCORES_FILE="$DEMO_DIAGNOSTIC_SCORES_FILE" \
  run_python "score_verify" <<'PYSCORE' || rc=$?
import json, os, sys
import psycopg2.extras
from src.common.db import get_connection

channel = os.environ["DEMO_CHANNEL"]
database = os.environ["DEMO_DATABASE"]
generation_run_id = os.environ["GENERATION_RUN_ID"]
bundle_id = int(os.environ["BUNDLE_ID"])
expected_alerts = int(os.environ["EXPECTED_SOURCE_ALERT_COUNT"])
diag_path = os.environ["DIAGNOSTIC_SCORES_FILE"]

problems = []
conn = get_connection(database)
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT count(*) AS n FROM source_alerts sa JOIN channel_events ce ON ce.event_id = sa.event_id "
                "WHERE ce.channel = %s AND ce.generation_run_id = %s", (channel, generation_run_id))
            population = cur.fetchone()["n"]

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT (fa.source_system, fa.source_alert_id)) AS distinct_keys "
                "FROM fraud_alerts fa JOIN channel_events ce ON ce.event_id = fa.event_id "
                "WHERE fa.channel = %s AND ce.generation_run_id = %s", (channel, generation_run_id))
            alerts = dict(cur.fetchone())

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT ae.alert_id) AS distinct_alerts, "
                "count(*) FILTER (WHERE ae.degraded) AS degraded "
                "FROM alert_evidence ae JOIN fraud_alerts fa ON fa.alert_id = ae.alert_id "
                "JOIN channel_events ce ON ce.event_id = fa.event_id "
                "WHERE fa.channel = %s AND ce.generation_run_id = %s AND ae.channel_model_bundle_id = %s",
                (channel, generation_run_id, bundle_id))
            evidence = dict(cur.fetchone())

            cur.execute(
                "SELECT count(*) AS n FROM fraud_alerts fa JOIN channel_events ce ON ce.event_id = fa.event_id "
                "WHERE fa.channel = %s AND ce.generation_run_id = %s AND NOT EXISTS ("
                "  SELECT 1 FROM alert_evidence ae WHERE ae.alert_id = fa.alert_id "
                "    AND ae.channel_model_bundle_id = %s)",
                (channel, generation_run_id, bundle_id))
            missing_evidence = cur.fetchone()["n"]

            cur.execute(
                "SELECT count(*) AS n FROM alert_evidence ae JOIN fraud_alerts fa ON fa.alert_id = ae.alert_id "
                "WHERE fa.channel = %s AND ae.channel_model_bundle_id != %s", (channel, bundle_id))
            foreign_bundle_evidence = cur.fetchone()["n"]

            cur.execute(
                "SELECT ae.component_statuses FROM alert_evidence ae JOIN fraud_alerts fa "
                "ON fa.alert_id = ae.alert_id WHERE fa.channel = %s AND ae.channel_model_bundle_id = %s",
                (channel, bundle_id))
            status_rows = [r["component_statuses"] for r in cur.fetchall()]

            cur.execute(
                "SELECT fa.source_alert_id, ae.operational_priority_score, ae.priority_band, ae.degraded "
                "FROM alert_evidence ae JOIN fraud_alerts fa ON fa.alert_id = ae.alert_id "
                "WHERE fa.channel = %s AND ae.channel_model_bundle_id = %s", (channel, bundle_id))
            persisted = {str(r["source_alert_id"]): dict(r) for r in cur.fetchall()}

            cur.execute(
                "SELECT run_id, status, records_processed, records_rejected FROM pipeline_runs "
                "WHERE pipeline_name = 'fraud_score' ORDER BY run_id")
            scoring_runs = [dict(r) for r in cur.fetchall()]
finally:
    conn.close()

if population != expected_alerts:
    problems.append("source-alert population %d != expected %d" % (population, expected_alerts))
if alerts["total"] != population or alerts["distinct_keys"] != population:
    problems.append("expected exactly one fraud_alert per source alert: %r (population=%d)" % (alerts, population))
if evidence["total"] != population or evidence["distinct_alerts"] != population:
    problems.append("expected exactly one current-bundle evidence row per alert: %r" % evidence)
if evidence["degraded"]:
    problems.append("%d degraded evidence row(s)" % evidence["degraded"])
if missing_evidence:
    problems.append("%d alert(s) have no current-bundle evidence" % missing_evidence)
if foreign_bundle_evidence:
    problems.append("%d evidence row(s) reference a different bundle" % foreign_bundle_evidence)

bad_components = []
for statuses in status_rows:
    for name, status in (statuses or {}).items():
        if status.get("status") != "OK":
            bad_components.append({name: status})
if bad_components:
    problems.append("%d non-OK component status(es), e.g. %r" % (len(bad_components), bad_components[:3]))

if len(scoring_runs) != 1 or scoring_runs[0]["status"] != "SUCCESS":
    problems.append("expected exactly one fraud_score/SUCCESS run, found %r" % scoring_runs)

# Persisted results must match the pre-promotion, in-memory diagnostic.
comparison = {"compared": 0, "band_mismatches": 0, "score_mismatches": 0, "missing_from_persisted": 0}
with open(diag_path) as fh:
    diagnostic_rows = json.load(fh)["rows"]
for row in diagnostic_rows:
    sid = row["source_alert_id"]
    got = persisted.get(sid)
    if got is None:
        comparison["missing_from_persisted"] += 1
        continue
    comparison["compared"] += 1
    if got["priority_band"] != row["priority_band"]:
        comparison["band_mismatches"] += 1
    # alert_evidence.operational_priority_score is NUMERIC(7,6).
    if round(float(got["operational_priority_score"]), 6) != round(float(row["operational_priority_score"]), 6):
        comparison["score_mismatches"] += 1
if comparison["missing_from_persisted"] or comparison["band_mismatches"] or comparison["score_mismatches"]:
    problems.append("persisted scoring disagrees with the pre-promotion diagnostic: %r" % comparison)

print(json.dumps({
    "population": population, "fraud_alerts": alerts, "current_bundle_evidence": evidence,
    "missing_current_bundle_evidence": missing_evidence, "foreign_bundle_evidence": foreign_bundle_evidence,
    "non_ok_component_statuses": len(bad_components), "fraud_score_runs": scoring_runs,
    "diagnostic_comparison": comparison, "problems": problems,
}, sort_keys=True, default=str))
if problems:
    sys.stderr.write("scoring verification failed:\n  " + "\n  ".join(problems) + "\n")
    raise SystemExit(1)
PYSCORE
  require_python_ok "$rc" "scoring verification"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/score_verify.json"
  note "fraud_alerts:            $(json_get "$PY_JSON_FILE" fraud_alerts)"
  note "current-bundle evidence: $(json_get "$PY_JSON_FILE" current_bundle_evidence)"
  note "diagnostic comparison:   $(json_get "$PY_JSON_FILE" diagnostic_comparison)"
  prove_no_forbidden_contact
  pass "STAGE J complete -- $pending alerts scored, one alert and one evidence row each, zero rejected/degraded/duplicate, results match the diagnostic"
}

# ---------------------------------------------------------------------------
# 22. STAGE K -- scoring retry
# ---------------------------------------------------------------------------

stage_score_retry() {
  stage_banner "K" "Scoring retry (identical command must be a no-op)"
  assert_demo_database "score_retry"
  resolve_generation_identity
  export GENERATION_RUN_ID DATASET_VERSION

  population_snapshot "population_before_score_retry"
  local before="$SNAPSHOT_FILE"
  local evidence_before="$DEMO_REPORT_DIR/evidence_digest_before_score_retry.json"
  evidence_digest "$evidence_before"

  local rc=0
  run_cli "score_retry" fraud-intel score \
    --channel "$DEMO_CHANNEL" \
    --generation-run-id "$GENERATION_RUN_ID" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel score (retry)"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/score_retry.json"

  expect_eq "retry.pending_count"     "$(json_get "$CLI_JSON_FILE" pending_count)"     "0"
  expect_eq "retry.records_processed" "$(json_get "$CLI_JSON_FILE" records_processed)" "0"
  expect_eq "retry.records_rejected"  "$(json_get "$CLI_JSON_FILE" records_rejected)"  "0"
  expect_eq "retry.alerts"            "$(json_get "$CLI_JSON_FILE" alerts)"            "[]"

  local evidence_after="$DEMO_REPORT_DIR/evidence_digest_after_score_retry.json"
  evidence_digest "$evidence_after"
  compare_snapshots "$evidence_before" "$evidence_after" "scoring retry leaves alert/evidence rows unchanged"

  population_snapshot "population_after_score_retry"
  # pipeline_runs legitimately gains one more fraud_score/SUCCESS row (the
  # retry itself is an audited run); everything else must be identical.
  compare_ignoring_pipeline_runs "$before" "$SNAPSHOT_FILE" "scoring retry"
  pass "STAGE K complete -- retry processed nothing and changed no alert or evidence row"
}

evidence_digest() {
  # evidence_digest <output.json>
  local out="$1"
  local rc=0
  run_python "evidence_digest" <<'PYED' || rc=$?
import json, os
import psycopg2.extras
from src.common.db import get_connection

channel = os.environ["DEMO_CHANNEL"]
conn = get_connection(os.environ["DEMO_DATABASE"])
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT fa.alert_id, fa.source_alert_id, fa.status, fa.initial_priority_band, "
                "fa.initial_operational_priority_score, ae.evidence_id, ae.score_execution_id, "
                "ae.priority_band, ae.operational_priority_score, ae.degraded, ae.channel_model_bundle_id "
                "FROM fraud_alerts fa LEFT JOIN alert_evidence ae ON ae.alert_id = fa.alert_id "
                "WHERE fa.channel = %s ORDER BY fa.alert_id, ae.evidence_id", (channel,))
            rows = [dict(r) for r in cur.fetchall()]
finally:
    conn.close()
print(json.dumps({"rows": rows, "row_count": len(rows)}, sort_keys=True, default=str))
PYED
  require_python_ok "$rc" "evidence_digest"
  cp "$PY_JSON_FILE" "$out"
}

compare_ignoring_pipeline_runs() {
  # compare_ignoring_pipeline_runs <before.json> <after.json> <label>
  local rc=0
  ( cd "$REPO_ROOT" && "$PYTHON_BIN" - "$1" "$2" <<'PYCMP2'
import json, sys
before = json.load(open(sys.argv[1]))
after = json.load(open(sys.argv[2]))
before.pop("pipeline_runs", None)
after.pop("pipeline_runs", None)
if before != after:
    diffs = ["%s: %r -> %r" % (k, before.get(k), after.get(k))
             for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)]
    sys.stderr.write("state changed:\n  " + "\n  ".join(diffs) + "\n")
    raise SystemExit(1)
PYCMP2
  ) || rc=$?
  [ "$rc" -eq 0 ] || die "$3: database state changed when it must not have (outside pipeline_runs)"
  pass "$3: database state unchanged (outside the audited pipeline_runs row)"
}

# ---------------------------------------------------------------------------
# 23. STAGE L / M -- label assessment and its retry
# ---------------------------------------------------------------------------

label_digest() {
  # label_digest <output.json>
  local rc=0
  run_python "label_digest" <<'PYLD' || rc=$?
import json, os
import psycopg2.extras
from src.common.db import get_connection

channel = os.environ["DEMO_CHANNEL"]
conn = get_connection(os.environ["DEMO_DATABASE"])
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT la.assessment_id, la.alert_id, la.policy_version, la.source_disposition_id, "
                "la.basis_timestamp, la.maturity_due_at, la.maturity_status, la.eligibility_result, "
                "la.eligibility_reason_code, la.resolved_label, la.resolved_label_source "
                "FROM label_assessments la JOIN fraud_alerts fa ON fa.alert_id = la.alert_id "
                "WHERE fa.channel = %s ORDER BY la.assessment_id", (channel,))
            rows = [dict(r) for r in cur.fetchall()]
finally:
    conn.close()
print(json.dumps({"rows": rows, "row_count": len(rows)}, sort_keys=True, default=str))
PYLD
  require_python_ok "$rc" "label_digest"
  cp "$PY_JSON_FILE" "$1"
}

stage_labels() {
  stage_banner "L" "First label assessment (generation-scoped)"
  assert_demo_database "labels"
  resolve_generation_identity
  export GENERATION_RUN_ID DATASET_VERSION

  local rc=0
  run_cli "labels" fraud-intel labels assess \
    --channel "$DEMO_CHANNEL" \
    --generation-run-id "$GENERATION_RUN_ID" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel labels assess"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/labels.json"

  local bases
  bases="$(json_get "$CLI_JSON_FILE" bases_evaluated)"
  expect_eq "labels.bases_evaluated"          "$bases"                                                      "$EXPECTED_SOURCE_ALERT_COUNT"
  expect_eq "labels.assessments_inserted"     "$(json_get "$CLI_JSON_FILE" assessments_inserted)"           "$bases"
  expect_eq "labels.assessments_unchanged"    "$(json_get "$CLI_JSON_FILE" assessments_unchanged)"          "0"
  expect_eq "labels.mature_count"             "$(json_get "$CLI_JSON_FILE" mature_count)"                   "$bases"
  expect_eq "labels.immature_count"           "$(json_get "$CLI_JSON_FILE" immature_count)"                 "0"
  expect_eq "labels.eligible_count"           "$(json_get "$CLI_JSON_FILE" eligible_count)"                 "$bases"
  expect_eq "labels.unresolved_count"         "$(json_get "$CLI_JSON_FILE" unresolved_count)"               "0"
  expect_eq "labels.resolved_fraud_count"     "$(json_get "$CLI_JSON_FILE" resolved_fraud_count)"           "$EXPECTED_ALERTED_FRAUD"
  expect_eq "labels.resolved_legitimate_count" "$(json_get "$CLI_JSON_FILE" resolved_legitimate_count)"     "$EXPECTED_ALERTED_LEGIT"
  expect_eq "labels.source_dataset_version"   "$(json_get "$CLI_JSON_FILE" source_dataset_version)"         "$DATASET_VERSION"

  label_digest "$DEMO_REPORT_DIR/label_digest_after_first_assessment.json"
  rc=0
  LABEL_DIGEST_FILE="$DEMO_REPORT_DIR/label_digest_after_first_assessment.json" \
  run_python "labels_verify" <<'PYLV' || rc=$?
import json, os, sys
from collections import Counter

digest = json.load(open(os.environ["LABEL_DIGEST_FILE"]))
rows = digest["rows"]
expected = int(os.environ["EXPECTED_SOURCE_ALERT_COUNT"])
expected_fraud = int(os.environ["EXPECTED_ALERTED_FRAUD"])
expected_legit = int(os.environ["EXPECTED_ALERTED_LEGIT"])

problems = []
alert_ids = Counter(r["alert_id"] for r in rows)
duplicates = [a for a, n in alert_ids.items() if n > 1]
if len(rows) != expected:
    problems.append("label_assessments rows=%d, expected %d" % (len(rows), expected))
if duplicates:
    problems.append("%d alert(s) have more than one assessment" % len(duplicates))
if any(r["maturity_status"] != "MATURE" for r in rows):
    problems.append("not every assessment is MATURE")
if any(not r["eligibility_result"] for r in rows):
    problems.append("not every assessment is eligible")
if any(r["resolved_label_source"] != "SYNTHETIC_GENERATOR" for r in rows):
    problems.append("expected every label to come from SYNTHETIC_GENERATOR")
if any(r["source_disposition_id"] is not None for r in rows):
    problems.append("synthetic assessments must have source_disposition_id = NULL")

labels = Counter(r["resolved_label"] for r in rows)
if labels.get("RESOLVED_FRAUD", 0) != expected_fraud or labels.get("RESOLVED_LEGITIMATE", 0) != expected_legit:
    problems.append("resolved labels %r != generated ground truth (fraud=%d, legitimate=%d)"
                    % (dict(labels), expected_fraud, expected_legit))
if labels.get("UNRESOLVED", 0):
    problems.append("%d UNRESOLVED assessment(s)" % labels["UNRESOLVED"])

print(json.dumps({
    "rows": len(rows), "duplicate_alerts": len(duplicates),
    "resolved_labels": dict(labels),
    "maturity": dict(Counter(r["maturity_status"] for r in rows)),
    "eligibility": dict(Counter(bool(r["eligibility_result"]) for r in rows)),
    "reason_codes": dict(Counter(r["eligibility_reason_code"] for r in rows)),
    "problems": problems,
}, sort_keys=True, default=str))
if problems:
    sys.stderr.write("label verification failed:\n  " + "\n  ".join(problems) + "\n")
    raise SystemExit(1)
PYLV
  require_python_ok "$rc" "label verification"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/labels_verify.json"
  note "resolved labels: $(json_get "$PY_JSON_FILE" resolved_labels)"
  note "reason codes:    $(json_get "$PY_JSON_FILE" reason_codes)"
  prove_no_forbidden_contact
  pass "STAGE L complete -- $bases assessments inserted, all MATURE and eligible, labels match generated ground truth"
}

stage_labels_retry() {
  stage_banner "M" "Label assessment retry (identical command must be a no-op)"
  assert_demo_database "labels_retry"
  resolve_generation_identity
  export GENERATION_RUN_ID DATASET_VERSION

  local before="$DEMO_REPORT_DIR/label_digest_before_retry.json"
  label_digest "$before"

  local rc=0
  run_cli "labels_retry" fraud-intel labels assess \
    --channel "$DEMO_CHANNEL" \
    --generation-run-id "$GENERATION_RUN_ID" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel labels assess (retry)"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/labels_retry.json"

  local bases
  bases="$(json_get "$CLI_JSON_FILE" bases_evaluated)"
  expect_eq "retry.bases_evaluated"       "$bases"                                              "$EXPECTED_SOURCE_ALERT_COUNT"
  expect_eq "retry.assessments_inserted"  "$(json_get "$CLI_JSON_FILE" assessments_inserted)"   "0"
  expect_eq "retry.assessments_unchanged" "$(json_get "$CLI_JSON_FILE" assessments_unchanged)"  "$bases"

  local after="$DEMO_REPORT_DIR/label_digest_after_retry.json"
  label_digest "$after"
  compare_snapshots "$before" "$after" "label assessment retry leaves label_assessments unchanged"
  pass "STAGE M complete -- retry inserted nothing; the assessment table is byte-identical"
}

# ---------------------------------------------------------------------------
# 24. STAGE N -- live evaluation (read-only)
# ---------------------------------------------------------------------------

stage_live_evaluate() {
  stage_banner "N" "Live evaluation (generation- and operational-bundle-scoped; read-only)"
  resolve_generation_identity
  resolve_bundle_identity
  export GENERATION_RUN_ID DATASET_VERSION BUNDLE_ID BUNDLE_VERSION

  population_snapshot "population_before_live_evaluation"
  local before="$SNAPSHOT_FILE"

  local rc=0
  run_cli "live_evaluate" fraud-intel evaluate \
    --channel "$DEMO_CHANNEL" \
    --generation-run-id "$GENERATION_RUN_ID" \
    --capacity-mode count \
    --capacity-value "$DEMO_ANALYST_CAPACITY" \
    --recall-target "$DEMO_RECALL_TARGET" \
    --database "$DEMO_DATABASE" \
    --json || rc=$?
  require_cli_ok "$rc" "fraud-intel evaluate (live)"
  cp "$CLI_JSON_FILE" "$DEMO_REPORT_DIR/live_evaluate.json"

  expect_eq "live.evaluation_mode"        "$(json_get "$CLI_JSON_FILE" evaluation_mode)"        "live_resolved_alerts"
  expect_eq "live.operational_bundle_id"  "$(json_get "$CLI_JSON_FILE" operational_bundle_id)"  "$BUNDLE_ID"
  expect_eq "live.evaluated_count"        "$(json_get "$CLI_JSON_FILE" evaluated_count)"        "$EXPECTED_SOURCE_ALERT_COUNT"
  expect_eq "live.resolved_eligible_count" "$(json_get "$CLI_JSON_FILE" resolved_eligible_count)" "$EXPECTED_SOURCE_ALERT_COUNT"
  expect_eq "live.missing_current_bundle_evidence_count" "$(json_get "$CLI_JSON_FILE" missing_current_bundle_evidence_count)" "0"
  expect_eq "live.degraded_evidence_count" "$(json_get "$CLI_JSON_FILE" degraded_evidence_count)" "0"

  rc=0
  LIVE_EVAL_FILE="$CLI_JSON_FILE" \
  run_python "live_evaluate_report" <<'PYLE' || rc=$?
import json, os

doc = json.load(open(os.environ["LIVE_EVAL_FILE"]))
op = doc["operational_evaluation"]

def value(metric):
    m = op.get(metric)
    if not isinstance(m, dict):
        return m
    return m.get("value") if m.get("status") == "ok" else "non-computable (%s)" % m.get("status")

precision = op["precision"].get("value") if op["precision"].get("status") == "ok" else None
recall = op["recall"].get("value") if op["recall"].get("status") == "ok" else None
if precision is not None and recall is not None and (precision + recall) > 0:
    f1 = 2 * precision * recall / (precision + recall)
elif precision is not None and recall is not None:
    f1 = 0.0
else:
    f1 = None

print(json.dumps({
    "evaluated_population": doc["evaluated_count"],
    "total_resolved_fraud": op["total_resolved_fraud"],
    "total_resolved_legitimate": op["total_resolved_legitimate"],
    "precision": value("precision"),
    "recall": value("recall"),
    # ChannelEvaluationResult (src/fraud_intel/evaluation/cross_channel.py)
    # has no F1 field -- this is DERIVED here by the demo script from the
    # CLI's own precision/recall, never produced by the product.
    "f1_derived_from_precision_recall": f1,
    "pr_auc": value("pr_auc"),
    "roc_auc": value("roc_auc"),
    "brier_score": value("brier_score"),
    "confusion_matrix_at_operational_threshold": op["confusion_matrix_at_operational_threshold"],
    "per_band_counts": op["per_band_counts"],
    "capacity_mode": op["capacity_mode"],
    "capacity_value": op["capacity_value"],
    "alerts_reviewed": op["alerts_reviewed"],
    "fraud_captured": op["fraud_captured"],
    "precision_at_capacity": value("precision_at_capacity"),
    "recall_at_capacity": value("recall_at_capacity"),
    "recall_target": op["recall_target"],
    "minimum_alerts_required_for_recall_target": op["minimum_alerts_required_for_recall_target"],
    "workload_reduction_at_recall_target": value("workload_reduction_at_recall_target"),
    "false_positive_reduction": value("false_positive_reduction"),
    "accuracy_supplementary_only": value("accuracy"),
    "rules_only_baseline": doc["rules_only_baseline"],
    "disclaimer": doc["disclaimer"],
}, sort_keys=True, default=str))
PYLE
  require_python_ok "$rc" "live evaluation report"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/live_evaluate_report.json"

  printf '\n--- LIVE EVALUATION ---------------------------------------------------------\n'
  note "evaluated population: $(json_get "$PY_JSON_FILE" evaluated_population) (fraud=$(json_get "$PY_JSON_FILE" total_resolved_fraud), legitimate=$(json_get "$PY_JSON_FILE" total_resolved_legitimate))"
  note "precision:            $(json_get "$PY_JSON_FILE" precision)"
  note "recall:               $(json_get "$PY_JSON_FILE" recall)"
  note "F1 (derived here):    $(json_get "$PY_JSON_FILE" f1_derived_from_precision_recall)"
  note "PR-AUC:               $(json_get "$PY_JSON_FILE" pr_auc)"
  note "ROC-AUC:              $(json_get "$PY_JSON_FILE" roc_auc)"
  note "Brier score:          $(json_get "$PY_JSON_FILE" brier_score)"
  note "confusion matrix:     $(json_get "$PY_JSON_FILE" confusion_matrix_at_operational_threshold)"
  note "per-band counts:      $(json_get "$PY_JSON_FILE" per_band_counts)"
  note "precision@$DEMO_ANALYST_CAPACITY:        $(json_get "$PY_JSON_FILE" precision_at_capacity)"
  note "recall@$DEMO_ANALYST_CAPACITY:           $(json_get "$PY_JSON_FILE" recall_at_capacity)"
  note "fraud captured in top $DEMO_ANALYST_CAPACITY: $(json_get "$PY_JSON_FILE" fraud_captured)"
  note "min alerts for recall $DEMO_RECALL_TARGET: $(json_get "$PY_JSON_FILE" minimum_alerts_required_for_recall_target)"
  note "workload reduction:   $(json_get "$PY_JSON_FILE" workload_reduction_at_recall_target)"
  note "rules-only baseline:  $(json_get "$PY_JSON_FILE" rules_only_baseline)"
  note "disclaimer:           $(json_get "$PY_JSON_FILE" disclaimer)"
  printf -- '-----------------------------------------------------------------------------\n'

  population_snapshot "population_after_live_evaluation"
  compare_snapshots "$before" "$SNAPSHOT_FILE" "live evaluation is read-only (no rows, no lifecycle run)"
  pass "STAGE N complete -- live evaluation produced no database write and no lifecycle row"
}

# ---------------------------------------------------------------------------
# 25. STAGE O -- final audit
# ---------------------------------------------------------------------------

stage_audit() {
  stage_banner "O" "Final audit"
  resolve_generation_identity
  resolve_bundle_identity
  export GENERATION_RUN_ID DATASET_VERSION BUNDLE_ID BUNDLE_VERSION

  local rc=0
  run_python "audit" <<'PYAUDIT' || rc=$?
import json, os, sys
import psycopg2.extras

import mlflow

from src.common.config import get_settings
from src.common.db import get_connection
from src.common.mlflow_setup import configure_mlflow
from src.fraud_intel.models.mlflow_naming import MODEL_COMPONENTS, registered_model_name
from src.fraud_intel.models.training import _experiment_name

channel = os.environ["DEMO_CHANNEL"]
database = os.environ["DEMO_DATABASE"]
generation_run_id = os.environ["GENERATION_RUN_ID"]
bundle_id = int(os.environ["BUNDLE_ID"])
expected_events = int(os.environ["DEMO_COUNT"])
expected_alerts = int(os.environ["EXPECTED_SOURCE_ALERT_COUNT"])
forbidden = set(os.environ["FORBIDDEN_DATABASES"].split())

problems = []
audit = {}

settings = get_settings()
conn = get_connection()
try:
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT current_database() AS db, inet_server_port() AS port")
            row = dict(cur.fetchone())
            audit["connection"] = {
                "current_database": row["db"], "server_port": row["port"],
                "settings_postgres_db": settings.postgres_db,
                "settings_postgres_test_db": settings.postgres_test_db,
                "settings_mlflow_tracking_uri": settings.mlflow_tracking_uri,
                "settings_minio_endpoint": settings.minio_endpoint,
            }
            if row["db"] != database:
                problems.append("current_database()=%r" % row["db"])

            cur.execute("SELECT datname FROM pg_database ORDER BY datname")
            datnames = [r["datname"] for r in cur.fetchall()]
            audit["cluster_databases"] = datnames
            present = sorted(forbidden.intersection(datnames))
            audit["forbidden_databases_present"] = present
            if present:
                problems.append("forbidden database(s) exist in this cluster: %r" % present)

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT event_id) AS distinct_ids "
                "FROM channel_events WHERE channel = %s", (channel,))
            audit["channel_events"] = dict(cur.fetchone())
            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT sa.source_alert_id) AS distinct_ids "
                "FROM source_alerts sa JOIN channel_events ce ON ce.event_id = sa.event_id "
                "WHERE ce.channel = %s", (channel,))
            audit["source_alerts"] = dict(cur.fetchone())
            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT sel.event_id) AS distinct_ids "
                "FROM synthetic_event_labels sel JOIN channel_events ce ON ce.event_id = sel.event_id "
                "WHERE ce.channel = %s", (channel,))
            audit["synthetic_event_labels"] = dict(cur.fetchone())

            cur.execute(
                "SELECT bundle_id, bundle_version, status, promoted_by, promoted_at, training_run_id, "
                "dataset_version FROM channel_model_bundles WHERE channel = %s ORDER BY bundle_version", (channel,))
            audit["bundles"] = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT (source_system, source_alert_id)) AS distinct_keys "
                "FROM fraud_alerts WHERE channel = %s", (channel,))
            audit["fraud_alerts"] = dict(cur.fetchone())
            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT ae.evidence_id) AS distinct_ids, "
                "count(*) FILTER (WHERE ae.degraded) AS degraded, "
                "count(DISTINCT ae.channel_model_bundle_id) AS distinct_bundles "
                "FROM alert_evidence ae JOIN fraud_alerts fa ON fa.alert_id = ae.alert_id "
                "WHERE fa.channel = %s", (channel,))
            audit["alert_evidence"] = dict(cur.fetchone())
            cur.execute(
                "SELECT count(*) AS total, count(DISTINCT la.alert_id) AS distinct_alerts "
                "FROM label_assessments la JOIN fraud_alerts fa ON fa.alert_id = la.alert_id "
                "WHERE fa.channel = %s", (channel,))
            audit["label_assessments"] = dict(cur.fetchone())

            cur.execute(
                "SELECT run_id, pipeline_name, status, trigger_source, records_processed, records_rejected, "
                "dataset_version, model_version FROM pipeline_runs ORDER BY run_id")
            audit["pipeline_runs"] = [dict(r) for r in cur.fetchall()]
finally:
    conn.close()

if audit["channel_events"]["total"] != expected_events:
    problems.append("channel_events=%d, expected %d" % (audit["channel_events"]["total"], expected_events))
if audit["channel_events"]["total"] != audit["channel_events"]["distinct_ids"]:
    problems.append("duplicate channel_events rows")
if audit["source_alerts"]["total"] != expected_alerts:
    problems.append("source_alerts=%d, expected %d" % (audit["source_alerts"]["total"], expected_alerts))
if audit["source_alerts"]["total"] != audit["source_alerts"]["distinct_ids"]:
    problems.append("duplicate source_alerts rows")
if audit["synthetic_event_labels"]["total"] != expected_events:
    problems.append("synthetic_event_labels=%d, expected %d" % (audit["synthetic_event_labels"]["total"], expected_events))

operational = [b for b in audit["bundles"] if b["status"] == "OPERATIONAL"]
if len(operational) != 1 or operational[0]["bundle_id"] != bundle_id:
    problems.append("expected exactly one OPERATIONAL bundle (%d), found %r" % (bundle_id, audit["bundles"]))

if audit["fraud_alerts"]["total"] != expected_alerts or audit["fraud_alerts"]["distinct_keys"] != expected_alerts:
    problems.append("fraud_alerts %r != one per source alert (%d)" % (audit["fraud_alerts"], expected_alerts))
if audit["alert_evidence"]["total"] != expected_alerts:
    problems.append("alert_evidence=%d, expected %d" % (audit["alert_evidence"]["total"], expected_alerts))
if audit["alert_evidence"]["degraded"]:
    problems.append("%d degraded evidence row(s)" % audit["alert_evidence"]["degraded"])
if audit["alert_evidence"]["distinct_bundles"] not in (0, 1):
    problems.append("evidence references %d bundles" % audit["alert_evidence"]["distinct_bundles"])
if audit["label_assessments"]["total"] != expected_alerts or audit["label_assessments"]["distinct_alerts"] != expected_alerts:
    problems.append("label_assessments %r != one per alert (%d)" % (audit["label_assessments"], expected_alerts))

bad_runs = [r for r in audit["pipeline_runs"] if r["status"] in ("RUNNING", "FAILED", "PENDING", "CANCELLED")]
audit["non_success_runs"] = bad_runs
if bad_runs:
    problems.append("%d pipeline_runs row(s) are not SUCCESS: %r" % (len(bad_runs), bad_runs))

expected_lifecycle = {"train": 1, "model_promotion": 1, "fraud_score": 2, "label_eligibility": 2}
actual_lifecycle = {}
for r in audit["pipeline_runs"]:
    actual_lifecycle[r["pipeline_name"]] = actual_lifecycle.get(r["pipeline_name"], 0) + 1
audit["lifecycle_counts"] = actual_lifecycle
audit["expected_lifecycle_counts"] = expected_lifecycle
if actual_lifecycle != expected_lifecycle:
    problems.append("lifecycle rows %r != expected %r (one train, one promotion, two scoring runs, two label runs; "
                    "evaluation is deliberately read-only and records none)" % (actual_lifecycle, expected_lifecycle))

configure_mlflow()
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(_experiment_name(channel))
mlflow_runs = client.search_runs([experiment.experiment_id]) if experiment else []
audit["mlflow"] = {
    "tracking_uri": mlflow.get_tracking_uri(),
    "experiment": _experiment_name(channel),
    "run_count": len(mlflow_runs),
    "run_names": sorted(r.data.tags.get("mlflow.runName", "") for r in mlflow_runs),
    "registered_models": {},
}
for component in MODEL_COMPONENTS:
    name = registered_model_name(channel, component)
    versions = client.search_model_versions("name='%s'" % name)
    audit["mlflow"]["registered_models"][name] = [
        {"version": v.version, "status": v.status, "run_id": v.run_id} for v in versions
    ]
    if len(versions) != 1:
        problems.append("%s has %d versions, expected 1" % (name, len(versions)))
    elif versions[0].status != "READY":
        problems.append("%s version %s is %r, expected READY" % (name, versions[0].version, versions[0].status))
if len(mlflow_runs) != 3:
    problems.append("expected 3 MLflow runs, found %d" % len(mlflow_runs))

audit["problems"] = problems
print(json.dumps(audit, sort_keys=True, default=str))
if problems:
    sys.stderr.write("final audit failed:\n  " + "\n  ".join(problems) + "\n")
    raise SystemExit(1)
PYAUDIT
  require_python_ok "$rc" "final audit"
  cp "$PY_JSON_FILE" "$DEMO_REPORT_DIR/audit.json"

  printf '\n--- FINAL AUDIT -------------------------------------------------------------\n'
  note "connection:          $(json_get "$PY_JSON_FILE" connection)"
  note "cluster databases:   $(json_get "$PY_JSON_FILE" cluster_databases)"
  note "channel_events:      $(json_get "$PY_JSON_FILE" channel_events)"
  note "source_alerts:       $(json_get "$PY_JSON_FILE" source_alerts)"
  note "synthetic labels:    $(json_get "$PY_JSON_FILE" synthetic_event_labels)"
  note "bundles:             $(json_get "$PY_JSON_FILE" bundles)"
  note "fraud_alerts:        $(json_get "$PY_JSON_FILE" fraud_alerts)"
  note "alert_evidence:      $(json_get "$PY_JSON_FILE" alert_evidence)"
  note "label_assessments:   $(json_get "$PY_JSON_FILE" label_assessments)"
  note "lifecycle counts:    $(json_get "$PY_JSON_FILE" lifecycle_counts)"
  note "non-SUCCESS runs:    $(json_get "$PY_JSON_FILE" non_success_runs)"
  note "MLflow:              $(json_get "$PY_JSON_FILE" mlflow)"
  printf -- '-----------------------------------------------------------------------------\n'

  prove_no_forbidden_contact
  report_git_state
  pass "STAGE O complete -- final audit clean"
}

# ---------------------------------------------------------------------------
# 26. Cleanup (isolated stack ONLY; never automatic)
# ---------------------------------------------------------------------------

do_cleanup() {
  stage_banner "--cleanup" "Tear down the isolated '$DEMO_COMPOSE_PROJECT' stack"
  require_correct_repository

  [ "$DEMO_COMPOSE_PROJECT" = "aidp-demo" ] || die "refusing to clean up compose project '$DEMO_COMPOSE_PROJECT'"

  local containers volumes
  containers="$(demo_compose ps -a --format '{{.Name}}' 2>/dev/null || true)"
  volumes="$(docker volume ls --filter "label=com.docker.compose.project=$DEMO_COMPOSE_PROJECT" --format '{{.Name}}' 2>/dev/null || true)"

  local n
  for n in $containers; do
    case "$n" in
      "${DEMO_CONTAINER_PREFIX}"*) : ;;
      *) die "refusing to clean up: project $DEMO_COMPOSE_PROJECT owns unexpected container '$n'" ;;
    esac
  done
  for n in $volumes; do
    case "$n" in
      "${DEMO_COMPOSE_PROJECT}_"*) : ;;
      *) die "refusing to clean up: unexpected volume '$n' labelled for project $DEMO_COMPOSE_PROJECT" ;;
    esac
  done

  printf 'The following ISOLATED resources will be removed:\n'
  printf '  containers: %s\n' "${containers:-<none>}"
  printf '  volumes:    %s\n' "${volumes:-<none>}"
  printf '  network:    aidp-demo-network\n'
  printf '  state file: %s\n' "$DEMO_STATE_FILE"
  printf '  diagnostic: %s\n' "$DEMO_DIAGNOSTIC_SCORES_FILE"
  printf '\nNOT touched: the shared aidp-poc stack, the aidp and aidp_test databases,\n'
  printf '             the shared MLflow backend, the shared MinIO buckets, .env,\n'
  printf '             and every log/report under %s/logs and %s/reports.\n\n' "$DEMO_WORK_DIR" "$DEMO_WORK_DIR"

  if [ "$OPT_ASSUME_YES" -ne 1 ]; then
    printf 'Proceed? [y/N] '
    local answer=""
    read -r answer || true
    case "$answer" in
      y|Y|yes|YES) : ;;
      *) info "cleanup aborted -- nothing was removed"; return 0 ;;
    esac
  fi

  demo_compose down -v --remove-orphans || die "docker compose down failed for project $DEMO_COMPOSE_PROJECT"
  rm -f "$DEMO_STATE_FILE" "$DEMO_DIAGNOSTIC_SCORES_FILE"

  local remaining
  remaining="$(docker volume ls --filter "label=com.docker.compose.project=$DEMO_COMPOSE_PROJECT" --format '{{.Name}}' 2>/dev/null || true)"
  [ -z "$remaining" ] || warn "demo volumes still present: $remaining"

  info "shared stack containers (untouched):"
  docker ps --filter "name=aidp-" --format '  {{.Names}}\t{{.Status}}' | grep -v "  ${DEMO_CONTAINER_PREFIX}" || true
  pass "isolated demo stack removed"
}

# ---------------------------------------------------------------------------
# 27. Main
# ---------------------------------------------------------------------------

stage_index() {
  local wanted="$1" i=0 s
  for s in $DEMO_STAGES; do
    if [ "$s" = "$wanted" ]; then printf '%s\n' "$i"; return 0; fi
    i=$((i + 1))
  done
  printf '%s\n' "-1"
}

run_stage() {
  case "$1" in
    preflight)          stage_preflight ;;
    generate)           stage_generate ;;
    generate_retry)     stage_generate_retry ;;
    verify)             stage_verify ;;
    split_preview)      stage_split_preview ;;
    train)              stage_train ;;
    evaluate_candidate) stage_evaluate_candidate ;;
    diagnostic)         stage_diagnostic ;;
    approval)           stage_approval ;;
    promote)            stage_promote ;;
    score)              stage_score ;;
    score_retry)        stage_score_retry ;;
    labels)             stage_labels ;;
    labels_retry)       stage_labels_retry ;;
    live_evaluate)      stage_live_evaluate ;;
    audit)              stage_audit ;;
    *) die "unknown stage '$1'" ;;
  esac
}

main() {
  printf 'AiDP Debit Card full-lifecycle demonstration\n'
  printf '  repository : %s\n' "$REPO_ROOT"
  printf '  channel    : %s   count: %s   seed: %s   reference_date: %s\n' \
    "$DEMO_CHANNEL" "$DEMO_COUNT" "$DEMO_SEED" "$DEMO_REFERENCE_DATE"
  printf '  database   : %s (isolated compose project: %s)\n' "$DEMO_DATABASE" "$DEMO_COMPOSE_PROJECT"
  printf '  logs       : %s\n' "$DEMO_LOG_DIR"
  printf '  reports    : %s\n' "$DEMO_REPORT_DIR"

  if [ "$OPT_CLEANUP" -eq 1 ]; then
    require_venv
    load_demo_env
    do_cleanup
    printf '\n[RESULT] CLEANUP COMPLETE\n'
    return 0
  fi

  local start_index=0
  if [ -n "$OPT_RESUME_FROM" ]; then
    start_index="$(stage_index "$OPT_RESUME_FROM")"
  fi

  if [ "$start_index" -le 0 ]; then
    DEMO_REQUIRE_EMPTY=1
  else
    DEMO_REQUIRE_EMPTY=0
    info "resuming from stage '$OPT_RESUME_FROM' -- stage A still runs (isolation, health and expectations are re-proved on every invocation)"
  fi
  export DEMO_REQUIRE_EMPTY

  run_stage preflight

  if [ "$OPT_PREFLIGHT_ONLY" -eq 1 ]; then
    printf '\n[RESULT] PREFLIGHT ONLY -- PASSED\n'
    printf '         logs: %s\n' "$DEMO_LOG_DIR"
    return 0
  fi

  local i=0 stage
  for stage in $DEMO_STAGES; do
    if [ "$i" -gt 0 ] && [ "$i" -ge "$start_index" ]; then
      run_stage "$stage"
    fi
    i=$((i + 1))
  done

  printf '\n===============================================================================\n'
  printf '[RESULT] DEMO PASSED -- every stage from %s through audit\n' "${OPT_RESUME_FROM:-preflight}"
  printf '  reports : %s\n' "$DEMO_REPORT_DIR"
  printf '  stdout  : %s\n' "$DEMO_STDOUT_LOG"
  printf '  stderr  : %s\n' "$DEMO_STDERR_LOG"
  printf '  The isolated stack is left RUNNING on purpose. Remove it explicitly with:\n'
  printf '    scripts/demo_debit_card_full_lifecycle.sh --cleanup\n'
  printf '===============================================================================\n'
}

main "$@"

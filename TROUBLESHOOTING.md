# Troubleshooting

Real issues actually hit while building this POC, and the fixes applied —
not a hypothetical checklist. See BUILD_LOG.md for the phase each was
found and fixed in, with full verification output.

## Docker not installed

**Symptom**: `command not found: docker`.

**Fix**: `brew install docker docker-compose colima && colima start --cpu 4 --memory 6 --disk 60`,
then point the docker CLI at the plugin dir in `~/.docker/config.json`:
```json
{"cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]}
```

## `minio/minio` / `minio/mc` pull fails ("repository does not exist")

**Symptom**: `docker compose up` fails with `pull access denied for
minio/minio`.

**Cause**: MinIO moved image distribution off Docker Hub.

**Fix**: use `quay.io/minio/minio` and `quay.io/minio/mc` (same version
tags) — already applied in `docker-compose.yml`.

## XGBoost fails to import on macOS

**Symptom**: `XGBoostError: XGBoost Library (libxgboost.dylib) could not
be loaded ... Library not loaded: @rpath/libomp.dylib`.

**Cause**: XGBoost needs the OpenMP runtime; macOS doesn't bundle it.

**Fix**: `brew install libomp`. Without it, `src/ml/train.py`
automatically and correctly falls back to `RandomForestClassifier` (logged
loudly, not silently) — so training still succeeds, just not with the
primary model.

## MLflow artifact logging fails with `InvalidAccessKeyId`

**Symptom**: `mlflow.log_artifact(...)` raises a boto3
`InvalidAccessKeyId` error even though MinIO is healthy.

**Cause**: MLflow's client uploads artifacts directly to the S3-compatible
store (MinIO), which needs `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
`MLFLOW_S3_ENDPOINT_URL` in the *client* process's environment — these
aren't implied by `MLFLOW_TRACKING_URI` alone.

**Fix**: call `src.common.mlflow_setup.configure_mlflow()` (sets these
from `.env`'s MinIO credentials) before any MLflow call, in every process
that talks to MLflow directly (training, the API if it loads models
in-process).

## `mlflow.xgboost.log_model()` raises `TypeError: missing 1 required
positional argument: 'artifact_path'`

**Cause**: this MLflow version (2.17.2) still expects `artifact_path=`,
not the newer `name=` kwarg some MLflow docs/examples use.

**Fix**: use `artifact_path="model"`.

## Streaming consumer never receives any messages

**Symptom**: `consumer.poll()` always returns `None`; `rpk group describe`
shows the consumer group with 0 members; Redpanda logs show:
```
Unsupported version 12 for fetch API
```

**Cause**: `confluent-kafka==2.6.0`'s bundled librdkafka defaults to
negotiating Kafka Fetch API v12. Redpanda doesn't implement it — confirmed
even at the newest available release (v25.2.1); `api.version.request=false`
+ `broker.version.fallback` client config did **not** suppress the v12
request either.

**Fix**: pin `confluent-kafka==2.3.0` (older librdkafka, doesn't request
v12) — already in `requirements.txt`.

**Note on Redpanda version upgrades**: Redpanda's on-disk cluster metadata
can't skip more than ~2-3 releases in one upgrade (jumping straight from
v24.2.7 to v25.2.1 crashes with `Attempted to upgrade from incompatible
logical version 13 to 16!`). If you ever need to upgrade the Redpanda
image version, step through intermediate releases one at a time
(`docker compose up -d redpanda` after each `image:` change), or accept
losing topic data by clearing the `redpanda_data` volume.

## API's `/health` stays `"degraded"` forever after a Postgres restart

**Symptom**: Postgres is confirmed healthy and reachable, but
`GET /health` keeps reporting `postgres_connected: false` until the API
process itself is restarted.

**Cause**: `psycopg2.pool.ThreadedConnectionPool` doesn't validate
connections on checkout — after Postgres restarts, the pool keeps handing
out (and getting back) a dead connection object indefinitely.

**Fix**: `src/api/main.py`'s `_checkout_validated_connection()` pings
(`SELECT 1`) every connection on checkout and discards+replaces it
(`putconn(..., close=True)`) if the ping fails.

## Streaming consumer's retries don't actually recover from a Postgres outage

**Symptom**: even after Postgres comes back up, a message being retried
keeps failing until it's exhausted its retry budget and gets DLQ'd.

**Cause**: the same class of bug as above — retrying the *same* Postgres
connection object doesn't help if that connection is already dead.

**Fix**: `src/ingestion/consumer.py`'s `_score_with_retry` (and the
`pipeline_runs` bookkeeping calls) acquire a **fresh** connection on every
retry attempt via `src.common.db.get_connection()`.

## Dashboard fails with `ModuleNotFoundError: No module named 'src'`

**Symptom**: opening http://127.0.0.1:8501 shows a traceback from
`from src.dashboard.data import ...` in `src/dashboard/app.py`.

**Cause**: `streamlit run <path>` only adds the script's *own* directory
(`src/dashboard/`) to `sys.path`, not the project root — so `src.*`
imports fail unless the root happens to already be on the path (which
depends on exactly how/where Streamlit was launched, and isn't reliable).

**Fix**: `src/dashboard/app.py` now inserts the project root onto
`sys.path` itself, at the very top of the file, before any `src.*`
import — so it works regardless of the working directory or invocation
method used to launch it. Verified by starting it from a different
working directory entirely and confirming it still loads with 0 errors.

## Port already in use

**Symptom**: `bind: address already in use` on `docker compose up`, or on
`uvicorn`/`streamlit` startup.

**Fix**:
```bash
lsof -tiTCP:<port> -sTCP:LISTEN | xargs -r kill
```
Then retry. This build already avoids the most common conflict (MLflow on
5000 clashing with macOS AirPlay Receiver — moved to 5001).

## Tests can't reach a service

Run `./scripts/healthcheck.sh` first. Integration/smoke tests need the
full stack running (`./scripts/start.sh`) and a trained model
(`./scripts/train_model.sh`) — they load the real registered model via
MLflow, same as the API does. `pytest tests/unit` never needs any of this.

## v1.2 CLI and control-plane

### `aidp` command fails with a validation error

**Symptom**: `aidp pipeline run batch ...` (or `train run`/`stream run`)
exits `2` with a message describing an invalid field, e.g. a negative
`--duration`.

**Cause**: the typed config (`BatchRunConfig`/`TrainingRunConfig`/
`StreamRunConfig`) rejects the value before any workload runs —
`StreamRunConfig.duration` requires a positive number, for example.

**Fix**: correct the flag value. This is deliberate fail-fast behavior, not
a bug — nothing is dispatched to the real pipeline/training/streaming code
until the config validates.

### `--json` output doesn't parse / looks contaminated with log lines

**Symptom**: piping `aidp ... --json` into `jq`/`json.loads` fails because
extra non-JSON text appears on stdout.

**Cause**: something wrote to stdout instead of stderr during the command.
`src/common/logging.py`'s `configure_logging()` sends both the
standard-library logger and structlog's `PrintLogger` to stderr by default
(`force=False`) — only the CLI itself calls `configure_logging("cli",
force=True)` once at startup. If you see this, first confirm you're on a
build that includes the v1.2 CLI logging fix; if you are, check whether a
third-party library your change pulled in prints directly to stdout (not
through the logger) rather than assuming the CLI itself regressed.

**Fix**: `--json` mode always emits exactly one JSON object on stdout — if
anything else appears there, treat it as a bug in whatever produced the
extra output, not something to work around downstream.

### `pipeline_runs` errors with "column ... does not exist" / "constraint ... does not exist"

**Symptom**: any workload run (via the CLI or a legacy entry point) fails
with a Postgres error naming a `pipeline_runs` column or constraint added
in migration `002` (e.g. `trigger_source`, `config_snapshot`).

**Cause**: this is an **existing** database that predates migration `002`
and hasn't had it applied yet. A fresh install gets the current schema
automatically; an existing one does not.

**Fix**: apply `infrastructure/postgres/migrations/
002_pipeline_run_provenance.sql` — see RUNBOOK.md's "Database migrations"
section for the exact, `aidp_test`-first procedure. Never run
`scripts/reset_demo.sh` to try to "fix" this — it clears data, not schema.

### `aidp model show <alias>` fails — is it "not found" or is MLflow down?

**Symptom**: `aidp model show champion --json` exits non-zero.

**Cause and how to tell which**: exit code distinguishes the two cases.
Exit `2` means MLflow's model registry specifically reported that the
model or alias doesn't exist (detected via MLflow's structured
`error_code` field, e.g. `INVALID_PARAMETER_VALUE` for a missing alias —
never by matching message text) — check the alias name and that a model
has actually been registered (`./scripts/train_model.sh` at least once).
Exit `3` means something else went wrong reaching MLflow — a connection
failure, an authentication failure, a server error — check
`./scripts/healthcheck.sh` and that MLflow is actually up.

### `git_sha` is `null` in a run record

**Symptom**: `aidp run show <id>` shows `"git_sha": null`.

**Cause**: this is a safe, deliberate fallback, not an error. `git rev-parse
HEAD` either wasn't available (no `git` binary), timed out, or the working
directory isn't a Git checkout at run time — `provenance.get_git_sha()`
returns `None` in every one of those cases rather than failing the run.

### A `stream run` / `run_consumer.sh` run stays `RUNNING` forever

**Symptom**: `aidp run show <id>` (or `run list`) shows an old streaming run
still in `RUNNING`, with no `completed_at`.

**Cause**: a graceful stop — the `--duration` timer elapsing, or an
operator's Ctrl+C — always records `SUCCESS` (see RUNBOOK.md). A run stuck
in `RUNNING` means the process ended some other way that bypasses Python's
exception handling entirely: a `SIGKILL`, a crash, or a power loss. This is
current v1.2 behavior, not an ideal production guarantee — there is no
stale-run recovery/reaping mechanism yet to detect and close out a run
whose process no longer exists.

**Fix**: there isn't an automated one yet. Treat a long-`RUNNING` row with
no corresponding live process as evidence of an abrupt termination, not a
live run.

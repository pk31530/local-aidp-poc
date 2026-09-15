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
MLflow, same as the API does.

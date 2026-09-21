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

## v1.3 fraud intelligence

Real failure modes actually hit while building the seven-channel fraud
intelligence subsystem — not a hypothetical checklist. All examples are
against synthetic `aidp_test` data only.

### MLflow not configured before client creation

**Symptom**: an opaque `MlflowException`, or MLflow silently talking to
its own default tracking URI instead of this project's, the first time a
new code path calls MLflow directly (first hit: real promotion
verification).

**Cause**: `mlflow.tracking.MlflowClient()` targets MLflow's process-wide
default tracking URI unless `configure_mlflow()`
(`src/common/mlflow_setup.py`) has been called first, in **that same
process** — importing a module that calls it elsewhere is not enough.

**Fix**: every function that talks to MLflow directly calls
`configure_mlflow()` itself, every time, rather than assuming some other
earlier call in the same process already did it (see
`_MlflowModelVersionVerifier.verify()` for the pattern).

### `FeatureNotSupported: FOR UPDATE is not allowed with aggregate functions`

**Symptom**: `channel_model_bundles.register_candidate()`'s next-version
computation fails the very first time it runs against a real database
(never caught by unit tests, which use the in-memory fake store).

**Cause**: `SELECT ... FOR UPDATE` combined with `MAX(bundle_version)` is
rejected outright by real Postgres — a session-level serialization problem
needs a session-level lock, not a row lock, and a row lock cannot
serialize a channel's very first bundle anyway (no row yet exists to
lock).

**Fix**: `pg_advisory_xact_lock(namespace, hashtext(channel))` — a
transaction-scoped advisory lock on a `(namespace, channel-hash)` key,
auto-released at commit/rollback, taken before the `MAX(bundle_version)` +
`INSERT` in the same transaction.

### Orphaned MLflow model versions after a failed/superseded training attempt

**Symptom**: `client.search_model_versions("name='...'")` shows more
versions than any current or retired `channel_model_bundles` row
references (confirmed for real: `online_banking`'s gbm/lr-shadow/anomaly
version `1` is orphaned — the channel's one real bundle row references
version `2`).

**Cause**: MLflow model-version registration and `channel_model_bundles`
row insertion are not one atomic transaction — a training run that
registers MLflow versions and then fails (or is superseded before its
candidate is ever promoted) leaves those versions registered with no
bundle ever pointing at them.

**Fix**: this is expected, not a bug to "clean up" — **never** delete or
modify an MLflow version to tidy this up; every version remains `READY`
and immutable. To audit which versions are actually live, cross-reference
`channel_model_bundles.{gbm,lr,anomaly}_model_version` (both `OPERATIONAL`
and `RETIRED` rows) against MLflow's registered versions, rather than
assuming version count == live count.

**Verified count as of Phase 7B completion**: 27 total registered
component versions across the seven channels (3 components ×
7 channels + 6 extra for `ach`/`online_banking`'s superseded/retired
history) — of which **21** are operationally referenced by the seven
current `OPERATIONAL` bundles (one `gbm`/`lr-shadow`/`anomaly` version
each), **3** are referenced by ACH's `RETIRED` bundle 3, and **3** are the
orphaned `online_banking` v1 components described above. `21` and `27`
are two different, both-correct counts answering two different
questions ("what's live right now" vs. "everything ever registered") —
never conflate them.

### Generation identity mismatch / ambiguous dataset

**Symptom**: `GenerationRunChannelMismatchError`, `GenerationRunDatasetVersionError`,
`UnknownGenerationRunError`, or `GenerationIdentityConflictError` from
`train`/`score`/`labels assess`/`evaluate`/generation itself.

**Cause**: every one of these is a deliberate refusal to silently mix
data — an unknown `generation_run_id` never generated (or generation
failed and rolled back); a `generation_run_id` that spans more than one
`dataset_version` for the channel; or the same `(channel, seed, count)`
regenerated under a **different** `reference_date`, which would silently
collide via `channel_events`' `ON CONFLICT (event_id) DO NOTHING` since
`event_id`s are deterministic per `(seed, channel)` alone, not per
`reference_date`.

**Fix**: use the exact `generation_run_id` a real `generate` call
returned; regenerate with a different `seed` if you deliberately want a
different `reference_date` for the same channel/count.

### Chronological split / purge-gap failure

**Symptom**: `InsufficientTrainingDataError` from `assign_chronological_split()`
or the partition-size/class-balance check that follows it.

**Cause**: too few distinct timestamp groups to place both the
train/calibration and calibration/test boundaries without splitting a
group, or a partition left with fewer than the minimum rows/only one
class after the (POC default: zero) purge gap is applied.

**Fix**: generate a larger population (`--count`) for that channel: this
project's own real training runs use `--count 5000`, which reliably
produces train/calibration/test splits of roughly 300/65/65 rows for
every channel actually built.

### Feature-schema mismatch

**Symptom**: a scoring or diagnostic run raises when computing features
for an event whose channel adapter doesn't match the requested
`feature_schema_version`, or a preprocessor trained on one feature set is
applied to a different one.

**Cause**: `feature_schema_version` is pinned per bundle
(`REQUIRED_OPERATIONAL_COMPONENTS`); promoting a bundle whose pinned
version no longer matches the channel adapter's current feature columns
is refused by `verify_bundle_components()`.

**Fix**: retrain the channel after a feature-adapter change — never hand-edit
`feature_schema_version` on an existing bundle row.

### Bundle compatibility / MLflow run-ID verification failure

**Symptom**: `BundleVerificationFailedError` at promotion time.

**Cause**: `verify_bundle_components()` checks every component version is
`READY` in MLflow **and**, when the candidate's own training report
recorded an expected `run_id`, that the registered version actually
points at that exact run — not merely that a version number exists under
the right name. Also checks the candidate's pinned rule/graph/ensemble/
reason-code versions still match the CURRENTLY loaded config files.

**Fix**: never promote a bundle whose pinned policy versions have drifted
from the live config — retrain (which re-pins the current versions) if a
policy file legitimately changed since the candidate was trained.

### Cold-start promotion gate failure

**Symptom**: `ColdStartPromotionGateFailedError` — a channel's first-ever
promotion attempt is refused.

**Cause**: `evaluate_cold_start_promotion_gate()` requires every split
(train/calibration/test) to have at least 5 rows of each class AND the
candidate's GBM `pr_auc` to exceed the test split's own fraud prevalence
— this is a POC demonstration floor on synthetic, held-out data, never a
production/regulatory threshold, and is always freshly recomputed from the
candidate's own immutable training report, never a cached value.

**Fix**: this is a genuine evaluation result, not a bug — do not lower the
threshold or fabricate a pass. Retrain with a larger/different population,
or accept the channel remains un-promotable in its current form.

### Missing current-bundle evidence

**Symptom**: `missing_current_bundle_evidence_count > 0` in a live
evaluation's output, or `alerts list`/`alerts show` shows `current_*`
fields as `null` for an alert that clearly exists.

**Cause**: the alert genuinely has zero `alert_evidence` rows at all for
its channel's current `OPERATIONAL` bundle — either it was never scored
under that bundle, or (rare) a catastrophic scoring failure's best-effort
evidence write also failed.

**Fix**: run `aidp fraud-intel score` for that channel/generation; this is
an explicit, documented null representation, never silently fabricated as
LOW or hidden from the queue.

### Degraded scoring

**Symptom**: `alert_evidence.degraded = true` for a real alert, with one
or more `component_statuses` entries `status=ERROR`.

**Cause**: a component (GBM/anomaly/graph) failed for that one alert —
per Phase 5 decision 6, a GBM failure floors the band at `HIGH`; an
anomaly/graph failure floors it at `MEDIUM`. The rules component and
overall scoring still complete; a degraded row is never dropped or
retried automatically.

**Fix**: inspect `component_statuses`' `error_code` for the failing
component; this project's own real Phase 7B run never hit a degraded row
(0/1,756 across four new channels), so a real one is worth investigating
as a genuine infrastructure/model issue, not routine noise.

### Stale initial-vs-current evidence confusion

**Symptom**: an analyst-facing view (or a script) shows a rescored
alert's original band/score instead of its real current one — e.g.
`--priority-band MEDIUM` returns 0 rows for a channel with real, current
MEDIUM evidence.

**Cause**: reading `fraud_alerts.initial_priority_band`/
`initial_operational_priority_score` and presenting it as current state.
Those fields are frozen at first-scoring time and never updated by a
later rescore (Phase 6 decision 2) — this exact defect was found for real
in `aidp alerts list` (fixed by sourcing "current" from the latest
`alert_evidence` row via `list_alerts_with_current_state()`, the same
contract `alerts show`/the dashboard use).

**Fix**: always read current state from `get_latest_evidence()`/
`list_alerts_with_current_state()`, never from `fraud_alerts.initial_*`
directly; use `initial_*` only where explicitly labeled historical/audit.

### Cross-channel feature-context inconsistency (train/serve skew)

**Symptom**: a real scoring run's persisted bands diverge from a
pre-promotion non-persistent diagnostic's predicted bands for the exact
same population and bundle.

**Cause**: training's supervised-population construction was, at one
point, scoped to build each row's historical feature context from ONE
channel only, while real scoring's context was already correctly
cross-channel (customer-scoped across all seven channels, per guide §15)
— a genuine train/serve skew, found for real on ACH's first full
promotion cycle.

**Fix**: `src/fraud_intel/features/history.py`'s shared selectors are now
the single source of both training's and scoring's historical context
construction — this class of skew is now structurally prevented, not just
patched for one channel. If it recurs, verify every caller of
`_build_supervised_population()`/`list_pending()`/
`load_resolved_alert_scoring_contexts()` still passes the same
cross-channel pool (`load_cross_channel_customer_pool()`).

### Duplicate / retry behavior

**Symptom**: uncertainty about whether re-running a `generate`/`train`/
`score`/`labels assess` command is safe.

**Cause/fix**: all four are designed to be safe, idempotent retries —
`generate` and `score` use real `ON CONFLICT ... DO NOTHING` constraints
(`(event_id)`; `(source_system, source_alert_id)`;
`(alert_id, score_execution_id)`); `labels assess` uses an
advisory-lock-then-compare-then-conditional-insert (`assessments_inserted=0,
assessments_unchanged=<total>` on a no-op retry); `train` always creates a
**new** candidate bundle version on every call (never a retry-safe no-op —
each real training run is a distinct, intentional action).

### Dashboard infrastructure / read failures

**Symptom**: the Fraud Intelligence tab shows an error message instead of
the queue, or `pytest tests/smoke/test_fraud_intel_dashboard.py` fails to
connect.

**Cause**: `src/dashboard/fraud_intel_tab.py` targets `aidp_test` only,
via the same `src.common.db.get_connection("aidp_test")` every fraud-intel
CLI command uses — if the local stack isn't running
(`./scripts/start.sh`) or `aidp_test` doesn't have migrations 003/004
applied, every read fails.

**Fix**: `./scripts/healthcheck.sh` first; confirm `aidp_test` has the
v1.3 tables (`\d fraud_alerts` via `docker exec ... psql -d aidp_test`).
The tab's own `load_queue()`/`load_alert_detail()` never raise past a
caught `st.error(...)` for a genuine data problem — a raw Python
traceback in the dashboard means an infrastructure/connectivity issue,
not a data issue.

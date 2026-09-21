# Runbook

## Prerequisites

- Docker (via Colima on macOS: `brew install docker docker-compose colima && colima start`)
- Python 3.10+
- macOS + XGBoost: `brew install libomp` (see TROUBLESHOOTING.md — without it, training automatically falls back to RandomForest)

## First-time setup

```bash
cd local-aidp-poc
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./scripts/bootstrap.sh      # creates .env, starts infrastructure, waits for healthy
./scripts/healthcheck.sh    # confirm: PostgreSQL, MinIO, Redpanda, MLflow all [OK]

./scripts/seed_data.sh                  # generates + loads synthetic customers/transactions
python -m src.processing.pipeline       # RAW -> CLEAN -> CURATED -> FEATURES
./scripts/train_model.sh                # trains + registers fraud-detection-model
```

Or with `make`: `make bootstrap && make seed && make train` (after activating the venv and running the pipeline once — `make` doesn't wrap the batch pipeline directly, run `python -m src.processing.pipeline` between seed and train).

## Running the platform

Each of these runs in the foreground in its own terminal (or background
with `&`):

```bash
./scripts/run_api.sh          # FastAPI on http://127.0.0.1:8000 (Swagger at /docs)
./scripts/run_dashboard.sh    # Streamlit on http://127.0.0.1:8501
./scripts/run_consumer.sh     # streaming consumer (needs the API's model, not the API itself)
```

## Unified CLI

`python -m src.cli` (or `./scripts/aidp.sh`, a thin wrapper that activates
`.venv` when present and forwards every argument) is a single command
surface over the batch/training/streaming entry points plus read-only run
and model-registry queries. The legacy entry points below (`python -m
src.processing.pipeline`, `python -m src.ml.train`, `./scripts/
run_consumer.sh`) **remain fully supported** — the CLI calls the exact same
underlying functions, just with `trigger_source="cli"` instead of
`trigger_source="legacy"`.

```bash
./scripts/aidp.sh config validate --json
./scripts/aidp.sh pipeline run batch --input data/batch/some_file.csv --json
./scripts/aidp.sh train run --features-path data/output/features/transactions.parquet --json
./scripts/aidp.sh stream run --duration 30 --json
./scripts/aidp.sh run show 42 --json
./scripts/aidp.sh run list --limit 10 --json
./scripts/aidp.sh model show champion --json
```

Human-readable output (indented text) is the default; `--json` emits
exactly one JSON object on stdout and nothing else — no logs, no banners,
no progress messages. **Every log line, in both modes, goes to stderr**,
never stdout — this is what keeps `--json` output pipeable/parseable.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Success |
| `2` | Validation/user-input error — bad CLI arguments, invalid typed configuration, `run show`/`model show` for something that doesn't exist, an out-of-range `--limit` |
| `3` | Operational failure — an unhandled exception from a real workload or infrastructure call (DB/Kafka/MinIO/MLflow unavailable, etc.) |

`aidp model show <alias>` is **read-only** — it queries MLflow's model
registry metadata (name, alias, version, model type, MLflow run id) via the
same lookup `load_champion_model()` uses, but never loads model weights and
never promotes, retrains, or otherwise modifies the registry. `aidp run
list` always applies a bounded limit (default 10, hard cap 500).

**Stopping `stream run` / `run_consumer.sh` with Ctrl+C** records the run as
`SUCCESS` — an operator-initiated stop is treated as a graceful, expected
way to end a local consumer, the same as its `--duration` timer elapsing.
This is current v1.2 behavior, not an ideal production guarantee: an abrupt
process crash, `SIGKILL`, or power loss bypasses all Python exception
handling and can leave the run recorded as `RUNNING` indefinitely — there is
no stale-run recovery/reaping mechanism yet. See `TROUBLESHOOTING.md`.

## Running the real-time demo

With the consumer already running (above), in another terminal:

```bash
./scripts/run_stream.sh --rate 10                    # runs until Ctrl+C
./scripts/run_stream.sh --rate 10 --duration 30       # runs for 30s then stops
```

`--rate` accepts 1, 10, or 100 (events/sec). See DEMO_SCRIPT.md for the
full walkthrough.

## Local URLs

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8000/docs |
| Streamlit Dashboard | http://127.0.0.1:8501 |
| MLflow | http://127.0.0.1:5001 |
| MinIO Console | http://127.0.0.1:9001 (user/pass in `.env`) |
| Redpanda Console | http://127.0.0.1:8080 |

## Health / status

```bash
./scripts/healthcheck.sh
```

Expected when everything (including the API and dashboard) is running:

```
[OK] PostgreSQL
[OK] MinIO
[OK] Redpanda
[OK] MLflow
[OK] FastAPI
[OK] Streamlit
```

## Running the tests

```bash
pytest                          # everything: unit + integration + smoke
pytest tests/unit                # no infrastructure required — safe to run anywhere, anytime
pytest tests/integration          # needs the full stack running; uses the isolated aidp_test DB + transactions-test topic
pytest tests/smoke                # the full generate->publish->consume->score->store->retrieve path
```

`tests/unit` covers the batch/training/streaming lifecycle wiring, the
typed configs, the CLI, and the provenance/lifecycle service — every
database/MLflow/Kafka call in it is a fake or a monkeypatched spy, never a
live connection. `tests/integration` and `tests/smoke` exercise the real
code paths against real infrastructure and never touch the demo database
(`aidp`) or the `transactions` topic — they use `aidp_test` and
`transactions-test` exclusively (fix H4). CI (see below) runs only
`tests/unit`; run `tests/integration`/`tests/smoke` locally, with the stack
up, before relying on a change that touches the database/Kafka/MLflow paths
directly.

## Database migrations

Fresh installs get the current `pipeline_runs` shape automatically —
`infrastructure/postgres/lib/schema.sql` is applied once, at container
first-start, to both `aidp` and `aidp_test` (see `infrastructure/postgres/
init.sql`). An **existing** database created before a given migration was
added needs that migration applied manually; it is never applied
automatically to a running database.

**Always validate against `aidp_test` first.** Never assume `aidp` should
be migrated automatically just because `aidp_test` was.

```bash
# 1. Confirm the target database and inspect its current state (read-only)
docker exec -i aidp-postgres psql -U aidp -d aidp_test \
  -c "SELECT current_database();"
docker exec -i aidp-postgres psql -U aidp -d aidp_test \
  -c "\d pipeline_runs"
docker exec -i aidp-postgres psql -U aidp -d aidp_test \
  -c "SELECT count(*) FROM pipeline_runs;"

# 2. Apply, with ON_ERROR_STOP so a failure aborts the whole transaction
#    instead of leaving a half-applied schema
docker exec -i aidp-postgres psql -v ON_ERROR_STOP=1 \
  -U aidp -d aidp_test -f - \
  < infrastructure/postgres/migrations/002_pipeline_run_provenance.sql

# 3. Verify: same row count as step 1, plus the new columns/constraints
docker exec -i aidp-postgres psql -U aidp -d aidp_test \
  -c "\d pipeline_runs"
docker exec -i aidp-postgres psql -U aidp -d aidp_test \
  -c "SELECT count(*) FROM pipeline_runs;"
```

Only after `aidp_test` is verified — and only with separate, explicit
intent — repeat the same three steps against `-d aidp`. Every migration
file under `infrastructure/postgres/migrations/` only adds columns and
widens existing `CHECK` constraints; none of them drop data or narrow a
constraint, so re-running an already-applied migration is a safe no-op.

**Never use `scripts/reset_demo.sh` as a migration mechanism.** It clears
demo *data* (rows, seed/output files, data-lake objects) — it does not
alter schema, and running it does not substitute for applying a migration.
Running it against a database that still needs a migration just leaves you
with an empty, still-out-of-date schema.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request (plus a manual `workflow_dispatch` trigger): install dependencies,
a byte-compile sanity check, `aidp config validate`, and `pytest
tests/unit`. It needs no repository secrets, starts no Docker service,
makes no network call to any AiDP infrastructure, and never trains,
registers, promotes, or applies a migration. `tests/integration` and
`tests/smoke` are intentionally **not** part of CI — they need the local
Docker stack and stay a local/manual verification step.

## Resetting demo data

```bash
./scripts/reset_demo.sh    # clears aidp's rows + data/seed + data/output + data-lake MinIO objects
./scripts/seed_data.sh     # regenerate
python -m src.processing.pipeline
./scripts/train_model.sh
```

`reset_demo.sh` does **not** touch `aidp_test`, Docker volumes/containers,
Redpanda topics, or the MLflow model registry.

## Stopping

```bash
./scripts/stop.sh    # docker compose down — containers removed, named volumes (data) preserved
```

Stop the API/dashboard/consumer/producer processes with Ctrl+C (they run
outside docker-compose).

## Resuming after a restart / reboot

```bash
cd local-aidp-poc
./scripts/start.sh          # docker compose up -d, waits for healthy
./scripts/healthcheck.sh
./scripts/run_api.sh &       # and dashboard/consumer as needed
```

All demo data (Postgres, MinIO, the MLflow model registry) persists across
`stop.sh`/`start.sh` — see BUILD_LOG.md Phase 2 for the verified proof.

## v1.3 fraud intelligence

**POC-only.** All seven channels below use synthetic, locally-generated
data. Nothing in this section constitutes Citizens Bank production or
regulatory evidence, and none of these commands ever touches `aidp` — every
one below requires an explicit `--database` (no default), and this
project's own convention is to always pass `--database aidp_test`.

### Prerequisites: migrations 003/004

Fresh installs get the current `fraud_alerts`/`alert_evidence`/
`label_assessments`/`channel_model_bundles`/`channel_events`/
`synthetic_event_labels`/`source_alerts`/`analyst_dispositions` schema
automatically. An **existing** database predating v1.3 needs migrations
`003_channel_events_and_labels.sql` and
`004_fraud_alerts_evidence_and_lifecycle.sql` applied manually — same
`aidp_test`-first procedure as migration `002` above:

```bash
docker exec -i aidp-postgres psql -U aidp -d aidp_test -c "SELECT count(*) FROM channel_events;"
docker exec -i aidp-postgres psql -v ON_ERROR_STOP=1 -U aidp -d aidp_test -f - \
  < infrastructure/postgres/migrations/003_channel_events_and_labels.sql
docker exec -i aidp-postgres psql -v ON_ERROR_STOP=1 -U aidp -d aidp_test -f - \
  < infrastructure/postgres/migrations/004_fraud_alerts_evidence_and_lifecycle.sql
docker exec -i aidp-postgres psql -U aidp -d aidp_test -c "SELECT count(*) FROM channel_events;"
```

Only after `aidp_test` is verified, and only with separate explicit
intent, repeat against `-d aidp`.

### Channels

`ach`, `atm`, `debit_card`, `mobile_deposit`, `online_banking`, `p2p`,
`wire` — every `--channel` flag below accepts exactly these seven values.

### Generation — `aidp fraud-intel generate`

```bash
python -m src.cli.__main__ fraud-intel generate \
  --channel <channel> --count 5000 --seed 42 \
  --reference-date 2026-09-21 --database aidp_test --json
```

`--reference-date` is `YYYY-MM-DD`; omit it and it defaults to today (not
reproducible across days — always pass it explicitly for a repeatable
demo). `generation_run_id`/`dataset_version` are deterministic SHA-256
fingerprints of `(channel, count, seed, reference_date,
GENERATION_SPEC_VERSION)` — re-running the **identical** command is a safe
no-op retry (`inserted_event_count=0, existing_event_count=<count>`,
identical IDs), never a duplicate or a new identity. Running the same
`(channel, seed, count)` with a **different** `reference_date` raises
`GenerationIdentityConflictError` rather than silently colliding.

### Training — `aidp fraud-intel train`

```bash
python -m src.cli.__main__ fraud-intel train \
  --channel <channel> --generation-run-id <genrun-...> --database aidp_test --json
```

Always requires `--generation-run-id` — training never silently uses
"whichever generation happens to be in the table." Registers 3 new MLflow
component versions (GBM, LR-shadow, anomaly) and one new `CANDIDATE`
`channel_model_bundles` row (never `OPERATIONAL` — promotion is always a
separate, later, manual step).

### Cold-start / live evaluation — `aidp fraud-intel evaluate`

```bash
python -m src.cli.__main__ fraud-intel evaluate \
  --channel <channel> --generation-run-id <genrun-...> \
  --capacity-mode count --capacity-value 50 --recall-target 0.8 \
  [--candidate-bundle-version N] \
  --database aidp_test --json
```

`--capacity-mode`/`--capacity-value`/`--recall-target` are always required
regardless of mode. Routing is automatic and unambiguous, chosen by
whether the channel currently has an `OPERATIONAL` bundle — **not** by
`--candidate-bundle-version`: no `OPERATIONAL` bundle yet →
`candidate_training_holdout` (evaluates the candidate's own training-time
held-out report; `--candidate-bundle-version` is then required); an
`OPERATIONAL` bundle exists → `live_resolved_alerts` (evaluates real,
scored, resolved alerts; `--candidate-bundle-version`, if given, adds an
optional in-memory shadow-candidate comparison — this never scores or
writes anything for the candidate).

### Promotion — `aidp fraud-intel promote`

```bash
python -m src.cli.__main__ fraud-intel promote \
  --channel <channel> --bundle-version <N> --promoted-by <operator-id> \
  --database aidp_test --json
```

Always manual, always requires `--promoted-by`. Verifies all nine
required components/policies against the currently-loaded config, locks
the channel's full bundle history in one transaction, retires the
previous `OPERATIONAL` bundle (if any), and promotes the candidate
atomically. A channel's first-ever promotion additionally, and always
freshly, recomputes the cold-start promotion gate from the candidate's
own immutable training report before promoting — never a cached result.

### Scoring — `aidp fraud-intel score`

```bash
python -m src.cli.__main__ fraud-intel score \
  --channel <channel> --generation-run-id <genrun-...> --database aidp_test --json
```

Scores every currently-pending source alert (no `alert_evidence` row yet
for the channel's current `OPERATIONAL` bundle) for that one
`generation_run_id`. Re-running the identical command is a safe,
idempotent retry: `pending_count=0, records_processed=0, alerts=[]` — no
duplicate `fraud_alerts`/`alert_evidence` row is ever created
(`ON CONFLICT` on `(source_system, source_alert_id)` and
`(alert_id, score_execution_id)` respectively).

### Label assessment — `aidp fraud-intel labels assess`

```bash
python -m src.cli.__main__ fraud-intel labels assess \
  --channel <channel> --generation-run-id <genrun-...> --database aidp_test --json
```

Resolves each generation-scoped alert's synthetic ground truth into a
`label_assessments` row (`RESOLVED_FRAUD`/`RESOLVED_LEGITIMATE`,
`label_source=SYNTHETIC_GENERATOR`, `source_disposition_id=NULL` for the
synthetic case). Idempotent retry: `assessments_inserted=0,
assessments_unchanged=<total>`.

### Read-only model registry — `aidp fraud-intel model show`

```bash
python -m src.cli.__main__ fraud-intel model show --channel <channel> --database aidp_test --json
```

### Analyst alert queue — `aidp alerts list` / `aidp alerts show` / `aidp alerts disposition`

```bash
python -m src.cli.__main__ alerts list \
  [--channel <channel>] [--status OPEN|IN_REVIEW|CLOSED] \
  [--priority-band LOW|MEDIUM|HIGH] --database aidp_test --json

python -m src.cli.__main__ alerts show <alert_id> --database aidp_test --json

python -m src.cli.__main__ alerts disposition <alert_id> \
  --analyst-id <id> --disposition CONFIRMED_FRAUD|CONFIRMED_LEGITIMATE|NEEDS_MORE_INFO|ESCALATED \
  [--notes "..."] --database aidp_test --json
```

**Current-state semantics**: `--priority-band` filters on the alert's
**current** (latest-evidence) band, never the frozen first-scoring
snapshot — `alerts list`'s output distinguishes `initial_priority_band`/
`initial_operational_priority_score` (historical, frozen) from
`current_priority_band`/`current_operational_priority_score`/
`current_evidence_id`/`current_channel_model_bundle_id`/
`current_scored_at` (live, sourced from the latest `alert_evidence` row).
`alerts show` returns the same alert plus its full `latest_evidence` row.
Disposition capture (`alerts disposition`) is the **only** write path for
analyst review state anywhere in this subsystem — the dashboard (below)
never writes one.

### Dashboard — Fraud Intelligence tab

```bash
./scripts/run_dashboard.sh    # same command as v1.1/v1.2 — one dashboard process, six tabs total
```

Open http://127.0.0.1:8501 and select the **Fraud Intelligence** tab.
Entirely read-only: a ranked, filterable current-state alert queue (rank
by current operational priority score, channel filter, priority-band
filter, LOW alerts visible by default) and a per-alert detail view (current
score/band, reason codes, component contributions/statuses, `score_execution_id`,
pinned bundle/policy provenance, and disposition history — analyst
disposition capture remains CLI-only, via `aidp alerts disposition` above).
Reads `aidp_test` only; never `aidp`.

## Make targets

```
make bootstrap   make start   make stop   make health
make seed        make train   make demo    make test
make api         make dashboard   make consumer   make reset
```

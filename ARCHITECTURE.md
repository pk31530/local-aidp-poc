# Architecture

A fully local implementation of an enterprise AI/Data platform pattern,
demonstrated on one use case: real-time payment transaction fraud
detection. Two pipelines share one feature-computation core.

## The two pipelines

```
Historical Synthetic Data                Live Transaction
        |                                        |
        v                                        v
      RAW  <───────────────────── validated ─────┘
        |
        v
      CLEAN  (dedupe, timestamp normalize, business-rule reject)
        |
        v
   CURATED + FEATURES  <──────┐
        |                     │  src/common/features.py
        v                     │  compute_features() — ONE function,
    Model Training            │  two adapters:
        |                     │    - batch: src/processing/enrich.py
        v                     │      (reads customers.parquet + prior
     MLflow (Postgres         │       rows + failed-attempts file)
     backend store +          │    - real-time: src/common/feature_store.py
     MinIO artifact store)    │      (live SQL against customer_profiles
        |                     │       + recent_events)
        v                     │
  Registered Model  ──────────┘
        |
        v
  src/common/scoring.py: score_and_persist()
   (used identically by the API's /score handler AND the streaming
    consumer — one scoring/persist path, two callers)
        |
        v
  src/decisioning/engine.py (thresholds + reason codes)
        |
        v
  PostgreSQL (transactions, fraud_scores, fraud_decisions, recent_events)
        |
        v
  Streamlit Dashboard  /  FastAPI GET endpoints
```

**Batch path**: `src/generator/seed.py` → `src/processing/{raw,clean,enrich,risk_lookups,views,pipeline}.py`
→ `src/ml/train.py` → MLflow.

**Real-time path**: `src/generator/stream_transactions.py` → Redpanda
(`transactions` topic) → `src/ingestion/consumer.py` → PostgreSQL/MinIO.
`src/api/main.py` (`POST /score`) is the synchronous equivalent of the
consumer for on-demand scoring.

Both paths call the exact same `compute_features()` and
`score_and_persist()` — this is deliberate, not incidental. See fixes C1/C2
below.

## Local technology stack

| Capability | Technology | Why |
|---|---|---|
| Containers | Docker Compose (via Colima on macOS) | No cloud dependency |
| Event streaming | Redpanda | Kafka-API-compatible, single binary |
| Object storage | MinIO | S3-compatible data lake |
| Operational DB + online feature store | PostgreSQL | `customer_profiles`/`recent_events` queried live |
| Data lake format | Parquet | Columnar, DuckDB/Polars-native |
| Batch processing | Polars | Fast, typed, Arrow-native |
| Local analytics | DuckDB | Zero-setup SQL over Parquet |
| ML model | XGBoost (RandomForest fallback) | Gradient boosting, imbalanced-class support |
| Experiment tracking + registry | MLflow | Backed by Postgres + MinIO, survives restarts |
| Serving API | FastAPI + Uvicorn | Async-capable, OpenAPI docs free |
| Dashboard | Streamlit | Fast to build, Python-native |
| Testing | pytest | Unit + integration + smoke |
| Synthetic data | Faker + NumPy | Seeded, reproducible |
| Validation | Pydantic | Shared schema, type + bounds checks |

## Repository layout

```
config/            settings.yaml, fraud_rules.yaml — non-secret tunables
infrastructure/    postgres/ (schema + init), mlflow/ (Dockerfile)
src/common/        shared code: config, logging, schemas, features (C2),
                   feature_store (C1), scoring, retry, db, storage, timeutil
src/generator/     synthetic data (batch + streaming) generators
src/processing/    RAW -> CLEAN -> CURATED -> FEATURES batch pipeline
src/ml/            model training
src/decisioning/   thresholds + reason codes
src/api/           FastAPI service
src/ingestion/      streaming consumer
src/dashboard/     Streamlit app
scripts/           bootstrap/start/stop/seed/train/stream/consumer/api/
                   dashboard/healthcheck/reset — see RUNBOOK.md
tests/             unit/ integration/ smoke/
```

## Design deviations from the original guide (the fixes list)

The build was scoped against a fixes list that overrides the base guide
wherever they conflict. Each is implemented and independently verified at
runtime — see BUILD_LOG.md for the exact commands and observed output per
phase.

| Fix | What | Where |
|---|---|---|
| C1 | Real online feature/profile store, queried live at scoring time | `postgres/customer_profiles` + `recent_events`; `src/common/feature_store.py` |
| C2 | One shared feature-transformation module, used identically by batch and real-time | `src/common/features.py` (`compute_features`); adapters in `src/processing/enrich.py` and `src/common/feature_store.py`; scoring itself shared via `src/common/scoring.py` |
| C3 | Default-profile fallback + pre-seeded demo customers | `DEFAULT_PROFILE` in `features.py`; `demo_customer_row()` in `src/generator/customers.py` guarantees `C101`/`C1001` exist |
| C4 | MLflow on host port 5001, not 5000 | `docker-compose.yml` |
| C5 | Persistent MLflow backend store (Postgres) + artifact store (MinIO) | `docker-compose.yml`; proven via a real container teardown/recreate in Phase 2 |
| C6 | `depends_on: condition: service_healthy` everywhere | `docker-compose.yml` |
| H1 | All services bound to 127.0.0.1 | `docker-compose.yml`, `.env.example`, every `scripts/run_*.sh` |
| H2 | 503 (not 500) when the model isn't loaded; health reflects it | `src/api/main.py` — proven via a real MLflow outage in Phase 6 |
| H3 | Retry-with-backoff for transient failures, separate from the DLQ path | `src/common/retry.py`, `src/ingestion/consumer.py` — proven via a real Postgres outage in Phase 7 |
| H4 | Isolated test database (`aidp_test`) + test topic (`transactions-test`) | `infrastructure/postgres/init.sql`; `tests/integration/`, `tests/smoke/` — isolation independently verified by direct query in Phase 9 |
| H5 | Default seed size 50,000 rows, not 250,000 | `config/settings.yaml`, `.env.example` |
| M1 | Structured logging with transaction-id correlation | `src/common/logging.py` |
| M2 | Timezone pinned (Asia/Kolkata), night flag tested against it | `src/common/timeutil.py`; explicit test in `tests/unit/test_features.py` |
| M3 | Message schema versioning, enforced | `schema_version` field on every message, `src/common/schemas.py`. Originally informational-only; enforcement (unsupported versions rejected and DLQ'd) landed in the v1.1 hardening pass — see `HARDENING_LOG.md` Fix 1. |
| M4 | Pydantic bounds validation on top of type validation | `src/common/schemas.py` (`MAX_REASONABLE_AMOUNT`), `src/api/schemas.py` |

## v1.2 control plane

Built on `feat/aidp-v1-2-control-plane` on top of the v1.1 build, without
importing distributed-GPU complexity. See
`AIDP_V1_2_CONTROL_PLANE_BUILD_GUIDE.md` for the full, phase-by-phase build
history. Three capabilities, all implemented:

### Typed run configuration

`src/control_plane/config.py` defines `BatchRunConfig`, `TrainingRunConfig`,
and `StreamRunConfig` — Pydantic models describing one invocation of the
batch pipeline, training, or the streaming consumer, plus its safe
overrides. Platform-level settings (thresholds, generation defaults,
timezone, connection info) remain sourced from `src/common/config.py` and
`config/*.yaml`; these models never duplicate them. Every model rejects
unknown fields (`extra="forbid"`) and provides:

- `redacted_snapshot()` — a deterministic, JSON-serialisable dict with any
  key matching `password`/`secret`/`token`/`key`/`credential`/`dsn`/`url`
  (case-insensitive) replaced with a redaction marker.
- `config_hash()` — a stable SHA-256 hex digest of that snapshot's
  canonical JSON, so two runs with identical configuration hash identically.

### Central run lifecycle and provenance

`src/control_plane/runs.py`'s `RunLifecycle` is the **one** implementation
that writes to the `pipeline_runs` table — it replaces three separate,
inconsistent write paths that existed independently in
`src/processing/pipeline.py`, `src/ingestion/consumer.py`, and (previously
nonexistent, now added) `src/ml/train.py`.

State machine:

```
PENDING --> RUNNING --> SUCCESS
   |            |
   |            +----> FAILED
   |            |
   +----> FAILED
   +----> CANCELLED       RUNNING --> CANCELLED
```

`SUCCESS`, `FAILED`, and `CANCELLED` are terminal — no further transition is
possible once reached. `PENDING -> SUCCESS` is deliberately unreachable: a
run must pass through `RUNNING` before it can succeed. Every terminal write
is an atomic compare-and-set; calling `succeed()`/`fail()`/`cancel()` twice
with the *same* target status is a safe no-op that returns the original,
unmutated row rather than re-applying (possibly different) arguments from
the second call.

`RunLifecycle.fail_from_exception(run_id, exc)` is the mechanism that
guarantees a provenance-write failure (e.g. the database is briefly down)
can never hide the *original* workload exception — it logs the secondary
failure and returns `None` instead of raising, so a caller's
`except Exception as exc: lifecycle.fail_from_exception(run_id, exc); raise`
always re-propagates the real error.

Each `pipeline_runs` row records, where applicable:

| Field | Meaning |
|---|---|
| `pipeline_name` | `batch`, `train`, or `stream` |
| `status` | `PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, `CANCELLED` |
| `trigger_source` | `cli`, `legacy`, `github_actions`, `api`, or `test` |
| `git_sha` | Current commit SHA, or `null` if unavailable (see below) |
| `config_snapshot` / `config_hash` | The redacted typed config and its hash |
| `dataset_version` | A short SHA-256 fingerprint of the input file (batch: raw CSV; training: features parquet) — `null` for streaming, which has no fixed input file |
| `model_version` | The registered MLflow model version, once known |
| `records_processed` / `records_rejected` | Counts |
| `artifacts` | Non-secret references only — local output paths, the MLflow run id, the raw-events bucket name — never credentials |
| `error_type` / `error_message` | Set on `FAILED`; the message is redacted and capped at 2000 characters |
| `heartbeat_at` | Streaming only — updated periodically during the poll loop |

**Git SHA fallback:** `src/control_plane/provenance.get_git_sha()` runs
`git rev-parse HEAD` with a short timeout and never raises — a missing
`git` binary, a non-checkout working directory, or a timeout all resolve to
`git_sha = None` rather than failing the run. This is a deliberate
best-effort fallback, not an error condition.

**Credential redaction is defense-in-depth, not a guarantee.** Two
independent mechanisms cover different shapes of leak:
`config.redact_secret_keys()` redacts dict *values* whose *key* matches a
known sensitive-term list (`password`, `secret`, `token`, `key`,
`credential`, `dsn`, `url`); `provenance.redact_credentials()` scrubs
`scheme://user:pass@host` userinfo and the values of sensitive-named query
parameters (`token`, `key`, `password`, `secret`, `credential`) out of free
text such as exception messages. Both are pattern-based safety nets over a
known, enumerated set of shapes — neither can guarantee that every possible
secret, in every possible format, is caught. Treat any provenance data as
"probably safe to share," not as a hard security boundary.

Fresh installs receive this full `pipeline_runs` shape directly via
`infrastructure/postgres/lib/schema.sql`. An **existing** database created
before v1.2 needs migration `002_pipeline_run_provenance.sql` applied
manually — see `RUNBOOK.md`'s "Database migrations" section for the exact,
`aidp_test`-first procedure. Migration `002` only adds columns and widens
existing `CHECK` constraints; it never drops or renames anything and
preserves every existing row.

### Unified CLI

`python -m src.cli` (or `./scripts/aidp.sh`, a thin wrapper) — one command
surface over the pipeline/train/stream entry points, plus read-only run and
model-registry queries. Every workload command builds a typed config and
delegates to the same `*_configured()` function the legacy entry points
use — `run_pipeline_configured()`, `train_configured()`, `run_configured()`
— with `trigger_source="cli"` (legacy direct calls use `trigger_source=
"legacy"`); no pipeline/training/streaming/model-registry/SQL logic is
duplicated in the CLI layer. See `RUNBOOK.md` for the full command
reference, output modes, and exit-code table.

### Not included in v1.2

Carried forward from the build guide's original scope boundary — none of
this exists, and nothing above should be read as implying it does:
multi-node/GPU orchestration, SSH provisioning, remote execution, automatic
model promotion (registering and aliasing `champion` remains a manual,
explicit `train`/`aidp train run` action — there is no drift detection,
approval workflow, or auto-promotion), stage checkpoint/resume, a
web-based control-plane UI, and destructive database reset via anything
other than `scripts/reset_demo.sh`'s existing, already-documented scope.
This remains a local proof-of-concept for demonstrating patterns — it makes
no claim of regulatory approval or production readiness.

### Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request (plus manual `workflow_dispatch`): install dependencies, a
byte-compile sanity check, `aidp config validate`, and `pytest tests/unit`.
It never starts Docker, never touches a database, never applies a
migration, and never trains or promotes a model — see RUNBOOK.md's
"Continuous integration" section.

## Real issues hit and fixed during the build

See TROUBLESHOOTING.md for the full detail; summary:
- `minio/minio` and `minio/mc` no longer resolve on Docker Hub — moved to `quay.io`.
- XGBoost needs `libomp` on macOS (not bundled) — documented, with an automatic RandomForest fallback if missing.
- MLflow's S3 client needs explicit credentials in every process that talks to it directly — centralized in `src/common/mlflow_setup.py`.
- Redpanda (even at the newest available release) doesn't support Kafka Fetch API v12, which newer `confluent-kafka`/librdkafka versions request by default — pinned `confluent-kafka==2.3.0`.
- A naive retry-with-backoff implementation that reuses one Postgres connection across retries doesn't actually recover from an outage — fixed to acquire a fresh connection per attempt.
- Pooled API connections can go stale after a Postgres restart — fixed with checkout-time validation and self-healing in `src/api/main.py`.

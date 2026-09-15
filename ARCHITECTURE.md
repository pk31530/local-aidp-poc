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
| M3 | Message schema versioning (informational) | `schema_version` field on every message, `src/common/schemas.py` |
| M4 | Pydantic bounds validation on top of type validation | `src/common/schemas.py` (`MAX_REASONABLE_AMOUNT`), `src/api/schemas.py` |

## Real issues hit and fixed during the build

See TROUBLESHOOTING.md for the full detail; summary:
- `minio/minio` and `minio/mc` no longer resolve on Docker Hub — moved to `quay.io`.
- XGBoost needs `libomp` on macOS (not bundled) — documented, with an automatic RandomForest fallback if missing.
- MLflow's S3 client needs explicit credentials in every process that talks to it directly — centralized in `src/common/mlflow_setup.py`.
- Redpanda (even at the newest available release) doesn't support Kafka Fetch API v12, which newer `confluent-kafka`/librdkafka versions request by default — pinned `confluent-kafka==2.3.0`.
- A naive retry-with-backoff implementation that reuses one Postgres connection across retries doesn't actually recover from an outage — fixed to acquire a fresh connection per attempt.
- Pooled API connections can go stale after a Postgres restart — fixed with checkout-time validation and self-healing in `src/api/main.py`.

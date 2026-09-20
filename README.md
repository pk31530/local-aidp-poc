# Local AiDP POC — Real-Time Fraud Detection

A fully local, zero-cloud-cost implementation of an enterprise AI/Data
platform architecture, demonstrated end to end on one use case: real-time
payment transaction fraud detection.

Everything runs on a laptop via Docker Compose — streaming, object storage,
a data lake, model training with a registry, an inference API, and a live
dashboard — with no cloud account and no paid services.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Docker Compose](https://img.shields.io/badge/docker--compose-required-blue)
![Tests](https://img.shields.io/badge/unit%20tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Status: complete, plus v1.1 hardening and a v1.2 control-plane pass.** All
10 build phases finished and runtime-verified — see `BUILD_LOG.md` for the
full history (what was built, what broke, how it was fixed, and the exact
verification commands run). A hardening pass found and fixed 5 correctness
issues (schema-version validation, model-flavor-aware loading, pipeline
failure recording, event-replay idempotency, and training/serving target
leakage) — see `HARDENING_LOG.md`. v1.2 added typed run configuration, a
central run-lifecycle/provenance service, and a unified `aidp` CLI — see
`ARCHITECTURE.md`'s "v1.2 control plane" section and `RUNBOOK.md` for the
command reference. `CHECKPOINT.md` has the current state.

## What this demonstrates

- **One feature-computation core, two pipelines.** `compute_features()` in
  `src/common/features.py` is called by both the batch path and the
  real-time path through two thin adapters — so training-time and
  serving-time features cannot silently drift apart.
- **One scoring path, two callers.** `score_and_persist()` in
  `src/common/scoring.py` is used identically by the API's `/score` handler
  and the streaming consumer.
- **A real medallion data lake** — RAW → CLEAN → CURATED → FEATURES as
  Parquet, queried with Polars and DuckDB.
- **A model registry** — MLflow backed by PostgreSQL for metadata and MinIO
  for artifacts, with models registered and loaded by name.
- **Explainable decisions** — a threshold-based decision engine that emits
  reason codes, not just a probability.

## Pipelines

```
Synthetic Transaction -> Redpanda -> Consumer -> Feature Engineering
  -> XGBoost -> Fraud Probability -> Decision Engine -> PostgreSQL
  -> FastAPI -> Streamlit Dashboard
```

plus a batch path:

```
Historical Data -> RAW -> CLEAN -> CURATED -> FEATURES
  -> Training -> MLflow -> Registered Model
```

See `ARCHITECTURE.md` for the full diagram and the design decisions behind it.

## Stack

| Layer | Technology |
|---|---|
| Orchestration | Docker Compose |
| Streaming | Redpanda (Kafka API) |
| Object storage | MinIO (S3 API) |
| Operational DB + online feature store | PostgreSQL |
| Batch data lake | Parquet + Polars + DuckDB |
| ML + registry | XGBoost, scikit-learn, MLflow |
| Serving | FastAPI |
| Dashboard | Streamlit |

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- macOS only: `brew install libomp` (XGBoost needs the OpenMP runtime, which
  isn't bundled). Without it, `src/ml/train.py` automatically falls back to
  `RandomForestClassifier` rather than crashing — see `TROUBLESHOOTING.md`.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./scripts/bootstrap.sh                  # creates .env, starts infrastructure
./scripts/healthcheck.sh                # verify all services are healthy

./scripts/seed_data.sh                  # generate + load synthetic data
python -m src.processing.pipeline       # RAW -> CLEAN -> CURATED -> FEATURES
./scripts/train_model.sh                # train + register the model

./scripts/run_api.sh &                  # FastAPI on :8000
./scripts/run_dashboard.sh &            # Streamlit on :8501
```

Equivalent `make` targets are available: `bootstrap`, `start`, `stop`,
`health`, `seed`, `train`, `demo`, `api`, `dashboard`, `consumer`, `test`,
`reset`.

A unified CLI (`./scripts/aidp.sh` or `python -m src.cli`) is also
available as an alternative to the commands above — the legacy commands
remain fully supported either way:

```bash
./scripts/aidp.sh config validate --json
./scripts/aidp.sh pipeline run batch --json
./scripts/aidp.sh train run --json
./scripts/aidp.sh run list --limit 10 --json
```

See `RUNBOOK.md`'s "Unified CLI" section for the full command reference,
output modes, and exit codes.

Then walk through `DEMO_SCRIPT.md`, or see `RUNBOOK.md` for the full
operational reference (stopping, resetting, resuming, running the real-time
streaming demo, running tests).

## Local URLs

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8000/docs |
| Streamlit Dashboard | http://127.0.0.1:8501 |
| MLflow | http://127.0.0.1:5001 |
| MinIO Console | http://127.0.0.1:9001 |
| Redpanda Console | http://127.0.0.1:8080 |

All ports are bound to `127.0.0.1` only (not exposed on the network).

## Repo layout

```
src/
  common/         config, db, storage, logging, retry, timeutil,
                  features.py (shared feature core), feature_store.py
                  (real-time adapter), scoring.py (shared scoring path)
  control_plane/  v1.2: typed run configuration, run lifecycle/provenance
  cli/            v1.2: unified `aidp` CLI (python -m src.cli)
  generator/      synthetic customers, history, and live transaction stream
  ingestion/      Redpanda consumer
  processing/     raw -> clean -> enrich -> pipeline, risk lookups, views
  ml/             training + MLflow registration
  decisioning/    threshold + reason-code engine
  api/            FastAPI app and schemas
  dashboard/      Streamlit app
tests/            unit, integration, smoke
scripts/          bootstrap, health, seed, train, run-*, aidp, reset, start/stop
infrastructure/   container and service configuration
config/           application configuration
.github/          CI workflow (unit tests only, no infrastructure)
```

## Tests

```bash
pytest tests/unit          # no infrastructure required — what CI runs
pytest                     # everything: unit + integration + smoke
```

`tests/unit` needs nothing beyond Python and the repo itself — every
database/MLflow/Kafka call in it is a fake or monkeypatched spy. Integration
and smoke tests use the full local Docker stack, and an isolated database
(`aidp_test`) and Redpanda topic (`transactions-test`) — never the demo
data. See `RUNBOOK.md` for the full breakdown and the CI workflow.

## Documentation

| File | What's in it |
|---|---|
| `ARCHITECTURE.md` | The two pipelines, tech stack rationale, repo layout, the full fixes-list (design deviations from the original spec) |
| `RUNBOOK.md` | Every operational command: setup, running, testing, resetting, stopping, resuming |
| `DEMO_SCRIPT.md` | A repeatable walkthrough of the guide's worked fraud scenario |
| `TROUBLESHOOTING.md` | Real issues hit while building this, and their fixes |
| `BUILD_LOG.md` | Full phase-by-phase build history with verification output |
| `HARDENING_LOG.md` | The v1.1 post-completion hardening pass: 5 fixes, each with problem/fix/verification |
| `AIDP_V1_2_CONTROL_PLANE_BUILD_GUIDE.md` | The v1.2 control-plane build plan: typed config, run lifecycle/provenance, unified CLI, CI |
| `CHECKPOINT.md` | Current build state (for resuming a session) |

## A note on data, credentials, and scope

All data in this project is **synthetic**, generated by `src/generator/`.
No real transaction, customer, or payment data is used anywhere.

Credentials live in `.env`, which is git-ignored; `.env.example` is the
template. The example values are local-development defaults for throwaway
containers on `127.0.0.1` — they are not secrets and must not be reused
outside this local POC.

This is a local proof-of-concept for demonstrating architecture patterns —
it makes no claim of regulatory approval or production readiness. Model
promotion (registering a new version and aliasing it `champion`) is a
manual, explicit action (`./scripts/train_model.sh` or `aidp train run`);
there is no automatic promotion, drift detection, or approval workflow.
Provenance redaction (see `ARCHITECTURE.md`) is a pattern-based
defense-in-depth measure over a known set of sensitive-value shapes, not a
guarantee that every possible secret is caught.

## License

MIT — see `LICENSE`.

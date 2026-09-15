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
pytest tests/unit                # fast, no live dependencies beyond Postgres/MLflow config reads
pytest tests/integration          # needs the full stack running; uses the isolated aidp_test DB + transactions-test topic
pytest tests/smoke                # the full generate->publish->consume->score->store->retrieve path
```

Integration and smoke tests never touch the demo database (`aidp`) or the
`transactions` topic — they use `aidp_test` and `transactions-test`
exclusively (fix H4).

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

## Make targets

```
make bootstrap   make start   make stop   make health
make seed        make train   make demo    make test
make api         make dashboard   make consumer   make reset
```

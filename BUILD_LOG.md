# Build Log

## Fixes list (binding, overrides the guide where they conflict)

Before building, the user supplied a fixes list applied on top of
`LOCAL_AIDP_POC_FINAL_GUIDE.md`. It is treated as binding for the rest of
this build:

- **C1** Real online feature/profile store: Postgres `customer_profiles` +
  `recent_events` tables, queried live at scoring time.
- **C2** One shared `src/common/features.py`, used identically by batch and
  real-time serving. No duplicated feature logic.
- **C3** Default-profile fallback for unknown/new customers; demo customers
  pre-seeded so the demo can't cold-start crash.
- **C4** MLflow on host port 5001, not 5000 (macOS AirPlay conflict).
- **C5** MLflow backend store + MinIO artifact store on persistent Docker
  volumes.
- **C6** `depends_on: condition: service_healthy` for all service
  dependencies in docker-compose.
- **H1** All services bind to 127.0.0.1, not 0.0.0.0.
- **H2** API returns 503 (not 500) when model isn't loaded; health check
  reflects model-loaded state.
- **H3** Consumer retry-with-backoff for transient failures, separate from
  the DLQ policy for malformed messages.
- **H4** Separate test database/schema (`aidp_test`) and test topic
  (`transactions-test`).
- **H5** Default demo seed = 50,000 rows, not 250,000.
- **M1** Structured logging with a transaction-id correlation key threaded
  generator → consumer → API → DB.
- **M2** Single timezone (`Asia/Kolkata`) pinned in config; explicitly tested
  against `night_transaction_flag`.
- **M3** Message schema/versioning noted for Redpanda (informational only,
  not enforced in v1).
- **M4** Pydantic bounds validation (amount > 0, sane max) on top of type
  validation.

---

## Phase 0 — Environment Validation

Status: **COMPLETE**

Checks:
- Python: OK (3.10.8, /opt/homebrew/bin/python3)
- Node.js: OK (v24.21.0, npm 11.19.0)
- Docker: **initially missing** (no Docker Desktop, no Colima, no Podman)
- Disk: OK (589 GB free)
- CPU: OK (8 cores)
- RAM: 16 GB (at the guide's stated minimum, not preferred 24 GB — informed
  the 50,000-row seed default, fix H5)
- Folder permissions: OK

Issue found:
- `docker` command not found; no container runtime present.

Resolution (user confirmed via question, chose Colima):
- `brew install docker docker-compose colima`
- Configured `~/.docker/config.json` with `cliPluginsExtraDirs` so the
  Homebrew `docker-compose` plugin is discovered by the `docker` CLI as
  `docker compose`.
- `colima start --cpu 4 --memory 6 --disk 60` — started a Docker-API VM.
- Verified: `docker version` (server 29.5.2), `docker compose version`
  (5.5.1), `docker run --rm hello-world` succeeded.
- Verified all target ports free: 8000, 8501, 5001, 5432, 9000, 9001, 8080,
  9092.

Commands executed:
- `python3 --version`, `node --version`, `npm --version`
- `brew install docker docker-compose colima`
- `colima start --cpu 4 --memory 6 --disk 60`
- `docker run --rm hello-world`
- `lsof -iTCP:<port> -sTCP:LISTEN` for each required port

---

## Phase 1 — Project Skeleton

Status: **COMPLETE**

Created:
- Full repo structure (`config/`, `data/`, `infrastructure/`, `src/*`,
  `scripts/`, `tests/*`).
- `docker-compose.yml`: postgres, minio (+ one-shot `createbuckets`),
  redpanda, redpanda-console (+ one-shot `create-topics`), mlflow (custom
  Dockerfile). All service-to-service dependencies use
  `depends_on: condition: service_healthy` or
  `condition: service_completed_successfully` (fix C6). All host port
  bindings use `127.0.0.1:` (fix H1). Named volumes `postgres_data`,
  `minio_data`, `redpanda_data` persist state (fix C5). MLflow host port is
  5001 (fix C4); backend store is Postgres db `mlflow`, artifact store is
  MinIO bucket `mlflow-artifacts`, both on the persistent volumes.
- `infrastructure/postgres/init.sql` + `infrastructure/postgres/lib/schema.sql`:
  creates `mlflow` and `aidp_test` databases (fix H4) and applies the same
  schema to both `aidp` and `aidp_test`. Schema includes the online feature
  store tables `customer_profiles` and `recent_events` (fix C1), plus
  `customers`, `transactions`, `fraud_scores`, `fraud_decisions`,
  `model_versions`, `pipeline_runs` from the base guide.
- `infrastructure/mlflow/Dockerfile`: python:3.11-slim + mlflow +
  psycopg2-binary + boto3, serves with `--serve-artifacts` against the
  Postgres backend store and MinIO (S3-compatible) artifact store.
- `config/settings.yaml`: timezone (fix M2), generation defaults (50,000
  rows, fix H5), velocity windows, streaming/data-lake config, retry policy
  parameters (fix H3), schema_version note (fix M3).
- `config/fraud_rules.yaml`: decision thresholds, reason codes, night window
  (evaluated in the pinned timezone, fix M2).
- `.env.example` / `.env`: all values marked local-dev-only; `.env` is
  git-ignored.
- `requirements.txt`: pinned versions for all stack components.
- `src/common/config.py`: `pydantic-settings`-based `Settings`, plus YAML
  loaders for `settings.yaml`/`fraud_rules.yaml`.
- `src/common/logging.py`: `structlog`-based JSON logging with a
  transaction-id contextvar (fix M1) — `configure_logging()` /
  `bind_transaction_id()` / `get_logger()`.
- `tests/unit/test_config.py`: 4 tests covering settings defaults, DSNs,
  YAML-loaded generation defaults (asserts 50,000 row default), and
  threshold ordering.
- `scripts/bootstrap.sh`, `scripts/start.sh`, `scripts/stop.sh`,
  `scripts/healthcheck.sh` (the latter already checks all 6 target
  services — Postgres, MinIO, Redpanda, MLflow, FastAPI, Streamlit — the
  last two will correctly [FAIL] until Phase 6/8 exist).
- `README.md` (placeholder, finalized in Phase 10), `.gitignore`.

Verification executed (not just file creation):
- `docker compose config` — **exit 0**, valid merged config.
- `python3 -m venv .venv && pip install -r requirements.txt` — **succeeded**,
  no dependency resolution errors.
- `python -m pytest tests/unit/test_config.py -v` — **4 passed**.

Issues: none blocking.

Deviation from guide: `src/common/features.py` (fix C2, the shared
feature-transformation module) is intentionally NOT created yet — it has no
real implementation until Phase 4 (batch pipeline) defines the feature
logic it will share with Phase 6/7 (real-time serving). Creating it earlier
would mean a half-finished stub. Tracked as required before Phase 4 is
considered done.

---

## Phase 2 — Infrastructure

Status: **COMPLETE**

Issue found on first `docker compose up -d`:
- `minio/minio` and `minio/mc` images no longer resolve on Docker Hub
  (`pull access denied ... repository does not exist`) — MinIO moved its
  image distribution to `quay.io/minio/minio` and `quay.io/minio/mc`.
  Fixed by repointing both `minio` and `createbuckets` services in
  `docker-compose.yml` to the `quay.io` images (same version tags). Verified
  both tags pull successfully before retrying.

Verification executed:
- `docker compose up -d` — all 5 long-running services (postgres, minio,
  redpanda, redpanda-console, mlflow) reached `Up ... (healthy)`; both
  one-shot jobs (`createbuckets`, `create-topics`) exited 0.
- Postgres: `\l` shows `aidp`, `aidp_test`, and `mlflow` databases all
  created. `\dt` on both `aidp` and `aidp_test` shows all 8 tables
  (`customers`, `customer_profiles`, `recent_events`, `transactions`,
  `fraud_scores`, `fraud_decisions`, `model_versions`, `pipeline_runs`).
  `\d customer_profiles` / `\d recent_events` confirm the online
  feature-store columns and the `idx_recent_events_customer_time` index
  (fix C1) are present exactly as designed.
- MinIO: `mc ls local` (via a throwaway container on `aidp-network`) shows
  all 6 target buckets (`aidp-raw`, `aidp-clean`, `aidp-curated`,
  `aidp-features`, `aidp-model-output`, `mlflow-artifacts`).
- Redpanda: `rpk topic list` shows all 4 topics (`transactions`,
  `fraud-decisions`, `dead-letter-transactions`, `transactions-test` — the
  last one for fix H4).
- MLflow end-to-end round-trip (proves fix C5's Postgres backend store +
  MinIO artifact store wiring, not just that the container is "healthy"):
  created a real run via the `mlflow` Python client against
  `http://127.0.0.1:5001`, logged a param + metric, uploaded an artifact —
  then listed and **downloaded** that artifact back and confirmed its
  content matched.
- **Persistence check (fix C5's actual claim):** ran `docker compose down`
  (containers + network removed, named volumes kept) followed by
  `docker compose up -d`. All services came back healthy within ~15s, and
  the same MLflow run (fetched via the REST API by run_id) was still
  present with its metric/param/artifact_uri intact — proving the registry
  survives a restart, not merely that containers can be stopped/started.
- `./scripts/healthcheck.sh`: `[OK]` for PostgreSQL, MinIO, Redpanda,
  MLflow; `[FAIL]` (expected, connection refused) for FastAPI/Streamlit,
  which don't exist until Phase 6/8.
- Web UIs confirmed reachable with HTTP 200: MinIO Console (9001), Redpanda
  Console (8080), MLflow UI (5001).

Commands executed (representative):
- `docker compose up -d`, `docker compose ps -a`
- `docker compose exec -T postgres psql -U aidp -d aidp -c "\dt"` (and
  `-d aidp_test`, `\d customer_profiles`, `\d recent_events`)
- `docker run --rm --network aidp-network --entrypoint sh quay.io/minio/mc:... -c "mc alias set ... && mc ls local"`
- `docker compose exec -T redpanda rpk topic list`
- Python: `mlflow.start_run()` / `log_param` / `log_metric` / `log_artifact`
  / `MlflowClient.list_artifacts` / `MlflowClient.download_artifacts`
- `docker compose down` then `docker compose up -d`; re-fetched the same
  run via `curl http://127.0.0.1:5001/api/2.0/mlflow/runs/get?run_id=...`
- `./scripts/healthcheck.sh`
- `curl -o /dev/null -w '%{http_code}'` against ports 9001, 8080, 5001

Issues: none blocking (the Docker Hub → quay.io image fix above was the only
issue, resolved within Phase 2).

Fixes verified in this phase: **C1, C4, C5, C6, H1, H4**.

---

## Phase 3 — Synthetic Data

Status: **COMPLETE**

Created:
- `src/common/schemas.py`: shared `Transaction` Pydantic model with type +
  bounds validation (`amount` `gt=0, le=10_000_000`, fix M4) — will be reused
  by the batch pipeline, consumer, and API in later phases (fix C2 spirit:
  one definition, not re-implemented per layer).
- `src/common/timeutil.py`: `APP_TZ` (pinned `Asia/Kolkata`) + `to_app_tz()`
  helper, the single source of truth later reused by `features.py` for
  `night_transaction_flag` (fix M2).
- `src/generator/customers.py`: `generate_customers()` (Faker + NumPy
  `default_rng`, fully seeded) producing customer dimension fields +
  profile-baseline fields (avg/stddev amount, known devices/countries) in
  one row — this row seeds both `customers` and `customer_profiles`
  directly. `demo_customer_row()` hand-specifies customer `C101` to match
  the guide's worked demo scenario exactly (typical spend ~8,000, home
  country India, fix C3).
- `src/generator/history.py`: `generate_history()` — distributes the
  configured transaction total across customers (moderate gamma skew,
  exact sum), generates realistic daytime transactions, then converts
  exactly `round(total * fraud_ratio)` of them to fraud by combining 2-3 of
  7 patterns (HIGH_AMOUNT, NEW_DEVICE, NEW_COUNTRY, NIGHT_TIME,
  RISKY_MERCHANT, RAPID_VELOCITY, FAILED_ATTEMPTS) per fraud case — matching
  the guide's example suspicious-pattern list. RAPID_VELOCITY re-times
  sibling transactions from the same customer into a tight burst rather
  than inventing new rows (keeps the configured total exact). FAILED_ATTEMPTS
  emits a separate failed-attempts event stream, not counted in the
  transaction total.
- `src/generator/seed.py`: CLI (`python -m src.generator.seed`) that
  generates customers + history, writes `data/seed/customers.parquet`,
  `data/seed/historical_transactions.csv`,
  `data/seed/historical_failed_attempts.csv`, then upserts `customers` +
  `customer_profiles` into Postgres for every generated customer (fix
  C1/C3) — idempotent (`ON CONFLICT ... DO UPDATE`).
- `scripts/seed_data.sh` wrapper.
- `tests/unit/test_generator.py`: 6 tests (repeatability of customers and
  history, exact size, exact fraud ratio, every generated row validates
  against the shared `Transaction` schema, different seeds diverge).

Verification executed (not just file creation):
- `pytest tests/unit -v` — **10/10 passed** (4 config + 6 generator).
- Full-scale run: `python -m src.generator.seed` with defaults (10,000
  customers, 50,000 transactions, fraud_ratio 0.03, seed 42) — completed in
  ~8s. Output: `transactions_written=50000`, `fraud_count=1500`
  (`fraud_ratio_actual=0.03`, exact), `failed_attempts_written=1113`,
  `customers_written=10001` (10,000 generated + the guaranteed `C101` demo
  row).
- Postgres: `SELECT count(*) FROM customer_profiles` → 10001;
  `SELECT count(*) FROM customers` → 10001. Confirmed both `C1001` (falls
  naturally in the generated `C{1000+i}` range) and `C101` (guaranteed, fix
  C3) exist in `customer_profiles` with sane values — `C101` has
  `avg_transaction_amount=8000.00`, `stddev=1500.00`, `home_country=India`,
  matching the guide's Section 2/55 worked demo scenario exactly.
- Data-quality spot check: fraud transactions cluster heavily in the night
  window (74-103 fraud txns/hour at 23:00-04:00 vs 13-44/hour at normal
  daytime hours) — confirms the NIGHT_TIME pattern is real, not cosmetic.
  Observed NEW_DEVICE (`DEVNEW...` ids), NEW_COUNTRY, and HIGH_AMOUNT
  patterns in sampled fraud rows. Failed-attempts CSV rows correctly
  reference the `transaction_id` they precede, with `occurred_at` before
  the transaction's timestamp.
- **True reproducibility check** (stronger than the in-process pytest
  check): ran the CLI as two separate OS processes back-to-back and
  diffed the output files byte-for-byte — `historical_transactions.csv`
  and `historical_failed_attempts.csv` were **identical** across runs.

Issues: none blocking.

Fixes verified in this phase: **C1, C3, C2 (schema only, full module still
pending Phase 4), M2 (timezone helper only), M4**.

---

## Phase 4 — Batch Pipeline

Status: **COMPLETE**

Created:
- `src/common/features.py` (fix C2, the centerpiece): `compute_features()`
  — one pure function taking a transaction's fields + a
  `CustomerProfileSnapshot` + a `Sequence[RecentEvent]` + `RiskLookups`, and
  returning every CURATED- and FEATURES-layer computed column in one pass.
  `DEFAULT_PROFILE` (fix C3) makes every ratio/flag well-defined for a
  customer with no profile and no history — no divide-by-zero, no crash.
  `MODEL_FEATURE_COLUMNS` is the single source of truth for the ML input
  column order, reused by Phase 5 training and (later) Phase 6 serving.
- `src/common/feature_store.py`: the **real-time** adapter onto this same
  module — `fetch_customer_profile()`/`fetch_recent_events()` query
  `customer_profiles`/`recent_events` live via Postgres (fix C1, "real and
  queried at scoring time"), falling back to `DEFAULT_PROFILE` for an
  unknown customer_id (fix C3). `record_event()` appends new events so the
  *next* transaction sees them. Not wired into a live caller yet (that's
  Phase 6/7) — its output shape is exercised now only by unit tests plus
  structural review; full correctness under real concurrent load is Phase
  6/7's job.
- `src/processing/enrich.py`: the **batch** adapter onto the same module —
  builds `CustomerProfileSnapshot` from the generator's `customers.parquet`
  baseline and `RecentEvent` history from the customer's own prior
  transactions plus `historical_failed_attempts.csv`, then calls the exact
  same `compute_features()`. One enrichment pass per dataset; CURATED and
  FEATURES are two column projections of its single output
  (`src/processing/views.py`), so nothing is computed twice.
- `src/processing/risk_lookups.py`: `merchant_risk_score`/
  `country_risk_score` computed from **observed historical fraud rate**
  (Laplace-smoothed, clipped to [0.02, 0.95]) rather than manually assigned
  per-merchant/per-country assumptions — deliberately avoids hardcoding any
  specific country as "risky." Unseen merchant/country falls back to the
  overall fraud rate (fix C3). Saved as `data/models/risk_lookups.json`
  and uploaded to the `aidp-model-output` bucket so Phase 6 serving can
  load the identical artifact.
- `src/processing/raw.py`: File (CSV or JSON) -> Validation via the shared
  `Transaction` Pydantic model (fix M4) -> RAW. Bad rows (missing field,
  non-numeric/out-of-bounds amount, bad timestamp) are rejected with a
  reason and reported, never silently dropped, never crash the run (guide
  section 22).
- `src/processing/clean.py`: duplicate removal (by `transaction_id`, keep
  first), timestamp normalization (canonical UTC), null handling, and a
  business-rule check (`payment_method` allow-list) Pydantic's type system
  can't express — each rejection reported with a specific reason.
- `src/processing/pipeline.py`: orchestrator CLI
  (`python -m src.processing.pipeline`) — runs RAW -> CLEAN -> risk-lookup
  computation -> CURATED/FEATURES, writes Parquet locally under
  `data/output/<layer>/` and uploads to the matching MinIO bucket, writes
  rejected-row CSV reports under `data/output/rejected/`, and records a
  `pipeline_runs` row in Postgres.
- Tests: `tests/unit/test_features.py` (14, covers the M2 timezone
  assertion explicitly, velocity windows, risk-lookup fallback,
  save/load round-trip), `tests/unit/test_raw.py` (11, missing field,
  4 invalid-amount variants, bad timestamp, mixed good/bad batch, JSON
  input, unsupported file type), `tests/unit/test_processing.py` (4,
  dedup/null/bad-payment-method rejection, risk-lookup signal direction,
  full enrich-then-project round trip).

Verification executed (not just file creation):
- `pytest tests/unit -v` — **38/38 passed**.
- Ran the real pipeline against the actual Phase 3 output:
  `python -m src.processing.pipeline` over the 50,000-row
  `historical_transactions.csv` — completed in ~6.5s. Summary:
  `raw_valid=50000, raw_rejected=0, clean_valid=50000, clean_rejected=0,
  curated_rows=50000, features_rows=50000` — counts reconcile end-to-end.
- DuckDB: queried all 4 output Parquet layers directly
  (`SELECT count(*) FROM 'data/output/<layer>/transactions.parquet'`) —
  50,000 rows at every layer. `DESCRIBE` confirms CURATED's and FEATURES'
  column sets exactly match the guide's section 8 field lists.
- **Signal sanity check** (proves the feature logic is real, not
  cosmetic): grouped FEATURES by `is_fraud` — fraud rows average
  `amount_vs_customer_average=4.28` vs `1.00` for non-fraud;
  `new_device_flag` is true for 34.5% of fraud rows vs **exactly 0%** of
  non-fraud; `new_country_flag` 35.7% vs 0%; `night_transaction_flag`
  36.6% vs 0.3%; `failed_attempts_last_1h` averages 0.654 vs 0.013;
  `merchant_risk_score` averages 0.051 vs 0.030. This is the expected
  separation given exactly how Phase 3 injected fraud patterns — confirms
  the shared feature module is correctly reading the data, not returning
  placeholder values.
- Postgres: `SELECT * FROM pipeline_runs` shows one `batch` /
  `SUCCESS` row with `records_processed=50000, records_rejected=0` and
  real `started_at`/`completed_at` timestamps.
- MinIO: `mc ls --recursive` confirms `transactions/transactions.parquet`
  actually landed in all 4 buckets (`aidp-raw`, `aidp-clean`,
  `aidp-curated`, `aidp-features`) plus `risk_lookups.json` in
  `aidp-model-output`.

Issues: none blocking.

Fixes verified in this phase: **C1 (real-time adapter written, not yet
exercised by a live caller), C2 (fully — one shared module, used
identically by the batch pipeline; real-time adapter shares it too, wiring
lands in Phase 6/7), C3, M1 (structured logs from every stage), M2 (the
explicit `night_transaction_flag` timezone test + real signal check
above), M4**.

---

## Phase 5 — ML Training

Status: **COMPLETE**

Created:
- `src/ml/train.py`: loads `data/output/features/transactions.parquet`,
  selects exactly `src.common.features.MODEL_FEATURE_COLUMNS` as X and
  `is_fraud` as y (never as a feature), stratified 70/15/15 train/val/test
  split (`random_state` pinned to the same seed as generation, 42). Trains
  XGBoost (primary, `scale_pos_weight` set from the train split's actual
  class imbalance, early stopping on the validation set), with a
  documented fallback to `RandomForestClassifier(class_weight="balanced")`
  only on a genuine local blocker. Computes precision/recall/F1/ROC-AUC/
  false-positive-rate/false-negative-rate/confusion-matrix as the headline
  metrics (accuracy computed too but explicitly marked supplementary, not
  the reported target metric). Logs everything to MLflow (params,
  hyperparameters, feature list, a content-hash `dataset_version`, all
  metrics, the confusion matrix as a JSON artifact, the model with an
  inferred signature) and registers the run as `fraud-detection-model`,
  aliasing the new version `champion`. Also writes a `model_versions` row
  to Postgres (deactivating any prior active version first).
- `src/common/mlflow_setup.py`: `configure_mlflow()` — one place that sets
  the tracking URI and the MinIO/S3 credentials the MLflow client needs
  for direct artifact access, reused by any future process that talks to
  MLflow (training now; API in Phase 6 if it loads models directly).

Issues found and fixed (root cause, not routed straight to the fallback):
1. **XGBoost failed to import**: `libxgboost.dylib` couldn't load —
   `libomp.dylib` (OpenMP runtime) isn't bundled on macOS. This is exactly
   the kind of "genuine local compatibility blocker" guide section 41
   allows a fallback for, and the RandomForest fallback *did* fire
   automatically and produced a legitimate result (precision 0.991,
   recall 0.978, F1 0.984, ROC-AUC 0.993) — but since XGBoost is the
   stack's specified primary model, fixed the actual root cause first:
   `brew install libomp`, then verified `import xgboost` succeeds. Documented
   in README.md under "macOS + XGBoost" so a future run on a machine
   without `libomp` degrades gracefully and visibly rather than silently.
2. **MLflow artifact logging failed** (`InvalidAccessKeyId` from MinIO):
   the training process had no AWS-style credentials in its environment,
   unlike the ad-hoc Phase 2 smoke test which set them inline. Fixed by
   adding `src/common/mlflow_setup.py` so every MLflow-talking process
   configures this consistently instead of each script re-deriving it.
3. **`mlflow.xgboost.log_model()` / `mlflow.sklearn.log_model()` signature
   mismatch**: this MLflow version (2.17.2) still expects `artifact_path=`,
   not the newer `name=` kwarg. Fixed.

Verification executed (not just "the script exited 0"):
- Full real training run against the actual 50,000-row FEATURES dataset
  (35,000 train / 7,500 val / 7,500 test, hashed `dataset_version=822b525753c2b3cb`).
  **XGBoost** result: precision=0.9367, recall=0.9867, F1=0.9610,
  roc_auc=0.9992, false_positive_rate=0.00206, false_negative_rate=0.01333,
  confusion matrix tn=7260/fp=15/fn=3/tp=222.
- Postgres: `SELECT * FROM model_versions` shows exactly one row,
  `model_version=1`, `is_active=true`, matching MLflow's run_id and metrics.
- **Model-loadability check, in a separate freshly-started Python process**
  (not the training script's memory): fetched
  `client.get_model_version_by_alias("fraud-detection-model", "champion")`
  → version 1, status `READY`; loaded it two ways —
  `mlflow.pyfunc.load_model("models:/fraud-detection-model@champion")`
  (confirmed input schema = exactly `MODEL_FEATURE_COLUMNS`, all double,
  and correctly classified a suspicious vs. a normal feature vector) and
  `mlflow.xgboost.load_model(...)` (native flavor, needed for
  `predict_proba` — pyfunc's `.predict()` returns the hard class label,
  not a probability, which Phase 6's threshold-based decisioning needs).
  Native-flavor fraud probability: **0.844** for the guide's suspicious
  demo pattern (new device + new country + night + failed attempts +
  10x amount) — lands in the REVIEW/BLOCK range — vs. **0.300** for an
  ordinary transaction — lands in APPROVE. Confirms both "the model can be
  loaded" and "the loaded model's output is directionally sane," not just
  that registration succeeded.
- `pytest tests/unit -v` — still **38/38 passed** (no regressions; Phase 5
  was primarily a runtime/infra-integration phase, exercised live above
  rather than via new unit tests).

Design note carried forward to Phase 6: the API must load the model via
`mlflow.xgboost.load_model()` (native flavor), not generic `pyfunc`, to get
`predict_proba()`. It must also load `data/models/risk_lookups.json` (or
the MinIO copy) — the same artifact Phase 4 computed — rather than
recomputing risk scores itself.

Issues: none blocking (all three issues above were fixed within this
phase).

Fixes verified in this phase: **none newly verified — Phase 5 builds on
C1/C2/C3 already in place; MLflow's persistent registry (fix C5) got a
second real proof point here (a genuinely new, independently-trained model
registered and reloaded, not just the Phase 2 dummy artifact)**.

---

## Phase 6 — API

Status: **COMPLETE**

Created:
- `src/decisioning/engine.py`: `classify()` applies the
  `config/fraud_rules.yaml` thresholds (0.00-0.40 APPROVE, 0.40-0.75
  MONITOR, 0.75-0.90 REVIEW, 0.90-1.00 BLOCK), mapped to risk_level
  LOW/MEDIUM/MEDIUM/HIGH+HIGH (matching the Postgres schema's 3-level
  `risk_level` CHECK constraint). `reason_codes()` implements all 6 codes
  from the guide (NEW_DEVICE, NEW_COUNTRY, HIGH_AMOUNT_VS_AVERAGE,
  MULTIPLE_RECENT_ATTEMPTS, NIGHT_TRANSACTION, HIGH_VELOCITY).
- `src/api/schemas.py` / `src/api/main.py`: the FastAPI service. This is
  where `src/common/feature_store.py` (fix C1, real-time Postgres adapter)
  gets its first live caller — `POST /score` fetches the customer's
  profile + recent events live, calls the exact same
  `src.common.features.compute_features()` the batch pipeline uses (fix
  C2), scores with the Phase 5 model (loaded once at startup via
  `mlflow.xgboost.load_model` to get `predict_proba`, not generic
  `pyfunc`), classifies via `src/decisioning/engine.py`, persists
  transaction + fraud_scores + fraud_decisions, and appends a
  `recent_events` row so the *next* request for that customer sees it
  (closing the C1 loop for real). `/health` returns 503 + `status:
  degraded` when the model isn't loaded, 200 otherwise (fix H2).
  `ScoreRequest.amount` uses the same `Field(gt=0, le=MAX_REASONABLE_AMOUNT)`
  bound as the shared schema (fix M4). `ThreadedConnectionPool` (not
  `SimpleConnectionPool` — FastAPI sync endpoints run across a threadpool,
  so the pool must be thread-safe). All endpoints bind to 127.0.0.1 (fix
  H1, via `API_HOST`/`API_PORT`). `M1` correlation: `bind_transaction_id()`
  wraps the whole `/score` handler.
- `src/common/mlflow_setup.py` reused (not re-derived) for the API's model
  loading.
- `scripts/run_api.sh`.
- `tests/unit/test_decisioning.py`: 11 tests covering every threshold
  boundary (0.40/0.41, 0.75/0.76, 0.90/0.91) and every reason code firing
  independently, in combination, and not firing on a clean transaction.

Issues found and fixed:
1. Pydantic warned about `model_version`/`model_name` colliding with its
   protected `model_` namespace (harmless but noisy) — added a shared
   `_Base` schema class with `protected_namespaces=()` rather than rename
   fields away from their natural, DB-matching names.
2. `docker compose down` earlier in the session meant the API's first real
   test run needed the stack back up — no code issue, just sequencing.

Verification executed against a **live, real** stack (not TestClient
mocks) — started with `uvicorn src.api.main:app --host 127.0.0.1 --port
8000` against the actual running Postgres/MLflow:
- `pytest tests/unit -v` — **49/49 passed** (38 prior + 11 new decisioning tests).
- `GET /health` → 200, `{"status":"ok","model_loaded":true,
  "postgres_connected":true,"model_version":"1"}`.
- `GET /model` → real precision/recall/F1/ROC-AUC pulled from Postgres
  `model_versions`, matching Phase 5's numbers exactly.
- **Guide's exact worked demo, via real HTTP calls**: normal transaction
  for `C101` (₹4,500, Grocery, India, known device) →
  `fraud_probability=0.176`, `LOW`/`APPROVE`, no reason codes. Suspicious
  transaction for `C101` (₹82,000, Electronics, Singapore, unknown device)
  → `fraud_probability=0.847`, `HIGH`/`REVIEW`,
  `reason_codes=["NEW_DEVICE","NEW_COUNTRY","HIGH_AMOUNT_VS_AVERAGE"]` —
  matches guide section 17's worked example almost field-for-field.
- `GET /transactions/{id}` and `GET /decisions/{id}` for that same
  transaction_id returned the persisted rows correctly; an unknown id
  correctly returned 404.
- **Velocity signal proven live, not just in a unit test**: fired 4 rapid
  `/score` calls for the same customer; the 4th correctly carried
  `HIGH_VELOCITY` in `reason_codes` (transactions_last_10m reached 3) —
  this can only happen if `fetch_recent_events()` is genuinely re-querying
  Postgres per request and seeing the previous calls' own inserts.
  `SELECT customer_id, event_type, count(*) FROM recent_events GROUP BY
  ...` confirmed 6 real rows written by these live API calls.
- `GET /metrics` aggregated correctly across all scored transactions
  (`total_scored=6, decision_counts={"REVIEW":1,"APPROVE":5},
  high_risk_rate=0.1667`).
- **Fix M4 proven live**: `amount=-50` → 422 with a clear Pydantic error;
  `amount=50000000` (exceeds the bound) → 422. Never a 500, never silently
  accepted.
- **Fix H2 proven live, with a real fault injection** (not code review):
  `docker compose stop mlflow`, then restarted the API — startup did not
  crash; `_load_model()` caught the connection failure and logged it;
  `GET /health` → **503**, `{"status":"degraded","model_loaded":false,
  "postgres_connected":true}`; `POST /score` → **503**,
  `{"detail":"Model not loaded"}` (never 500). Then `docker compose start
  mlflow` + API restart → `GET /health` → 200 again, model reloaded
  automatically. Full degrade/recover cycle proven, not assumed.
- `./scripts/healthcheck.sh`: FastAPI now shows `[OK]`; Streamlit still
  correctly `[FAIL]` (Phase 8 not built yet).
- `GET /docs` and `GET /openapi.json` both return 200 (Swagger/OpenAPI
  stays enabled, per guide section 17).

Issues: none blocking.

Fixes verified in this phase: **C1 (fully — real live caller now, proven
with the velocity test), C2 (real-time path now actually exercised, not
just structurally reviewed), H1, H2 (proven via real fault injection), M1,
M4 (proven live)**.

---

## Phase 7 — Real-Time Stream

Status: **COMPLETE**

Created:
- `src/common/scoring.py`: extracted `score_and_persist()` — refactored
  out of Phase 6's API handler so the API and the new consumer call
  **identical** scoring/persist logic (extends fix C2's "no duplicated
  logic" principle beyond feature computation to the whole scoring path).
  Re-verified the API against the guide's demo scenario after this
  refactor — unchanged behavior.
- `src/common/retry.py`: `transient_retry()` — tenacity-based
  retry-with-backoff (fix H3) configured from `config/settings.yaml`'s
  `streaming.retry` block (max_attempts=5, exponential-jitter backoff),
  with a structured `before_sleep` log hook.
- `src/generator/stream_transactions.py`: Redpanda producer, configurable
  rate (`--rate`), pulls a real customer pool from Postgres
  `customer_profiles` (not synthetic-in-isolation), injects the same
  fraud-pattern families as Phase 3 at a configurable ratio, tags every
  message `schema_version: 1` (fix M3), and supports
  `--inject-malformed-rate` to deliberately publish broken messages for
  exercising the DLQ path end-to-end.
- `src/ingestion/consumer.py`: Redpanda consumer. Two distinct failure
  paths (fix H3): malformed messages (bad JSON / schema-and-bounds
  violation via the shared `Transaction` model) go straight to
  `dead-letter-transactions`; transient failures during
  `score_and_persist` are retried with backoff and only DLQ'd (with a
  `processing_failed_after_retries` reason) if retries are exhausted.
  Raw successfully-scored events are micro-batched (10 rows) to Parquet
  and uploaded to the `aidp-raw` bucket under `streaming/<date>/`,
  mirroring the batch pipeline's RAW layer. Records a `pipeline_runs` row
  (`pipeline_name='stream'`) per run. Manual offset commits (only after a
  message is fully handled — scored or DLQ'd), so nothing is
  acknowledged before it's actually dealt with.
- `scripts/run_stream.sh`, `scripts/run_consumer.sh`.
- `tests/unit/test_retry.py`: 2 tests, both against the real tenacity
  decorator (not mocked) — recovers after 2 transient failures (3rd call
  succeeds), and exhausts after exactly `max_attempts=5` then re-raises.

Issues found and fixed (root cause, not routed around):
1. **`confluent-kafka` couldn't consume at all.** `Producer` worked, but
   `Consumer.poll()` always returned nothing and the consumer group never
   showed a member. Redpanda logs showed the real cause:
   `Unsupported version 12 for fetch API` — librdkafka 2.6.0 (bundled with
   `confluent-kafka==2.6.0`) defaults to negotiating Kafka Fetch API v12,
   which Redpanda doesn't implement. Tried `api.version.request=false` +
   `broker.version.fallback` client-side config first (didn't help — the
   client still requested v12). Tried upgrading Redpanda instead, stepping
   through v24.3.13 -> v25.1.1 -> v25.2.1 (had to step one release at a
   time — jumping straight from v24.2.7 to v25.2.1 crashed with
   `Attempted to upgrade from incompatible logical version 13 to 16!`,
   Redpanda's cluster-metadata format doesn't support skipping that many
   releases). Even at the newest available tag (v25.2.1), Fetch v12 was
   still unsupported. Root cause was therefore client-side: pinned
   `confluent-kafka==2.3.0` (older librdkafka, doesn't request v12) in
   `requirements.txt` with a comment explaining why — confirmed this
   resolves it. Kept the Redpanda upgrade to v25.2.1 anyway since it's a
   more current, better-supported release and the stepped upgrade
   preserved all existing topic data (verified: 98 pre-existing test
   messages were still present and readable after the version bump).
2. **Design gap found during my own H3 verification, not by inspection**:
   the consumer originally held one Postgres connection for its entire
   run and passed it into the retried scoring call. If Postgres was
   briefly down, every retry attempt reused the same now-dead connection
   object and would keep failing even after Postgres actually recovered
   (retrying doesn't help if you keep retrying against a socket that's
   already dead). Fixed by having `_score_with_retry` (and
   `_record_run_start`/`_record_run_end`) acquire a **fresh** connection
   on each attempt via `src.common.db.get_connection()`, so a later retry
   genuinely gets a chance against recovered infrastructure.
3. **Same class of bug found in the Phase 6 API**, while checking on it
   after the Postgres restarts above: `GET /health` stayed permanently
   `"degraded"/"postgres_connected": false` even after Postgres was fully
   healthy again, because `ThreadedConnectionPool` had handed out (and got
   back) a now-dead pooled connection with no validation. Fixed by adding
   `_checkout_validated_connection()` to `src/api/main.py`: pings
   (`SELECT 1`) every connection on checkout, and if that fails, discards
   it (`putconn(..., close=True)`) and gets a fresh one before handing
   anything to a request handler.

Verification executed (all against the real running stack):
- `pytest tests/unit -v` — **51/51 passed** (49 prior + 2 new retry tests;
  the retry-exhaustion test takes ~9.5s real wall-clock time since it
  actually sleeps through the real backoff schedule — deliberate, to prove
  actual timing, not just call counts).
- **End-to-end live run**: started the consumer, then the producer
  (`--rate 5 --duration 20 --fraud-ratio 0.15 --inject-malformed-rate 0.1`)
  against it, then a second live burst (`--rate 5 --duration 15
  --fraud-ratio 0.2`) while the *same* consumer instance was still
  running. Consumer summary: `processed=160, rejected=11` — reconciles
  exactly against `98 + 73 = 171` messages sent across both producer runs.
- Postgres: `transactions` grouped by `source` shows `stream: 160`
  (plus `api: 7` from earlier manual testing); `fraud_decisions` joined
  against `transactions` shows a realistic decision spread for the stream
  source (135 APPROVE / 11 MONITOR / 14 REVIEW); `pipeline_runs` has a
  `stream`/`SUCCESS` row per consumer run with accurate
  `records_processed`/`records_rejected`.
- MinIO: `aidp-raw/streaming/<date>/*.parquet` — **16 files**, exactly
  matching `160 processed / 10 per batch`.
- **DLQ proven for real**: consumed `dead-letter-transactions` directly —
  **11 messages**, matching the rejected count exactly, each with the
  original message, a specific `rejection_reason` (missing
  `transaction_id`, `amount <= 0`, or an unparseable timestamp), and a
  `failed_at` timestamp.
- **Fix H3 recovery proven live** (not just unit-tested): produced a
  valid message, stopped Postgres, started the consumer (its
  `_record_run_start` retried against the down DB), restarted Postgres
  ~6s later, and confirmed both the pre-outage test messages were
  eventually scored and persisted (`SELECT ... FROM transactions WHERE
  transaction_id LIKE 'TXH3TEST%'` → both rows present, `source=stream`)
  — proving retries genuinely re-attempt against recovered infrastructure
  rather than a stale connection.
- **API connection-pool self-healing proven live**: with the API process
  left running continuously (not restarted) from Phase 6, ran
  `docker compose restart postgres`, then immediately called `GET
  /health` — before the fix this stayed `"degraded"` forever; after the
  fix it correctly showed `"status":"ok","postgres_connected":true"`
  without restarting the API, and a subsequent `POST /score` succeeded
  normally.
- `./scripts/healthcheck.sh`: still `[OK]` for all 5 built services,
  `[FAIL]` (expected) for Streamlit.

Issues: none blocking (all four issues above were root-caused and fixed
within this phase).

Fixes verified in this phase: **C2 (extended to the whole scoring path,
not just feature computation), H3 (both halves — recovery and, via unit
test, exhaustion — proven), M1 (transaction_id correlation through
consumer logs), M3 (schema_version present on every streamed message),
M4 (malformed/out-of-bounds messages correctly DLQ'd, never crash the
consumer)**. Also hardened fix H2's spirit further (the API's
self-healing connection pool).

---

## Phase 8 — Dashboard

Status: **COMPLETE**

Created:
- `src/dashboard/data.py`: query/health functions reading directly from
  the same Postgres tables the API/consumer write to (`transactions`,
  `fraud_decisions`, `model_versions`) — no separate dashboard-only data
  path. Platform Health checks reuse the same signals
  `scripts/healthcheck.sh` uses (Postgres connect, MinIO `/minio/health/live`,
  Redpanda admin API `/v1/status/ready`, MLflow `/health`, FastAPI
  `/health`), via `httpx` (already a pinned dependency — used instead of
  the unpinned `requests`, which turned out to only be present
  transitively).
- `src/dashboard/app.py`: Streamlit, 5 tabs matching guide section 18
  exactly — Executive Overview (transactions processed, transaction
  value, fraud alerts, fraud rate, blocked transactions, potential fraud
  value, active model version, system health), Live Transaction Feed
  (sortable/filterable table, adjustable row count), Fraud Analysis
  (score distribution histogram, fraud by country/merchant bar charts,
  fraud-over-time line chart, amount-vs-score scatter, highest-risk
  table), Model Performance (precision/recall/F1/ROC-AUC +
  confusion-matrix table from Phase 5's saved artifact), Platform Health
  (live status per service). All charts use Streamlit's native
  `bar_chart`/`line_chart`/`scatter_chart` — no new charting dependency
  added.
- `scripts/run_dashboard.sh`.
- `tests/unit/test_dashboard.py`: runs the **actual** dashboard script
  headlessly via Streamlit's own `AppTest` framework (not a mock) against
  the real live stack — asserts zero exceptions and that all 5 Platform
  Health service labels render.

Issue found and fixed:
- Initially wrote `get_platform_health()` using the `requests` library,
  which isn't in `requirements.txt`. It happened to import successfully
  anyway (pulled in transitively by another dependency), which would have
  been a silent landmine — swapped to `httpx`, already an explicit pinned
  dependency, before it could bite anyone relying on a clean venv.

Verification executed (real content, not just "the process starts") — no
browser tool was available this session (the user declined the Chrome
extension), so verification used Streamlit's own headless `AppTest`
framework, which genuinely executes `src/dashboard/app.py` server-side
(Streamlit runs every tab's code on each script run regardless of which
tab is visually selected — this is not a partial/lazy test):
- `AppTest.from_file("src/dashboard/app.py").run()` → **zero exceptions**.
- Inspected actual rendered widget values against the real database state
  built up over Phases 6-7: Executive Overview showed
  `Transactions Processed=171` (matches the live total exactly),
  `Transaction Value=₹1,042,650.67`, `Fraud Alerts=15`,
  `Fraud Rate=8.77%`, `Potential Fraud Value=₹451,548.95`,
  `Active Model Version=1`, `System Health=🟢 Healthy`.
  `Blocked Transactions=0` — correct/expected, since no test transaction
  so far exceeded the 0.90 BLOCK threshold (max seen was ~0.85).
- Platform Health tab: all 5 services (PostgreSQL, MinIO, Redpanda,
  MLflow, FastAPI) showed 🟢 OK.
- Model Performance tab: Precision/Recall/F1/ROC-AUC matched Phase 5's
  training output exactly (0.9367/0.9867/0.9610/0.9992); the rendered
  confusion-matrix table matched the saved artifact exactly
  (tn=7260/fp=15/fn=3/tp=222).
- Live Transaction Feed: rendered a real 100-row dataframe including the
  Phase 6 demo transactions and the Phase 7 H3 test transactions.
- Fraud Analysis "Highest-Risk Transactions" table: top-ranked entry was
  the guide's own suspicious demo transaction
  (`TXA88CA764D7669`, ₹82,000, `fraud_probability=0.847416`, `REVIEW`) —
  the dashboard surfaces exactly the case the guide's worked demo cares
  about, at the top, unprompted.
- `pytest tests/unit -v` — **53/53 passed** (51 prior + 2 new dashboard
  tests).
- `./scripts/healthcheck.sh` — all **6/6 services `[OK]` simultaneously**
  for the first time this build (PostgreSQL, MinIO, Redpanda, MLflow,
  FastAPI, Streamlit) — matches the guide's section 24 target output
  exactly. (One run showed a transient `[FAIL]` for Redpanda immediately
  after the AppTest suite; reproduced the exact failing command manually
  and it succeeded — confirmed non-reproducible/transient via 2 further
  consecutive clean runs, not a real regression.)

Known limitation: no visual/browser confirmation of layout, styling, or
interactive widget behavior (tab-clicking, refresh buttons, chart
tooltips) was possible this session — `AppTest` proves the script executes
correctly and produces correct data-bearing widgets, but does not render
pixels. The user should open http://127.0.0.1:8501 to visually confirm
layout before considering Phase 8 fully signed off.

Issues: none blocking.

Fixes verified in this phase: **H1 (dashboard bound to 127.0.0.1)**. No
other fixes newly introduced — this phase is a read-only consumer of data
already correctly produced by Phases 3-7.

---

## Phase 9 — Tests

Status: **COMPLETE**

Refactor required first (fix H4 compliance): `src/ingestion/consumer.py`'s
`run()` and `src/generator/stream_transactions.py`'s `run()`
(`_load_customer_pool`) originally hardcoded the production topic name and
default database. Since every integration test needs to exercise the
*real* producer/consumer code against the *isolated* `transactions-test`
topic and `aidp_test` database rather than reimplementing test-only
versions of that logic, both functions now accept optional
`topic`/`database`/`group_id`/`dlq_topic`/`raw_bucket` overrides
(default = production values, so Phase 6/7's existing behavior is
unchanged when called with no arguments — re-verified: `pytest tests/unit`
still 53/53 green after this refactor, before writing a single new test).

Created:
- `tests/integration/conftest.py`: `clean_test_db` (autouse, truncates all
  relevant `aidp_test` tables before every integration test — real
  isolation, not just a different table prefix) and `seeded_customer`
  (inserts one known customer + profile into `aidp_test`).
- `tests/integration/test_streaming_integration.py`: producer → Redpanda →
  consumer → PostgreSQL → MinIO, and a malformed-message → DLQ variant.
  Each test starts `consumer.run()` in a background thread with a
  **brand-new consumer group** (`auto.offset.reset=latest`) *before*
  producing, so it only ever sees the message that specific test sent —
  avoiding the alternative (`from_beginning=True` + fresh group) which
  would replay the *entire* accumulated `transactions-test` history on
  every run and break exact-count assertions over repeated test runs.
- `tests/integration/test_feature_pipeline_to_model.py`: loads the real
  registered model, runs `compute_features()` for a suspicious vs. normal
  transaction (reusing the guide's exact worked-demo numbers), asserts
  the model's predictions are directionally correct end to end.
- `tests/integration/test_api_integration.py`: FastAPI `TestClient`
  (triggers the real lifespan → loads the real model) with the `get_conn`
  dependency overridden to `aidp_test` — `POST /score`,
  `GET /transactions/{id}`, `GET /decisions/{id}` all exercised for real
  against isolated data; plus a 422-on-invalid-amount check (fix M4).
- `tests/smoke/test_end_to_end.py`: the guide's exact required path in
  one test — generate → publish (Redpanda, test topic) → consume (real
  consumer, test DB) → feature calculation → model score → decision →
  store → retrieve through the API (`TestClient`, test DB). Uses a
  ₹82,000/Singapore/new-device pattern and asserts the retrieved decision
  is not a quiet APPROVE and carries `NEW_DEVICE`/`NEW_COUNTRY` reason
  codes — a real assertion about platform behavior, not just "the calls
  didn't throw."

Unit test coverage audit (guide section 23's required areas: schemas,
feature calculations, fraud rules, risk thresholds, predictor, config) —
all already covered by Phases 3-8's tests, no gaps found requiring new
unit tests: `test_raw.py`/`test_generator.py` (schemas),
`test_features.py` (feature calculations), `test_decisioning.py` (fraud
rules/thresholds), `test_config.py` (config). "Predictor" has no separate
unit beyond `compute_features()` + `model.predict_proba()`, both covered
directly (unit) and end-to-end (integration/smoke).

Verification executed:
- `pytest tests/integration/test_streaming_integration.py -v` — **2/2
  passed**, both with exact `processed`/`rejected` counts (not `>=`).
- `pytest tests/integration/test_api_integration.py
  tests/integration/test_feature_pipeline_to_model.py -v` — **3/3 passed**.
- `pytest tests/smoke/test_end_to_end.py -v` — **1/1 passed**; consumer
  log shows `fraud_probability=0.8474, decision=REVIEW` for the smoke
  test's suspicious pattern, matching expectations.
- **Full suite together** (`pytest -q`, no path filter — checks for
  cross-file interference, e.g. the shared FastAPI `app.dependency_overrides`
  state touched by both `test_api_integration.py` and
  `test_end_to_end.py`) — **59/59 passed** in 48.6s (53 unit + 6
  integration + 1 smoke; no order-dependence or leakage between files).
- **Isolation actually verified, not assumed** (the whole point of fix
  H4): after the full suite ran, queried the **demo** database directly —
  `aidp.transactions` count unchanged at 171 (exactly where Phase 7 left
  it), `aidp.customer_profiles` unchanged at 10,001. Queried `aidp_test` —
  only the last test's leftover row (autouse fixture truncates at the
  *start* of each test, not after, so one row persisting between suite
  runs is expected and harmless). Confirms the entire test suite never
  touched demo data, exactly as fix H4 requires.

Issues: none blocking.

Fixes verified in this phase: **H4 (fully — isolated test DB and topic
actually exercised by real tests, and demo-data non-interference
independently confirmed via direct query, not just asserted in test
code)**.

---

## Phase 10 — Final Demo Packaging

Status: **COMPLETE**

Cleanup found and fixed:
- Removed a stray, empty `mlruns/` directory at the project root (a local
  MLflow file-store default that got created by some process before
  `configure_mlflow()` ran; already gitignored, contained no real data,
  but confusing clutter — deleted).
- Found the guide's own required script/file list wasn't fully built yet:
  `scripts/train_model.sh` and `scripts/reset_demo.sh` didn't exist, and
  there was no `Makefile`. Created all three.

Created:
- `scripts/train_model.sh` — thin wrapper around `python -m src.ml.train`.
- `scripts/reset_demo.sh` — clears only this POC's demo data (aidp table
  rows, `data/seed/`, `data/output/`, the 4 data-lake MinIO buckets);
  explicitly does **not** touch `aidp_test`, Docker volumes/containers,
  Redpanda topics, or the MLflow model registry. Scope documented in the
  script's own header comment per guide section 60's requirement.
- `Makefile` — `bootstrap start stop health seed train demo api dashboard
  consumer test reset` targets, each a thin wrapper around the
  corresponding script.
- `ARCHITECTURE.md`, `RUNBOOK.md`, `DEMO_SCRIPT.md`, `TROUBLESHOOTING.md`
  — see below.
- Finalized `README.md` with real local URLs, the actual quick-start
  sequence, and a documentation index.

Verification executed (not just "the files exist"):
- **`reset_demo.sh` actually run** (not just read): confirmed it emptied
  `aidp`'s `transactions`/`customer_profiles`/`fraud_decisions` (all to 0),
  cleared `data/seed/` and `data/output/`, cleared the `aidp-raw`/
  `aidp-curated` MinIO buckets — and confirmed `aidp_test` was
  **untouched** (still showed its pre-existing row), proving the script's
  scope is exactly as documented.
- **Full demo data regenerated from scratch** to restore a working demo
  after the reset test: `seed_data.sh` (10,001 customers, 50,000
  transactions, exactly 3% fraud — identical to the original Phase 3 run,
  confirming full reproducibility), `python -m src.processing.pipeline`
  (50,000 rows through all 4 layers, 0 rejected), `train_model.sh`
  (registered as version **2**, identical metrics to version 1:
  precision=0.9367, recall=0.9867, F1=0.9610, ROC-AUC=0.9992 — same
  seed, same data, same result).
- Restarted the API (picked up model v2 automatically) and reproduced the
  guide's exact demo scenario fresh: normal C101 transaction →
  `fraud_probability=0.176`/APPROVE; suspicious 82,000/Singapore/new-device
  → `fraud_probability=0.847`/REVIEW/`[NEW_DEVICE, NEW_COUNTRY,
  HIGH_AMOUNT_VS_AVERAGE]` — byte-for-byte the same result as the
  original Phase 6 run.
- Ran a fresh streaming burst (49 events) through the consumer to
  repopulate live dashboard data; confirmed via the dashboard's own
  `AppTest` run afterward (`Transactions Processed = 148`, 0 exceptions).
- **Definition of Done (guide section 26), walked item by item with a
  fresh command for each, not cited from memory**:
  - Docker Compose starts: `docker compose ps` — 5/5 services `Up
    (healthy)`.
  - Redpanda works / topic exists: `rpk topic list` — all 4 topics
    present.
  - MinIO works / buckets exist: `mc ls local` — all 6 buckets present.
  - PostgreSQL works / schema exists: `\dt` — 8 tables.
  - Historical data generates / batch processing / RAW / CLEAN / CURATED
    / FEATURES / Parquet generated: all 4 layer Parquet files present
    with correct sizes.
  - DuckDB queries work: `SELECT count(*) FROM '...features/transactions.parquet'`
    via DuckDB → 50,000.
  - Model trains / MLflow records the run / model can be loaded:
    `MlflowClient.get_model_version_by_alias` → version 2, `READY`;
    `mlflow.xgboost.load_model(...)` → loads successfully.
  - FastAPI starts / `/health` works / `/score` works: live `curl` calls,
    200 responses.
  - Producer publishes events / consumer receives events / model scores
    live transactions / decisions are stored: live producer+consumer run,
    `{"sent": 49, ...}` / `{"processed": 49, "rejected": 0, ...}`.
  - Dashboard starts / live transactions appear: Streamlit `/_stcore/health`
    → 200; `AppTest` shows real, current transaction counts.
  - Fraud demo scenario works: reproduced fresh (above).
  - Unit tests pass / integration tests pass / smoke test passes:
    `pytest -q` (full suite, no path filter) → **59/59 passed**.

**Every single Definition of Done checklist item is independently
confirmed via a command actually run in this final pass — not inferred
from an earlier phase's log entry.**

Issues: none blocking.

---

## Post-completion fix: dashboard `ModuleNotFoundError`

The user opened http://127.0.0.1:8501 after Phase 10 and hit
`ModuleNotFoundError: No module named 'src'` from `src/dashboard/app.py`'s
`from src.dashboard.data import ...`. This had not surfaced during this
session's own testing because every dashboard launch here happened to run
from the project root with the venv active; the user's launch context
apparently differed enough that `sys.path` didn't include the project
root — `streamlit run <path>` only ever adds the *script's own* directory
to `sys.path`, not the project root, so this was latent and
environment-dependent, not something the prior testing would reliably
have caught.

**Fix**: added an explicit `sys.path.insert(0, <project root>)` at the top
of `src/dashboard/app.py`, before any `src.*` import, computed from
`Path(__file__).resolve().parents[2]` (robust regardless of CWD).

**Verification**: reproduced the likely failure mode directly — launched
`streamlit run` with an absolute path to `app.py` from a completely
different working directory (`/tmp`) — confirmed it failed the same way
conceptually before the fix would apply, then confirmed after the fix
that the exact same launch method starts cleanly (`Streamlit health: 200`,
`AppTest` shows 0 exceptions, real data still renders correctly:
`Transactions Processed = 148`). `pytest tests/unit/test_dashboard.py`
still 2/2 passing. Dashboard restarted cleanly via
`./scripts/run_dashboard.sh` afterward to return to normal operation.
Documented in `TROUBLESHOOTING.md`.

---

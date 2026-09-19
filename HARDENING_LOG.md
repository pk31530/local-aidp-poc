# Hardening Log — v1.1

This documents a post-completion hardening pass on branch
`feat/aidp-v1-1-hardening`, run against the finished v1 build recorded in
`BUILD_LOG.md`. It is a targeted correctness review, not a rebuild: five
issues were found, each fixed independently with its own tests, then the
batch pipeline and model were re-run end to end so the running system
reflects every fix. See `CHECKPOINT.md` for current state and
`ARCHITECTURE.md` for the original build's fixes list (C1–M4), which this
pass leaves unchanged.

Every fix below shipped as its own commit, in the order listed, each with
new or updated tests reviewed and run before committing.

## Fix 1 — schema_version silently accepted unsupported values

**Commit:** `1228abe`
**Files:** `src/common/schemas.py`, `tests/unit/test_schemas.py`,
`tests/integration/test_streaming_integration.py`

`Transaction.schema_version` had no upper/allowed-value bound, so a message
declaring an unrecognized future `schema_version` was processed as if it
were v1 instead of being safely DLQ'd — contradicting the documented
contract in `config/settings.yaml` (build fix M3). Added a
`SUPPORTED_SCHEMA_VERSIONS` validator; the existing
`ValidationError -> DLQ` path in the consumer handles the rejection with no
consumer changes needed.

## Fix 2 — serving assumed the champion model was always XGBoost

**Commit:** `b427e0d`
**Files:** `src/api/main.py`, `src/common/mlflow_setup.py`,
`src/ingestion/consumer.py`, `tests/unit/test_mlflow_setup.py`

`src/ml/train.py` can register a RandomForest fallback via
`mlflow.sklearn.log_model` when XGBoost training fails, but both
`src/api/main.py` and `src/ingestion/consumer.py` unconditionally loaded
the champion model via `mlflow.xgboost.load_model`. The API degraded to
503 on a flavor mismatch; the consumer crashed outright at startup with no
`pipeline_runs` record. Added a shared `load_champion_model()` that reads
the registered `model_type` param and picks the matching loader.

## Fix 3 — pipeline failures were dropped or misrecorded as SUCCESS

**Commit:** `2079f7f`
**Files:** `src/processing/pipeline.py`, `src/ingestion/consumer.py`,
`tests/unit/test_pipeline.py`, `tests/unit/test_consumer_run_status.py`

Batch: `run_pipeline()` had no try/except and only ever wrote a
`pipeline_runs` row on success as the last statement, so a failed batch
run left no row at all. Stream: `_record_run_end()` hardcoded
`status='SUCCESS'` and was called unconditionally from a `finally` block,
so a `KafkaException` or any other unhandled exception in the poll loop
(or a model-load failure, which happened before any `pipeline_runs` row
even existed) got recorded as a false success or left no trace. Wrapped
`run_pipeline()` in try/except to write `FAILED` with best-effort counts
and re-raise; moved the consumer's `RUNNING` row before model loading,
gave `_record_run_end()` a status parameter, and added a broad
`except Exception` branch that records `FAILED` and re-raises — `SUCCESS`
is now only recorded on genuinely graceful exits.

## Fix 4 — duplicate velocity-feature rows on event replay

**Commit:** `9f032c0`
**Files:** `infrastructure/postgres/lib/schema.sql`,
`infrastructure/postgres/migrations/001_recent_events_unique_transaction_id.sql`,
`src/common/feature_store.py`,
`tests/integration/test_streaming_integration.py`

`transactions`/`fraud_scores`/`fraud_decisions` are idempotent via
`ON CONFLICT DO NOTHING` keyed on `transaction_id`, but `recent_events` had
only a surrogate `BIGSERIAL` primary key and no constraint on
`transaction_id`, so replaying the same message (redelivery, a retried
call whose earlier attempt actually committed, etc.) inserted a second
row. `recent_events` is queried live by `fetch_recent_events` to compute
the velocity features fed to the model, so duplicates inflated those
counts for the customer's subsequent transactions. Added a partial unique
index on `transaction_id` (nullable preserved for any future
non-transaction-linked event) and routed `record_event`'s INSERT through
`ON CONFLICT ... DO NOTHING`. `schema.sql` covers fresh installs; the
migration file (not auto-applied) de-duplicates and adds the same index on
an already-running database — applied to the live `aidp` and `aidp_test`
databases (0 duplicate rows existed in either).

## Fix 5 — merchant/country risk lookups leaked val/test labels into training

**Commit:** `f67c0a1`
**Files:** `src/common/splits.py` (new), `src/processing/pipeline.py`,
`src/processing/views.py`, `src/ml/train.py`,
`tests/unit/test_processing.py`, `tests/unit/test_splits.py` (new),
`tests/unit/test_train.py` (new)

`compute_risk_lookups()` computed smoothed merchant/country fraud rates
from the *entire* labeled batch, and `src/ml/train.py` independently
re-derived its own `train_test_split` afterward — so a merchant or country
that was fraud-heavy only in what became the val/test partition still
inflated the lookup those same val/test rows were later scored against, a
form of target leakage.

Added `src.common.splits.assign_split(df, seed)` — a single, stratified
70/15/15 train/val/test split on `is_fraud` — as the one source of truth
for partitioning. The pipeline now assigns `split` immediately after the
CLEAN stage, before any lookup is computed; `compute_risk_lookups` runs
only on `split == "train"` rows; `split` is carried through enrichment and
persisted in the FEATURES parquet (`views.FEATURES_COLUMNS`); and
`src/ml/train.py` no longer calls `train_test_split` at all — it reads the
persisted `split` column, guaranteeing evaluation uses exactly the
partition the lookups were fit on.

New tests cover: a merchant that is fraud-only in val/test still scores at
the neutral default (proving no leakage), `assign_split` determinism and
stratified proportions, and that `train.py`'s partition sizes match the
persisted column exactly — including a deliberately uneven split, proving
`_split()` never re-derives its own proportions.

## Post-fix verification: full pipeline re-run and retrain

With all five fixes committed, the batch pipeline and model were re-run
end to end so the live system reflects Fix 5 (the only fix that changes
model-training behavior):

- `python -m src.processing.pipeline` — 50,000 rows in, 0 rejected at every
  stage, `split` column confirmed stratified exactly 70/15/15
  (35,000 / 7,500 / 7,500) with a uniform 0.03 fraud rate across all three
  partitions.
- `./scripts/train_model.sh` — registered **model version 6** as
  `champion` (registration logic unchanged from every previous retrain).

| metric | v5 (pre-fix, leaky) | v6 (post-fix, train-only lookups) | Δ |
|---|---|---|---|
| precision | 0.9907 | 0.9775 | −0.0132 |
| recall | 0.9511 | 0.9644 | +0.0133 |
| f1 | 0.9705 | 0.9709 | +0.0004 |
| roc_auc | 0.9961 | 0.9983 | +0.0022 |

Reading: precision fell and recall rose by almost the same amount — the
expected signature of removing a leak that had let the lookup partially
"see" val/test fraud labels before being scored against those same rows.
F1 is flat and roc_auc improved, so v6 is not a regression; it is a more
honest measurement of the model's real generalization.

The FastAPI scoring service (long-running, loads the champion model once
at startup) was restarted via `scripts/run_api.sh` to pick up v6;
`/health` and `/score` were re-verified live afterward.

## Test suite

Full suite after all five fixes: `pytest -q` → **79 passed** (up from the
v1 build's 59 — 20 new tests added across the five fixes, none removed).

## Scope notes

- No infrastructure was started, stopped, or reset during this pass — the
  existing Docker Compose stack (Postgres, MinIO, Redpanda, Redpanda
  Console, MLflow) was already running throughout.
- `reset_demo.sh` was not run; no Docker volumes, data, or
  already-registered MLflow model versions were deleted.
- The champion-alias assignment mechanism itself (`train.py` registering a
  new version and aliasing it `champion`) was not changed — only its input
  data (via Fix 5) and, separately, which MLflow flavor loader reads it
  back (Fix 2).

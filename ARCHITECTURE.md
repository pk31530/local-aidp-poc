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

## v1.3 fraud intelligence

Built on `feat/aidp-v1-3-fraud-intelligence` on top of the v1.2 control
plane. A separate, seven-channel synthetic fraud-alert triage platform
(`src/fraud_intel/`) sharing v1.2's `RunLifecycle`/typed-config
conventions but with its own tables (migrations `003`/`004`), its own CLI
subcommands (`aidp fraud-intel ...`, `aidp alerts ...`), and its own
read-only dashboard tab. It does not replace or modify the v1.1/v1.2
single-channel `transactions`/`fraud_decisions` pipeline described above.

**This section documents what Phase 7B actually verified, in this exact
POC repository, against `aidp_test` — not aspirations.** As of Phase 7B
completion: seven channels (`online_banking`, `mobile_deposit`, `ach`,
`wire`, `atm`, `debit_card`, `p2p`), each with exactly one `OPERATIONAL`
`channel_model_bundles` row, 35,000 generated `channel_events`, 3,134
`source_alerts`/`fraud_alerts`, 3,134 current-operational-bundle
`alert_evidence` rows (3,575 including ACH's retired bundle-3 history),
and 3,134 eligible `label_assessments`. All of it is **synthetic,
local-only POC evidence** — see "POC scope and disclaimers" below.

### Seven channels, one shared framework

`src/fraud_intel/registry.py` (`get_channel_adapter(channel)`) is the one
place a channel is "known" — one shared training/scoring/evaluation
framework parameterized per channel, not seven copies. Each channel has
its own generator (`src/fraud_intel/generator/<channel>.py`), payload
schema (`src/fraud_intel/events/<channel>.py`), feature adapter, and
config files (`config/fraud_intel/rules_<channel>.yaml`,
`graph_policy_<channel>.yaml`, `ensemble_policy_<channel>.yaml`).

### Generation-scoped everything

Every real generate/train/score/labels-assess/evaluate command requires
an explicit `--generation-run-id` (and, for generation itself,
`--database`) — there is no silent "whichever data happens to be in the
table" default anywhere in this subsystem, and no command may ever fall
back to the `.env` default `POSTGRES_DB` (`aidp`). `generation_run_id`/
`dataset_version` are deterministic SHA-256 fingerprints of
`(channel, count, seed, reference_date, GENERATION_SPEC_VERSION)`
(`src/fraud_intel/cli_data_access._generation_identity()`), so re-running
the exact same generation command is a safe, idempotent no-op
(`ON CONFLICT ... DO NOTHING`) that recomputes the identical IDs rather
than minting new ones.

### Shared cross-channel, customer-scoped feature history

`src/fraud_intel/features/history.py`'s `select_customer_historical_events()`/
`select_customer_historical_source_alerts()` are the **single** shared
selectors used by all three of: real training
(`src/fraud_intel/models/training.py::_build_supervised_population()`),
real scoring (`src/fraud_intel/scoring/dispatch.py::_PostgresScoringDataAccess.list_pending()`),
and live-resolved-alert scoring-context reconstruction
(`src/fraud_intel/cli_data_access.py::load_resolved_alert_scoring_contexts()`).
History is **customer-scoped across ALL seven channels** (guide §15 — the
1,000-customer synthetic pool, `generate_customers()`, is
channel-independent and reused by every channel with the same seed), and
**strictly earlier than the target event's own timestamp** — ties broken
deterministically by `(event_timestamp, event_id)`, never insertion order.
This was a real corrective pass, not a design given up front: an initial
version scoped training's own history construction to one channel only,
which silently diverged from real scoring's (always-correct) cross-channel
context, first surfaced as a real ACH score mismatch between a
pre-promotion diagnostic and real persisted scoring. Fixed by extracting
the shared selectors above (commit `6a4683c`), verified by a real,
zero-diff parity re-derivation across all affected alerts.

### Immutable alert identity, versioned evidence, current-vs-initial state

`fraud_alerts` is the alert's immutable identity row, written once
(`create_alert_if_new()`, first-write-wins on `(source_system,
source_alert_id)`) — its own `initial_operational_priority_score`/
`initial_priority_band`/`initial_ensemble_policy_version` fields are
frozen at first-scoring time and **never updated by a later rescore**.
`alert_evidence` is separate, append-only, versioned scoring state — one
row per scoring attempt, uniquely keyed `(alert_id, score_execution_id)`.
"Current" state always means the alert's **latest** evidence row, selected
by `scored_at DESC, evidence_id DESC` (the `LATEST_EVIDENCE_ORDER_SQL`
constant in `src/fraud_intel/alerts/queue.py`) — with **no** bundle
filter (the repository's one, consistently-applied contract is "absolute
latest evidence," not "current OPERATIONAL bundle's evidence"). Both
`aidp alerts show` (single alert) and `aidp alerts list` / the dashboard's
ranked queue (batch, via `list_alerts_with_current_state()`, a single
`LEFT JOIN LATERAL` query — no N+1) source "current" from this exact same
rule, so they can never disagree. This, too, was a real corrective pass:
`aidp alerts list` originally read only `fraud_alerts.initial_priority_band`,
so a rescored alert's real current band was invisible to it — found for
real when ACH's replacement-bundle promotion (bundle 3 → bundle 4)
produced real MEDIUM-band evidence that `alerts list --priority-band
MEDIUM` still reported as zero rows.

### Manual, whole-bundle promotion with lifecycle auditing

A `channel_model_bundles` row bundles nine required components (GBM/LR-
shadow/anomaly model versions, preprocessing artifact, feature-schema
version, and four policy versions: rule set, graph policy, ensemble
policy, reason-code version) as one atomic, immutable unit —
`validate_promotion_eligible()` refuses a bundle missing any of the nine.
Promotion (`aidp fraud-intel promote`) is always a manual, explicit
`--promoted-by <operator>` action; there is no automatic promotion
anywhere. `promote_bundle()` (`src/fraud_intel/models/promotion.py`)
verifies every component/policy against the CURRENTLY loaded config before
promoting, locks the channel's full bundle history (`FOR UPDATE`) inside
one transaction, retires the previous `OPERATIONAL` bundle (if any) and
promotes the candidate atomically, and — for a channel's **first-ever**
promotion only — freshly recomputes the cold-start promotion gate from the
candidate's own immutable training-time held-out report (never a cached
value). A second-or-later promotion for an already-`OPERATIONAL` channel
does not re-run that gate at the code level (verified directly in this
build: ACH's replacement promotion took the "warm/replacement" path) — the
non-persistent full-candidate diagnostic remains this project's own
process-level governance practice for that case, not something
`promote_bundle()` itself enforces. Every promotion writes one
`model_promotion` row via the shared `RunLifecycle`, with the full gate
evidence, provenance, and component/policy versions in `artifacts`.

### Cold-start vs. live evaluation

`aidp fraud-intel evaluate` has two distinct, clearly-labeled modes,
chosen automatically by whether the channel currently has an
`OPERATIONAL` bundle: `candidate_training_holdout` (no `OPERATIONAL`
bundle yet — evaluates the candidate's own immutable training-time
held-out test-split report) and `live_resolved_alerts` (an `OPERATIONAL`
bundle exists — evaluates real, scored, resolved alerts, with an optional
shadow-candidate comparison). A caller cannot force cold-start mode for a
channel that already has an `OPERATIONAL` bundle; the non-persistent
full-candidate diagnostic (`score_source_alert()` run in memory, no
`fraud_alerts`/`alert_evidence` write, no `RunLifecycle`) is this
project's own real production-code-path answer for that case.

### The read-only dashboard

`src/dashboard/fraud_intel_tab.py` — a "Fraud Intelligence" tab in the
existing Streamlit dashboard, entirely separate from the legacy v1.1
tabs. Strictly read-only: it only ever calls
`create_default_alert_queue_store(database).list_alerts_with_current_state()`/
`.get_alert()`/`.get_latest_evidence()`/`.list_dispositions()` (a
Phase-8-added pure getter, alongside the existing two) — never any
write function. It ranks the current-state queue by
`current_operational_priority_score` (deterministic tie-break: score
desc, `current_scored_at` asc, `alert_id` asc), keeps LOW-band alerts
visible by default, and never displays `scenario_id`/
`synthetic_scenario_label`/any other synthetic-generator ground truth —
those fields don't exist on any type this tab reads, so there is nothing
to accidentally leak. Disposition **capture** remains exclusively
`aidp alerts disposition` (CLI-only); the dashboard only ever reads
disposition history.

### POC scope and disclaimers

Every policy file except `online_banking`'s (which predates the field)
carries `calibration_status: UNVALIDATED_POC_DEFAULT` and a
`promotion_note` stating thresholds are an untuned placeholder — verify
the exact wording in `config/fraud_intel/ensemble_policy_<channel>.yaml`
before quoting it, rather than assuming it's identical across channels.
Real, Phase-7B-verified per-channel results (synthetic data only):
Wire's fixed-threshold confusion matrix has 21 legitimate alerts in its
MEDIUM band (**not** a perfect result); ATM's fixed-threshold recall is
10.7% (14/131) — capacity-ranked review, not the fixed 0.40 threshold, is
this POC's intended interpretation for it; Debit Card and P2P achieved
perfect (1.0/1.0) precision/recall/ROC-AUC/PR-AUC, which most likely
reflects unusually clean synthetic-generator scenario separability (P2P's
`P2P_NEW_RECIPIENT_NO_MEMO` rule alone perfectly separates its synthetic
fraud population) rather than validated real-world model quality; Wire,
ATM, Debit Card, and P2P produced **zero** HIGH-band alerts in real
scoring (mathematically reachable given the pinned ensemble weights, but
never empirically reached); Mobile Deposit is the one channel whose HIGH
alerts come through a real, firing `MANDATORY_REVIEW`-category rule
(`MOBILE_DEPOSIT_DUPLICATE_IMAGE_AND_CAR_LAR_MISMATCH`), not the weighted
score threshold. **None of this constitutes Citizens Bank production or
regulatory approval, automatic governance, or a claim of real-world
accuracy** — it is a local, synthetic proof-of-concept only.

## Real issues hit and fixed during the build

See TROUBLESHOOTING.md for the full detail; summary:
- `minio/minio` and `minio/mc` no longer resolve on Docker Hub — moved to `quay.io`.
- XGBoost needs `libomp` on macOS (not bundled) — documented, with an automatic RandomForest fallback if missing.
- MLflow's S3 client needs explicit credentials in every process that talks to it directly — centralized in `src/common/mlflow_setup.py`.
- Redpanda (even at the newest available release) doesn't support Kafka Fetch API v12, which newer `confluent-kafka`/librdkafka versions request by default — pinned `confluent-kafka==2.3.0`.
- A naive retry-with-backoff implementation that reuses one Postgres connection across retries doesn't actually recover from an outage — fixed to acquire a fresh connection per attempt.
- Pooled API connections can go stale after a Postgres restart — fixed with checkout-time validation and self-healing in `src/api/main.py`.

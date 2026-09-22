# Debit Card — Full Fraud-Intelligence Lifecycle Demonstration

A scripted, end-to-end walkthrough of the AiDP v1.3 fraud-intelligence lifecycle for the
**Debit Card** channel, executed against a **completely isolated** demonstration stack.

```
generate → generation retry → population verification → chronological split preview
→ train → candidate evaluation → non-persistent full-candidate diagnostic
→ MANUAL APPROVAL PAUSE
→ promote → score → scoring retry → labels assess → label retry
→ live evaluation → final audit
```

| Parameter | Value |
|---|---|
| channel | `debit_card` |
| count | `5000` |
| seed | `99` |
| reference date | `2026-09-22` |
| database | `aidp_demo` |
| analyst capacity | `100` |
| recall target | `0.8` |
| promoted by | `prabhat-kumar` |

---

## 1. What "isolated" means here, and why it is safe

Everything this demo touches is a second, parallel stack. Nothing it does can read or write
the normal POC's data.

| Concern | Shared POC stack | This demo |
|---|---|---|
| Compose project | `aidp-poc` (`docker-compose.yml`) | `aidp-demo` (`docker-compose.demo.yml`) |
| Containers | `aidp-postgres`, `aidp-minio`, `aidp-mlflow`, `aidp-redpanda`… | `aidp-demo-postgres`, `aidp-demo-minio`, `aidp-demo-mlflow` |
| Network | `aidp-network` | `aidp-demo-network` |
| Volumes | `aidp-poc_postgres_data`, `aidp-poc_minio_data`, … | `aidp-demo_demo_postgres_data`, `aidp-demo_demo_minio_data` |
| Databases | `aidp`, `aidp_test`, `mlflow` | `aidp_demo`, `mlflow` — **`aidp` and `aidp_test` do not exist in this cluster** |
| Postgres port | `127.0.0.1:5432` | `127.0.0.1:55432` |
| MinIO | `127.0.0.1:9000` / console `9001` | `127.0.0.1:59000` / console `59001` |
| MLflow | `http://127.0.0.1:5001` | `http://127.0.0.1:55001` |
| Streamlit | `127.0.0.1:8501` | `127.0.0.1:58501` |
| Env file | `.env` (**never read or written by the demo**) | `.env.demo.example` + `.env.demo.local` |

The isolation mechanism is taken from the source, not invented:

* `src/common/config.py` builds `Settings` with **pydantic-settings**, which gives real process
  environment variables precedence over `.env`. The script exports the demo values, so every
  CLI process resolves Postgres/MinIO/MLflow to the isolated stack while `.env` is untouched.
* `src/dashboard/data.py::_dashboard_database()` resolves the dashboard's database from
  `settings.postgres_test_db` (not `postgres_db`). The demo therefore pins **both**
  `POSTGRES_DB` and `POSTGRES_TEST_DB` to `aidp_demo`, which removes the last code path that
  could ever resolve to `aidp_test`.
* `docker-compose.yml` hardcodes `container_name` for every service and hardcodes MinIO's and
  MLflow's host ports, so a second project cannot be built from it. `docker-compose.demo.yml`
  is therefore a separate file with distinct names, ports, network and volumes.
* `infrastructure/demo/postgres-init.sql` deliberately does **not** create `aidp_test`
  (the shared `infrastructure/postgres/init.sql` does). The shared, unmodified
  `infrastructure/postgres/lib/schema.sql` is applied to `aidp_demo`.

The script proves non-contact four independent ways, at preflight and again after every
write-producing stage:

1. the effective `Settings` resolve only to `aidp_demo`;
2. `SELECT current_database()` through the application's **own** connection factory returns
   `aidp_demo`;
3. `SELECT datname FROM pg_database` on the demo cluster contains **no** `aidp` and no
   `aidp_test` — so nothing running against it can reach one, even by mistake;
4. every `--database` argument issued during the run, recorded in `commands.log`, is
   `aidp_demo`.

---

## 2. Prerequisites

* macOS or Linux, Docker Desktop running, `docker compose` v2, `curl`, `git`, `bash`.
* The repository checked out at `local-aidp-poc`, with a working tree you are happy to run from
  (the script reports the branch and clean/dirty state; a dirty tree is allowed but the
  recorded `git_sha` provenance will not describe a committed state).
* `.venv` present at the repository root with `requirements.txt` installed.
* Roughly **4 GB free RAM** and **3 GB free disk** for the second Postgres/MinIO/MLflow stack.
* Free host ports: `55432`, `59000`, `59001`, `55001`, and `58501` if you start the dashboard.
* The shared `aidp-poc` stack may keep running throughout — the demo never talks to it.

No secrets are needed. On first run the script generates demo-only Postgres/MinIO credentials
into `.env.demo.local` (mode `600`, git-ignored). `.env.demo.example` contains no credentials.

---

## 3. Exact terminal commands

### Terminal 1 — the demo

```bash
cd /path/to/local-aidp-poc

# (optional) prove the isolation and infrastructure without running the lifecycle
./scripts/demo_debit_card_full_lifecycle.sh --preflight-only

# Part 1: stages A–G, then HALT for manual approval. Nothing is promoted.
./scripts/demo_debit_card_full_lifecycle.sh

# ...read the candidate report, then explicitly approve:
# Part 2: stages I–O.
./scripts/demo_debit_card_full_lifecycle.sh --resume-from promote --approve-promotion
```

### Terminal 2 — observation (optional but recommended)

```bash
cd /path/to/local-aidp-poc

# Load ONLY the isolated demo environment into this shell.
set -a; . ./.env.demo.example; . ./.env.demo.local; set +a
. .venv/bin/activate

# Follow the demo's own stdout while stage output scrolls in Terminal 1
tail -f "$(ls -dt .demo/logs/* | head -1)/demo.stdout.log"

# Or watch the isolated database directly
watch -n 2 'docker exec aidp-demo-postgres psql -U aidp_demo -d aidp_demo -c "
  SELECT (SELECT count(*) FROM channel_events)       AS events,
         (SELECT count(*) FROM source_alerts)        AS source_alerts,
         (SELECT count(*) FROM fraud_alerts)         AS alerts,
         (SELECT count(*) FROM alert_evidence)       AS evidence,
         (SELECT count(*) FROM label_assessments)    AS assessments,
         (SELECT count(*) FROM channel_model_bundles) AS bundles;"'
```

### Isolated UIs

| What | URL | Notes |
|---|---|---|
| MLflow (demo) | <http://127.0.0.1:55001> | experiment `fraud-intel-debit_card`; models `fraud-detection-model-fraud-intel-debit-card-{gbm,lr-shadow,anomaly}` |
| MinIO console (demo) | <http://127.0.0.1:59001> | user `aidpdemoadmin`, password = `MINIO_SECRET_KEY` in `.env.demo.local`; bucket `mlflow-artifacts` |
| MinIO S3 API (demo) | <http://127.0.0.1:59000> | what the MLflow client uploads artifacts to |
| Streamlit dashboard (demo) | <http://127.0.0.1:58501> | see below |
| Shared stack MLflow / MinIO | <http://127.0.0.1:5001>, <http://127.0.0.1:9001> | **untouched** — useful to show side by side that they stay empty of `debit_card` |

### Dashboard startup (Terminal 2 or 3)

`src/dashboard/data.py` resolves its database from `settings.postgres_test_db`, which the demo
environment pins to `aidp_demo`. Export the demo environment **first**, then start Streamlit:

```bash
cd /path/to/local-aidp-poc
set -a; . ./.env.demo.example; . ./.env.demo.local; set +a
. .venv/bin/activate
streamlit run src/dashboard/app.py \
  --server.address 127.0.0.1 --server.port 58501 --server.headless true
# → http://127.0.0.1:58501
```

Starting it without exporting the demo environment first would point the dashboard at the
shared `aidp_test` database. Always export first.

---

## 4. Expected duration

These are **estimates** on a 16 GB laptop with the images already pulled, not measured timings.
The first run also builds the MLflow image and initialises the Postgres/MinIO volumes.

| Stage | Estimate |
|---|---|
| A — preflight (first run, incl. image build + volume init) | 2–5 min |
| A — preflight (subsequent runs) | 20–40 s |
| B — generate 5 000 events | 10–40 s |
| C — generation retry | 5–20 s |
| B/C — population verification | 5–15 s |
| D — split preview | 20–90 s |
| E — train (3 models, 3 MLflow runs, 3 registrations, artifact upload) | 1–4 min |
| F — candidate evaluation | 5–15 s |
| G — full-candidate diagnostic (per-alert graph + model scoring) | 1–5 min |
| H — approval halt | instant |
| I — promote | 10–30 s |
| J — first scoring | 1–5 min |
| K — scoring retry | 5–20 s |
| L — label assessment | 10–40 s |
| M — label retry | 10–40 s |
| N — live evaluation | 10–40 s |
| O — final audit | 10–30 s |
| **Part 1 (A–G)** | **~5–15 min** |
| **Part 2 (I–O)** | **~3–10 min** |

Budget ~25 minutes of wall clock for a live presentation, plus talking time.

---

## 5. Stage-by-stage: what to show, what to say, what to expect

Every stage prints `[PASS]` lines as it goes and writes a JSON artefact under
`.demo/reports/<UTC timestamp>/`. Raw command output is kept with **stdout and stderr in
separate files** under `.demo/logs/<UTC timestamp>/`.

### A — Infrastructure preflight

**Show:** the `[PASS]` block, then `docker ps` side by side with the shared stack.

**Say:** "Before anything runs, we prove the blast radius. This demo has its own Postgres, its
own MinIO and its own MLflow. The demo cluster does not even contain a database called `aidp`
or `aidp_test` — so there is no path, accidental or otherwise, to production POC data."

**Asserts:** repository identity; git branch and clean/dirty state; `.venv`; the isolated stack
is healthy; only `aidp-demo-*` containers belong to the project; `current_database()` is
`aidp_demo`; `debit_card` has **zero** events, source alerts, labels, bundles, alerts,
evidence and assessments; the isolated MLflow has **no** experiment and **no** registered
version for the channel.

**Also computes, without writing anything:** the real deterministic generator is run purely in
memory to derive the *expected* source-alert and fraud counts. They are never hardcoded.

**Artefacts:** `preflight_population.json`, `expected_population.json`.

```json
{ "event_count": 5000, "source_alert_count": <derived>, "fraud_count": <derived>,
  "alerted_fraud_count": <derived>, "alerted_legitimate_count": <derived>,
  "generation_run_id": "genrun-…", "dataset_version": "dsv-…",
  "max_pending_alerts_per_scoring_run": 500 }
```

> **Stop condition.** If the derived source-alert population exceeds
> `src/fraud_intel/scoring/dispatch.py::_MAX_PENDING_ALERTS_PER_RUN` (500), preflight fails
> with an explicit message. A single `fraud-intel score` run would otherwise silently process
> only the first 500 alerts, and the demo's "pending equals the whole population" and
> "retry pending is zero" gates could not hold. See §9.

---

### B — First generation

**Command run:**

```
python -m src.cli fraud-intel generate --channel debit_card --count 5000 --seed 99 \
  --reference-date 2026-09-22 --database aidp_demo --json
```

**Say:** "This is the data-engineering boundary. We generate a deterministic, synthetic
population of debit-card events. Each event carries a *simulated upstream* source alert — that
is what the bank's existing rule engine would have raised — and a separate synthetic ground-
truth label that no scoring or feature code ever reads."

**Expected JSON fields:** `channel`, `requested_count`, `inserted_event_count`,
`existing_event_count`, `source_alert_count`, `label_count`, `generation_run_id`,
`dataset_version`, `reference_date`, `seed`.

**Asserts:** `inserted_event_count = 5000`; `existing_event_count = 0`; `label_count = 5000`;
`source_alert_count` equals the value derived from the generator at preflight; exactly one
`generation_run_id` and one `dataset_version` persisted; zero duplicate events, source alerts
or labels; events/source-alerts/labels reconcile against the generator's own counts.

**Artefacts:** `generate.json`, `population_after_generate.json`.

---

### C — Generation retry

**Command run:** *byte-identical* to stage B.

**Say:** "Re-running the exact same command must be a no-op. The generation identity is a hash
of the full specification — channel, count, seed, reference date, generator version — so a
retry recomputes the same `generation_run_id` instead of minting a phantom one."

**Asserts:** `inserted_event_count = 0`; `existing_event_count = 5000`; identical
`generation_run_id` and `dataset_version`; the database snapshot is byte-identical before and
after.

**Artefacts:** `generate_retry.json`, `population_before/after_generate_retry.json`.

---

### B/C — Population verification

**Say:** "Before we model anything, we reconcile what is on disk against what the generator
said it produced."

**Asserts:** all of stage B's reconciliation, re-read from Postgres; plus the ground-truth
split of the *source-alerted* subpopulation (fraud vs legitimate) matches the generator.

**Artefacts:** `population_verified.json`.

---

### D — Chronological split preview *(read-only)*

**Say:** "This is the data-science boundary, and the most important slide for anyone worried
about leakage. We use the **real production loaders** and the **shared historical feature
selectors** — the same functions real scoring uses — then apply the real chronological split.
Rows are cut by time, never randomly, and a timestamp group is never split across a boundary."

**Reports:** supervised population size and its fraud/legitimate counts; train / calibration /
test row counts and class counts; each partition's timestamp range; realized split fractions;
purged row count; row-id overlap between partitions; equal-timestamp boundary overlap;
the configured minimum row/class gates.

**Asserts:** every overlap count is `0`; partitions are in chronological order; the real
`_validate_partition()` gates (minimum rows **and** both classes present) pass for train,
calibration and test.

**Artefacts:** `split_preview.json`.

> **Stop condition.** Any gate failure, any row-id overlap, any equal-timestamp straddle, or
> any out-of-order partition stops the demo here. Training is never attempted on a population
> that would leak.

---

### E — Training

**Command run:**

```
python -m src.cli fraud-intel train --channel debit_card \
  --generation-run-id <captured at stage B> --database aidp_demo --json
```

**Say:** "Training produces a **CANDIDATE** bundle, not a live model. Three components are
fitted — a calibrated gradient-boosted primary, a logistic-regression *shadow* that is logged
and never scored on, and an unsupervised anomaly detector fitted on the training window only.
Nothing is aliased 'champion'. Nothing is promoted."

**Expected JSON fields:** `run_id`, `bundle_id`, `bundle_version`, `gbm_model_version`,
`lr_model_version`, `anomaly_model_version`, `dataset_version`,
`supervised_population_hash`, `source_generation_run_id`, `source_dataset_version`,
`gbm_evaluation`, `lr_shadow_evaluation`, `realized_split_fractions`.

**Asserts:** exactly one bundle row, status `CANDIDATE`; all nine
`REQUIRED_OPERATIONAL_COMPONENTS` populated; exactly one `train`/`SUCCESS` lifecycle row and
**no** `fraud_score`, `label_eligibility` or `model_promotion` row; exactly three MLflow runs;
all three registered model versions `READY` and pointing at the run ids the bundle itself
records; `preprocessor.json` and `anomaly_normalization.json` reload successfully from MLflow;
generation / dataset / supervised-population-hash provenance all agree; zero alerts, evidence
or assessments were created.

**Show:** MLflow at <http://127.0.0.1:55001> — the three runs and three registered models —
then the **shared** MLflow at <http://127.0.0.1:5001>, which has none of them.

**Artefacts:** `train.json`, `train_verify.json`.

---

### F — Candidate evaluation *(read-only)*

**Command run:**

```
python -m src.cli fraud-intel evaluate --channel debit_card \
  --generation-run-id <captured> --candidate-bundle-version <captured> \
  --capacity-mode count --capacity-value 100 --recall-target 0.8 \
  --database aidp_demo --json
```

**Say:** "There is no operational bundle yet, so there is structurally no live scored
population to evaluate against. The CLI detects that and switches to `candidate_training_holdout`
mode: it reads the candidate's own immutable, training-time held-out report. That is real
evidence — but it is *held-out training* evidence, and the output says so."

**Expected JSON fields:** `evaluation_mode` (`candidate_training_holdout`),
`candidate_bundle`, `candidate_evaluation.gbm_evaluation`,
`candidate_evaluation.lr_shadow_evaluation`, `candidate_evaluation.realized_split_fractions`,
`promotion_gate_result.{passed,reasons,test_fraud_prevalence,gbm_pr_auc,split_class_counts}`,
`rules_only_baseline`, `disclaimer`.

**Asserts:** mode is `candidate_training_holdout`; generation and dataset version match;
`promotion_gate_result.passed` is `true`; the database is unchanged before and after.

> **Stop condition.** A failed promotion gate stops the demo. Promotion is blocked.

**Artefacts:** `evaluate_candidate.json`.

---

### G — Full-candidate diagnostic *(nothing is persisted)*

**Say:** "This is the slide that usually gets asked for and usually does not exist. Before we
promote anything, we score the **entire generated source-alert population** with the candidate
artefacts, through the real pure scoring path — and we write nothing. No alert rows. No
evidence rows. No lifecycle rows. It is a look, not a commitment."

**Reports:** LOW/MEDIUM/HIGH counts; score min / mean / max; the band × ground-truth cross-tab;
**every** rule id in the channel's rule set with its firing count (including rules that fired
zero times) and its category; component OK and error counts; degraded alert count; fraud
captured in the top 100; precision@100 and recall@100; and the HIGH band broken into
*mandatory-review* versus *threshold-exceeded* versus *component-failure floor*.

**Asserts:** the selected population equals the full generated source-alert population; every
alert scored without error; `fraud_alerts`, `alert_evidence` and `pipeline_runs` counts are
identical before and after.

**Say, pointing at the warning:** "`config/fraud_intel/ensemble_policy_debit_card.yaml`
declares `calibration_status: UNVALIDATED_POC_DEFAULT`. The ensemble weights and the
LOW/MEDIUM/HIGH thresholds behind every number on this screen are an untuned placeholder
copied from the online-banking channel. These numbers demonstrate that the machinery works
end to end. They are not a claim about accuracy."

**Artefacts:** `diagnostic.json`, `diagnostic_scores.json` (per-alert scores, reused in stage J).

---

### H — Manual approval pause

The script **stops**. It prints the candidate report paths and the captured runtime
identifiers, and exits `0` with `[RESULT] Stages A–G PASSED. Awaiting approval.`

**Say:** "Nothing has been promoted. A model becomes operational because a named human decided
it should, not because a script reached the end of its own list. Promotion needs a second,
explicit invocation with an explicit approval flag."

```bash
./scripts/demo_debit_card_full_lifecycle.sh --resume-from promote --approve-promotion
```

Running `--resume-from promote` **without** `--approve-promotion` fails immediately with a
message telling you to add the flag. There is no path through this script that promotes
without it.

---

### I — Promotion

**Command run:**

```
python -m src.cli fraud-intel promote --channel debit_card \
  --bundle-version <captured> --promoted-by prabhat-kumar --database aidp_demo --json
```

**Say:** "Promotion re-verifies the bundle from scratch — every required component present,
every registered MLflow version `READY` and pointing at the right run, and, because this is the
channel's first-ever promotion, the cold-start gate is recomputed rather than trusted from the
earlier `evaluate` call. `promoted_by` is mandatory even when the gate passes: passing the gate
makes a candidate *eligible*, never *promoted*."

**Asserts:** exactly one `OPERATIONAL` bundle for the channel and it is the one we trained;
`promoted_by = prabhat-kumar`; exactly one `model_promotion`/`SUCCESS` lifecycle row; every
model version and pinned policy version byte-identical before and after promotion; **no**
scoring occurred (zero `fraud_score` runs, zero alerts, zero evidence).

**Artefacts:** `promote.json`, `promote_verify.json`, `bundle_components_before/after_promote.json`.

---

### J — First scoring

**Command run:**

```
python -m src.cli fraud-intel score --channel debit_card \
  --generation-run-id <captured> --database aidp_demo --json
```

**Say:** "Now it is real. Scoring is scoped to one explicit generation and to the *current*
operational bundle — it can never silently rescore everything a channel has ever produced.
Each source alert becomes one alert-queue row plus one immutable evidence row carrying the full
provenance trail: bundle id, every component version, every policy version, the config hash and
the git sha."

**Expected JSON fields:** `run_id`, `channel`, `generation_run_id`, `source_dataset_version`,
`bundle_id`, `bundle_version`, `pending_count`, `records_processed`, `records_rejected`,
`alerts[]`.

**Asserts:** `pending_count` equals the generated source-alert population;
`records_processed = pending_count`; `records_rejected = 0`; exactly one `fraud_alerts` row and
exactly one current-bundle `alert_evidence` row per source alert; zero duplicates; zero alerts
missing current-bundle evidence; zero evidence rows referencing any other bundle; **zero
degraded** evidence; every component status `OK`; exactly one `fraud_score`/`SUCCESS` run; and
**the persisted band and score for every alert match the stage-G diagnostic exactly**.

**Say:** "That last check is the point of stage G. What we previewed before promotion is
bit-for-bit what the platform persisted after it."

**Artefacts:** `score.json`, `score_verify.json`.

---

### K — Scoring retry

**Command run:** identical to stage J.

**Asserts:** `pending_count = 0`; `records_processed = 0`; `records_rejected = 0`;
`alerts = []`; every alert and evidence row byte-identical before and after. One additional
`fraud_score`/`SUCCESS` lifecycle row is expected and correct — the retry itself is an audited
event.

**Artefacts:** `score_retry.json`, `evidence_digest_before/after_score_retry.json`.

---

### L — First label assessment

**Command run:**

```
python -m src.cli fraud-intel labels assess --channel debit_card \
  --generation-run-id <captured> --database aidp_demo --json
```

**Say:** "Scoring never creates a training label. This is the only command that does, and only
when you explicitly ask. For synthetic data the ground truth is known at generation time, so
every label matures immediately — with a reason code that says exactly that
(`SYNTHETIC_IMMEDIATE_MATURITY`). For an analyst-derived label the maturity clock starts when
the *evidence* arrived, not when the event happened."

**Expected JSON fields:** `bases_evaluated`, `assessments_inserted`, `assessments_unchanged`,
`mature_count`, `immature_count`, `eligible_count`, `resolved_fraud_count`,
`resolved_legitimate_count`, `unresolved_count`, `policy_version`, `source_dataset_version`.

**Asserts:** `bases_evaluated` equals the source-alert population;
`assessments_inserted = bases_evaluated`; `assessments_unchanged = 0`; every assessment
`MATURE` and eligible; `unresolved_count = 0`; `resolved_fraud_count` /
`resolved_legitimate_count` match the generated ground truth exactly; one assessment per alert
(zero duplicates); every label sourced `SYNTHETIC_GENERATOR` with a `NULL` disposition id.

**Artefacts:** `labels.json`, `labels_verify.json`, `label_digest_after_first_assessment.json`.

---

### M — Label retry

**Command run:** identical to stage L.

**Asserts:** `assessments_inserted = 0`; `assessments_unchanged = bases_evaluated`; the
`label_assessments` table is byte-identical. The table is append-only — an exact retry appends
nothing, while a genuinely changed basis would always append a new row.

**Artefacts:** `labels_retry.json`, `label_digest_before/after_retry.json`.

---

### N — Live evaluation *(read-only)*

**Command run:**

```
python -m src.cli fraud-intel evaluate --channel debit_card \
  --generation-run-id <captured> --capacity-mode count --capacity-value 100 \
  --recall-target 0.8 --database aidp_demo --json
```

**Say:** "Now that an operational bundle exists and the population is scored and resolved, the
same command switches to `live_resolved_alerts` mode. And notice the capacity framing: we are
not asking 'is the model accurate', we are asking 'if one analyst can review 100 alerts today,
how much of the fraud do they see, and how few alerts would they need to review to catch 80%
of it'."

**Reports:** evaluated population; precision; recall; **F1**; PR-AUC; ROC-AUC; Brier score;
confusion matrix at the operational threshold; LOW/MEDIUM/HIGH counts; precision@100;
recall@100; fraud captured in the top 100; minimum alerts required for 80% recall; workload
reduction; the rules-only baseline; and the POC disclaimer.

**Asserts:** mode is `live_resolved_alerts`; the operational bundle id matches; the evaluated
population equals the resolved-eligible population equals the source-alert population;
`missing_current_bundle_evidence_count = 0`; `degraded_evidence_count = 0`; and the database
snapshot — **including `pipeline_runs`** — is identical before and after, proving that
evaluation writes nothing and records no lifecycle row.

**Artefacts:** `live_evaluate.json`, `live_evaluate_report.json`.

> F1 is **derived by this demo script** from the CLI's own precision and recall.
> `ChannelEvaluationResult` (`src/fraud_intel/evaluation/cross_channel.py`) has no F1 field;
> the report labels it `f1_derived_from_precision_recall` so it is never mistaken for a product
> output. See §9.

---

### O — Final audit

**Reports:** event / source-alert / label counts; bundle state; fraud-alert and evidence
counts; label-assessment counts; every lifecycle row; MLflow runs and registered versions;
the demo cluster's database list; and the git branch and working-tree state.

**Asserts:** counts match the generator's expectation; zero duplicates anywhere; zero degraded
evidence; evidence references exactly one bundle; exactly one `OPERATIONAL` bundle; **zero
`RUNNING`, `PENDING`, `FAILED` or `CANCELLED`** pipeline runs; lifecycle rows are exactly
`{train: 1, model_promotion: 1, fraud_score: 2, label_eligibility: 2}` — evaluation records
none, by design; exactly three MLflow runs and exactly one `READY` version per component; and
`aidp` / `aidp_test` were never contacted.

**Artefacts:** `audit.json`, `git_status.txt`.

---

## 6. Stop conditions

The script exits non-zero and prints `[FAIL] … [RESULT] DEMO FAILED at stage: <stage>` on any
of the following. None of them are warnings; all of them halt the run.

| Category | Examples |
|---|---|
| Wrong context | not the `local-aidp-poc` repository; not run from the repository root; no `.venv`; no `docker compose` v2 |
| Broken isolation | `POSTGRES_DB` or `POSTGRES_TEST_DB` is not `aidp_demo`; a demo endpoint equals the shared stack's; the demo cluster contains a database named `aidp` or `aidp_test`; `current_database()` is not `aidp_demo`; a `--database` argument other than `aidp_demo` appears in the command log; the `aidp-demo` project owns a container not named `aidp-demo-*` |
| Infrastructure | a demo service never becomes healthy; MinIO or MLflow health check fails |
| Dirty start | the channel already has events, alerts, bundles, evidence, assessments or MLflow versions (fresh runs only — suppressed when resuming) |
| Generator mismatch | the generator emits a count other than `--count`, duplicate event or source-alert ids, or a source-alert population above `_MAX_PENDING_ALERTS_PER_RUN` |
| Non-zero exit | any CLI command exits non-zero |
| Invalid JSON | any CLI or helper produces output that is not parseable JSON, or a JSON object carrying an `error` key |
| Unexpected count | any asserted count in stages B–O differs from the value derived at preflight |
| Rejected records | `records_rejected > 0` in any scoring run |
| Degraded evidence | any `alert_evidence.degraded = true`, or any component status other than `OK` |
| Duplicates | duplicate events, source alerts, labels, alerts, evidence rows or label assessments |
| Lifecycle failure | any `PENDING`/`RUNNING`/`FAILED`/`CANCELLED` pipeline run; the wrong number of lifecycle rows; a lifecycle row created by a stage that must not create one |
| Provenance mismatch | `source_generation_run_id` / `source_dataset_version` / `supervised_population_hash` disagreeing with the bundle row or the captured identifiers; an MLflow version pointing at the wrong run |
| Leakage/split gate | any row-id overlap or equal-timestamp straddle between partitions; partitions out of chronological order; any `_validate_partition()` failure |
| Gate failure | `promotion_gate_result.passed` is not `true` |
| Unintended writes | the diagnostic or either evaluation stage changes any table; promotion scores; training creates alerts or assessments |
| Diagnostic mismatch | a persisted band or score disagrees with the stage-G diagnostic |
| Missing approval | reaching `promote` without `--approve-promotion` |

---

## 7. Recovery and resume

All runtime identifiers are captured, never hardcoded. They live in `.demo/state.json`:

```json
{ "generation_run_id": "genrun-…", "dataset_version": "dsv-…",
  "bundle_id": 1, "bundle_version": 1, "training_run_id": 1,
  "expected_source_alert_count": …, "expected_fraud_count": … }
```

If that file is lost, the script re-derives rather than guessing:

* the generation identity is recomputed from the **real production function**
  `src.fraud_intel.cli_data_access._generation_identity` — it is a deterministic hash of
  `(channel, count, seed, reference_date, generation_spec_version)`;
* the bundle identity is read back from `channel_model_bundles`, and the script **refuses** to
  continue if there is anything other than exactly one bundle row for the channel.

### Resuming

```bash
./scripts/demo_debit_card_full_lifecycle.sh --resume-from <stage>
```

Valid stages, in order:

```
preflight generate generate_retry verify split_preview train
evaluate_candidate diagnostic approval promote score score_retry
labels labels_retry live_evaluate audit
```

* **Stage A always runs**, on every invocation, including when resuming. Isolation, health and
  the generator expectations are re-proved every time. Only the "channel must be empty"
  assertion is suppressed when resuming (it reports the current population instead).
* Resuming at or after `promote` requires `--approve-promotion`.
* Stage J compares against `.demo/diagnostic_scores.json`, which stage G writes to a stable
  path precisely so it survives into the second invocation. If it is missing, the script says
  so and tells you to `--resume-from diagnostic`.

### Common recoveries

| Situation | What to do |
|---|---|
| A stage failed mid-run | Read `.demo/logs/<ts>/*.stderr.log` for that stage, fix the cause, then `--resume-from <that stage>`. Generation, scoring and label assessment are all idempotent, so re-running a completed stage is safe. |
| Docker Desktop restarted | Just re-run; stage A brings the stack back up and waits for health. |
| Port already in use | Another process holds `55432`/`59000`/`59001`/`55001`. Free it, or edit the ports in `.env.demo.example` (they are used by both compose and the app). |
| `.env.demo.local` deleted but volumes still exist | Authentication will fail. Run `--cleanup` first, then start fresh. |
| You want a completely clean slate | `--cleanup`, then run again from the top. |
| You want to demo again without regenerating | Everything is idempotent — `--resume-from live_evaluate` re-runs just the evaluation and audit. |

---

## 8. Safe cleanup

Cleanup **never runs automatically**, including after a fully successful demo. The isolated
stack is deliberately left running so you can keep exploring MLflow, MinIO and the dashboard.

```bash
./scripts/demo_debit_card_full_lifecycle.sh --cleanup          # prompts for confirmation
./scripts/demo_debit_card_full_lifecycle.sh --cleanup --yes    # no prompt
```

**Scope — what is removed:**

* the `aidp-demo` compose project's containers (`aidp-demo-postgres`, `aidp-demo-minio`,
  `aidp-demo-createbuckets`, `aidp-demo-mlflow`);
* the `aidp-demo-network` network;
* the volumes labelled for the `aidp-demo` project only
  (`aidp-demo_demo_postgres_data`, `aidp-demo_demo_minio_data`);
* `.demo/state.json` and `.demo/diagnostic_scores.json`.

**Scope — what is never touched:**

* the shared `aidp-poc` stack, its containers, its network and its volumes;
* the `aidp` and `aidp_test` databases;
* the shared MLflow backend store and registry;
* the shared MinIO buckets;
* `.env`, `.env.demo.example`, `.env.demo.local`;
* every log and report under `.demo/logs/` and `.demo/reports/` — the evidence trail survives.

Before removing anything, the script refuses outright if the compose project name is not
`aidp-demo`, if the project owns a container whose name does not start with `aidp-demo-`, or if
a volume labelled for the project is not prefixed `aidp-demo_`. Every `docker compose`
invocation in the script passes `-p aidp-demo -f docker-compose.demo.yml` — there is no code
path that runs a bare `docker compose` command.

After teardown it prints the shared stack's containers so you can see they are still running.

---

## 9. Known limitations, derived from the source

These are properties of the repository as it stands, not of the demo script. The script
surfaces each one rather than working around it.

1. **Scoring batch cap.** `src/fraud_intel/scoring/dispatch.py` sets
   `_MAX_PENDING_ALERTS_PER_RUN = 500`. One `fraud-intel score` run therefore scores at most
   500 pending alerts. For `--count 5000` on `debit_card` the generator's own prevalences
   (~3% fraud firing at 0.85, ~5% hard-false-positive firing at 0.90, the remainder firing at
   0.02) put the expected source-alert population in the mid-400s — under the cap, but not by
   a wide margin. Preflight computes the exact deterministic figure and **fails fast with an
   explicit message** if it exceeds 500, rather than letting stages J and K quietly misreport.
   Raising the cap would be a source change, which this demo deliberately does not make.
2. **No F1 in the product.** `ChannelEvaluationResult` exposes precision, recall, PR-AUC,
   ROC-AUC, Brier and a supplementary accuracy — but no F1. Stage N derives F1 from the CLI's
   own precision and recall and labels it `f1_derived_from_precision_recall`.
3. **Untuned channel policy.** `config/fraud_intel/ensemble_policy_debit_card.yaml` declares
   `calibration_status: UNVALIDATED_POC_DEFAULT` and
   `promotion_note: "Requires Phase 7B channel-specific evaluation before promotion"`. Its
   weights and thresholds are a placeholder copy of the online-banking channel's. The demo
   changes none of them and prints the warning at stage G.
4. **No cross-generation isolation for graph evidence.** `list_pending()`'s resolved-fraud
   evidence lookup is time-scoped, not generation-scoped. In this demo the database contains
   exactly one generation, so the distinction never arises.
5. **Redpanda is not started.** The fraud-intelligence lifecycle never touches Kafka. The demo
   environment points `REDPANDA_BROKERS` at an unused port so a stray streaming command fails
   fast instead of reaching the shared broker.
6. **Promotion audit gap (documented in `promote_bundle`'s own docstring).** The bundle row is
   made `OPERATIONAL` in a different transaction from the `model_promotion` lifecycle write. If
   the lifecycle write failed, `channel_model_bundles` would remain the authoritative record
   and the run row would need manual reconciliation. Stage I asserts both, so the demo would
   catch it.
7. **The Debit Card generator is at version v2, and v1 populations are retired.** The original
   generator assigned `card_present_flag`, `cross_border_flag`, `card_token` and
   `amount_minor_units` deterministically from `is_fraud`, which made the
   `card_not_present_flag` *model feature* an exact copy of the label and produced meaningless
   perfect held-out metrics. `src/fraud_intel/generator/debit_card.py` now draws all four
   probabilistically with overlapping distributions, and
   `src.fraud_intel.cli_data_access.CHANNEL_GENERATOR_VERSION` pins this channel at `v2` so the
   corrected data can never share an identity with the pre-fix data. For
   `count=5000 seed=99 reference_date=2026-09-22` the identity is now
   `genrun-bd703bdc10869000` / `dsv-bd703bdc10869000`; the retired v1 identity was
   `genrun-95ad70f19e5ec9fc` / `dsv-95ad70f19e5ec9fc`. Event and source-alert counts are
   unchanged (5 000 / 456) because only payload fields changed, not the scenario draw. **Any
   isolated demo database still holding a v1 population must be torn down with `--cleanup`
   before rebuilding** — `load_cross_channel_customer_pool()` is deliberately not scoped to a
   generation, so leftover v1 rows for the same customers would contaminate v2 feature history.
   `tests/unit/test_fraud_intel_debit_card_label_leakage.py` pins all of this down.

---

## 10. What these numbers are — and are not

### Synthetic metrics are not production evidence

Every number this demo produces comes from data that a deterministic Python generator invented
a few minutes earlier. Read that sentence to the room before you read any metric.

* **The labels are free.** The generator decides which events are fraud and writes the answer
  down. In production, a label costs an analyst's time, a customer's phone call, a chargeback
  window, or a confirmed loss — and arrives weeks late, incompletely, and with bias about
  *which* alerts ever got investigated at all.
* **The population is conveniently shaped.** Fraud prevalence is ~3%, hard false positives are
  ~5%, and the upstream rule engine is simulated by three probabilities. Real debit-card
  fraud is rarer, adversarial, non-stationary and correlated in ways no `numpy` draw
  reproduces.
* **The thresholds are untuned.** `calibration_status: UNVALIDATED_POC_DEFAULT`. The
  LOW/MEDIUM/HIGH boundaries and the ensemble weights were copied from another channel so that
  scoring was structurally possible at all.
* **The model never met a distribution shift.** Train, calibration and test all come from the
  same 30-day synthetic window generated in one pass.
* **The CLI says so itself.** Both evaluation modes emit a `disclaimer` field, and the
  cold-start gate emits a `poc_disclaimer`. Show them.

What the numbers **do** legitimately demonstrate: that the pipeline is wired correctly end to
end; that provenance is complete and checkable; that idempotency holds; that leakage controls
are enforced rather than intended; that promotion is gated and human-authorised; and that the
same evidence the demo previews before promotion is exactly what the platform persists after
it. Those are engineering claims, and they are the claims this demo can actually support.

The honest one-liner: **"This shows the control plane works. It does not show the model
works."**

### Prediction and resolved ground truth are different things

The single most common misreading of a fraud dashboard is treating a HIGH-priority alert as a
statement that fraud occurred. It is not. The platform keeps the two concepts in physically
separate tables, with different lifecycles and different write paths.

| | Prediction | Resolved ground truth |
|---|---|---|
| Table | `alert_evidence` | `label_assessments` |
| Key fields | `operational_priority_score`, `priority_band`, `reason_codes`, `degraded` | `resolved_label`, `resolved_label_source`, `maturity_status`, `eligibility_result` |
| Created by | `fraud-intel score` | `fraud-intel labels assess` (**only** — scoring never writes one) |
| Means | "given what was knowable at event time, this is how far up the queue it belongs" | "this is what the outcome turned out to be, and the evidence for it is mature enough to train on" |
| Available | immediately at scoring time | only after the outcome exists and has matured |
| Mutability | immutable, insert-only; a rescore adds a new row | append-only; a changed basis appends a new assessment |
| In this demo | stage J | stage L |

Three consequences worth saying out loud:

1. **A HIGH band is a queue position, not a verdict.** It says "review this first", and in a
   well-tuned queue most HIGH alerts will still turn out to be legitimate. That is not failure;
   that is what a recall-oriented triage layer looks like.
2. **Maturity is a first-class concept, not a detail.** `assess_label()` refuses to mark an
   analyst-derived label eligible until `ANALYST_MATURITY_WINDOW_DAYS` has elapsed *from when
   the evidence arrived*, and it refuses outright when the disposition was `NEEDS_MORE_INFO` or
   `ESCALATED` (`MATURE_BUT_UNRESOLVED`). Training on immature or unresolved labels is how
   teams accidentally train a model to predict "what the last model flagged".
   In this demo synthetic labels mature instantly with the reason code
   `SYNTHETIC_IMMEDIATE_MATURITY` — which is precisely the shortcut that does not exist in
   production.
3. **Evaluation only ever reads resolved, eligible rows.** Stage N evaluates the population for
   which both a current-bundle prediction *and* a mature, resolved label exist, and it refuses
   to report on a partial population rather than quietly evaluating a subset.

---

## 11. Who owns what

This lifecycle deliberately crosses four job families. Naming the boundary at each stage is
usually more valuable to the audience than the metric on the screen.

### Data engineering — *"is the data there, correct, and traceable?"*

**Owns:** stages B, C and the population verification; schema and migrations; the generation
identity contract.

**Responsible for:** deterministic, reproducible ingestion; idempotency (running the same
command twice must not double the data); referential integrity between `channel_events`,
`source_alerts` and `synthetic_event_labels`; one `generation_run_id` and one
`dataset_version` per population; duplicate detection; and keeping ground-truth labels
physically separate from anything a feature or scoring path can read.

**The question they answer:** *"If someone asks in six months which rows this model was trained
on, can you name them exactly?"* Here, yes: `generation_run_id` is a hash of the complete
generation specification.

### Data science — *"does the model learn something real, and is the evaluation honest?"*

**Owns:** stages D, E, F and the metric interpretation in N.

**Responsible for:** feature definitions and the shared feature/history selectors used
identically by training and serving; the chronological split with timestamp-group boundaries
(never a random split on time-ordered fraud data); fitting preprocessing on the training window
only; calibration on a held-out calibration split; keeping the logistic-regression model a
*shadow* that is logged and never scored on; and choosing the metrics that match the decision —
PR-AUC and precision/recall at analyst capacity, not accuracy, on a 3%-prevalence problem.

**The question they answer:** *"Is any of this leakage?"* Stage D exists to answer it in public:
zero row-id overlap, zero equal-timestamp straddle, partitions in chronological order, and the
real training gates enforced before a single model is fitted.

### MLOps — *"can you reproduce, deploy, roll back and audit it?"*

**Owns:** stages E, G, H, I, K, M and O.

**Responsible for:** the model registry and artifact store; bundle versioning
(`CANDIDATE` → `OPERATIONAL` → `RETIRED`, with a database-level unique index allowing at most
one `OPERATIONAL` bundle per channel); verifying that every registered version is `READY` and
points at the exact MLflow run the bundle records; pinning policy versions into the bundle so a
YAML edit cannot silently change a deployed model's behaviour; the lifecycle audit trail in
`pipeline_runs`; idempotent retries; and the promotion gate being a gate — a human-authorised
step with a named `promoted_by`, not the end of a script.

**The question they answer:** *"Who promoted this, when, from which candidate, on what
evidence, and can you put the previous one back?"* Stage I records all of it; the previous
bundle is `RETIRED`, never deleted.

### Risk engineering / fraud strategy — *"what does this cost, and what does it miss?"*

**Owns:** the reading of stages G and N, and the rule set, policies and thresholds the demo
deliberately does not modify.

**Responsible for:** the rule set (`config/fraud_intel/rules_debit_card.yaml`) and its
categories — `MANDATORY_REVIEW` rules force HIGH regardless of model score, which is how policy
overrides a model; the ensemble weights and band thresholds; analyst capacity as the real
operating constraint; the recall target and the loss/friction trade-off behind it; and the
degradation policy (a failed primary model forces HIGH rather than silently scoring low, and a
failed anomaly or graph component forces at least MEDIUM — a missing signal must never look
like a confident low-risk score).

**The question they answer:** *"If my team can review 100 alerts a day, what do we catch, what
do we miss, and how much better is this than the rules we already run?"* That is precisely the
shape of stage N: precision@100, recall@100, minimum alerts for 80% recall, workload reduction,
and a rules-only baseline computed through the *same* scoring mechanism so the comparison is
apples to apples.

### The handoffs, in one line each

* **DE → DS:** a named, immutable, reconciled population with a generation id.
* **DS → MLOps:** a `CANDIDATE` bundle with complete components and its own held-out report.
* **MLOps → Risk:** an `OPERATIONAL`, fully provenanced bundle and a scored, resolved
  population.
* **Risk → DS:** dispositions, which become mature labels, which become the next training set.

That last arrow is the loop. It is also the one that does not exist in this demo — every label
here came from the generator, not from an analyst. Say so.

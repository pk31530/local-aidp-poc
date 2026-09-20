# AiDP v1.2 Control Plane — Step-by-Step Build Guide

## 1. Purpose

AiDP v1.1 already proves the complete local fraud-detection flow: batch data,
streaming, shared features, training, MLflow registration, real-time scoring,
decisioning, persistence and a dashboard.

The v1.2 goal is to add a small control plane inspired by the best operational
ideas in Higgsfield without importing its distributed-GPU complexity.

This version adds exactly three capabilities:

1. **Typed run configuration** — batch, training and streaming runs accept
   validated configuration objects.
2. **Unified CLI** — operators use one `aidp` command instead of remembering
   several Python modules and shell scripts.
3. **Complete run provenance** — every run records what executed, with which
   code, configuration, dataset and model, and how it finished.

This guide is written for Claude Code in VS Code. Build one phase at a time.
Do not paste all implementation prompts together.

---

## 2. Scope boundaries

### Included in v1.2

- Pydantic run-configuration models
- Safe configuration overrides
- Central run lifecycle/provenance service
- PostgreSQL migration for richer run metadata
- Batch, training and streaming integration with the run service
- A standard-library `argparse` CLI
- Human-readable and JSON output
- Unit and integration tests
- Safe GitHub Actions unit-test CI
- Updated architecture, runbook and troubleshooting documentation

### Not included in v1.2

- Multi-node or GPU orchestration
- SSH provisioning
- DeepSpeed, FSDP or LLM loaders
- Remote execution on the user's Mac
- Kubernetes
- Automatic model promotion
- Dataset regeneration unless a test specifically needs an isolated fixture
- Destructive database reset
- Stage checkpoint/resume
- A web-based control-plane UI

These can be considered only after v1.2 is stable.

---

## 3. Target operator experience

The final interface should support commands similar to:

```bash
./scripts/aidp.sh config validate --json
./scripts/aidp.sh pipeline run batch --json
./scripts/aidp.sh train run --features-path data/output/features/features.parquet --json
./scripts/aidp.sh stream run --json
./scripts/aidp.sh run show 42 --json
./scripts/aidp.sh run list --limit 10 --json
./scripts/aidp.sh model show champion --json
```

Human-readable output is the default. `--json` must produce valid JSON on
standard output so the CLI can later be used from automation.

Existing commands must continue to work during v1.2:

```bash
python -m src.processing.pipeline
python -m src.ml.train
./scripts/run_consumer.sh
```

The CLI should call existing Python functions. It must not duplicate pipeline,
training or scoring business logic.

---

## 4. Intended architecture

```text
CLI / existing scripts
        |
        v
Typed run configuration
        |
        v
Central run lifecycle service
        |
        +---- batch pipeline
        +---- model training
        +---- stream consumer
        |
        v
PostgreSQL pipeline_runs + MLflow
```

### Proposed modules

```text
src/control_plane/
  __init__.py
  config.py       # typed BatchRunConfig, TrainingRunConfig, StreamRunConfig
  provenance.py   # metadata capture and sanitisation
  runs.py         # begin, heartbeat, succeed, fail and query run records

src/cli/
  __init__.py
  __main__.py     # python -m src.cli
  output.py       # human and JSON rendering

scripts/
  aidp.sh         # activates .venv when present and invokes src.cli
```

The exact names may change if the repository structure makes another choice
clearly better. Claude must explain any change before implementing it.

---

## 5. Run provenance contract

Each run should record as much of the following as is applicable:

| Field | Meaning |
|---|---|
| `run_id` | Stable database identifier |
| `pipeline_name` | `batch`, `train` or `stream` |
| `status` | `PENDING`, `RUNNING`, `SUCCESS`, `FAILED` or `CANCELLED` |
| `trigger_source` | `cli`, `legacy`, `github_actions`, `api` or `test` |
| `git_sha` | Current commit SHA, or null if unavailable |
| `config_snapshot` | Redacted JSON configuration used by the run |
| `config_hash` | Stable SHA-256 hash of the redacted canonical config |
| `dataset_version` | Input dataset fingerprint when applicable |
| `model_version` | Resulting or serving model version when applicable |
| `records_processed` | Successfully processed count |
| `records_rejected` | Rejected count |
| `artifacts` | JSON object containing non-secret artifact references |
| `error_type` | Exception class for a failed run |
| `error_message` | Sanitised and length-limited failure summary |
| `started_at` | Start time |
| `heartbeat_at` | Most recent progress signal |
| `completed_at` | Terminal completion time |

### Security rules

- Never persist passwords, tokens, connection strings or environment secrets.
- Redact keys containing terms such as `password`, `secret`, `token`, `key`,
  `credential`, `dsn` and `url` where the value could contain credentials.
- Do not store a raw shell command containing user-provided values.
- Limit stored error messages to a reasonable length.
- Preserve full exception details in structured application logs, but store
  only a safe summary in PostgreSQL.

### Lifecycle rules

- A run must be created before its main work begins.
- `SUCCESS` may be recorded only after all required work succeeds.
- An exception must result in `FAILED`, followed by re-raising the original
  exception.
- Terminal states must not be overwritten.
- Finalisation must be safe if accidentally requested twice.
- A failure to update provenance must not hide the original pipeline error.
- Database writes should use parameterised SQL.

---

## 6. Before opening Claude Code

Run these commands in Terminal:

```bash
cd /Users/prabhatkumar/Documents/AiDP/local-aidp-poc
git fetch origin
git switch feat/aidp-v1-1-hardening
git pull --ff-only
git status
```

Expected result:

```text
On branch feat/aidp-v1-1-hardening
nothing to commit, working tree clean
```

If v1.1 has already been merged into `main`, use:

```bash
git switch main
git pull --ff-only
git switch -c feat/aidp-v1-2-control-plane
```

If v1.1 has not yet been merged, create v1.2 directly from the hardened branch:

```bash
git switch -c feat/aidp-v1-2-control-plane
git push -u origin feat/aidp-v1-2-control-plane
```

Confirm:

```bash
git branch --show-current
git status
```

Do not continue if the working tree is not clean. Do not use `git reset --hard`
or delete user files to make it clean.

---

## 7. Claude Code operating method

Recommended model: **Claude Sonnet** for implementation. Opus is optional for
one final architecture review, but it is not required.

For every phase:

1. Start in Plan mode.
2. Paste only that phase's prompt.
3. Review Claude's plan.
4. Select **Tell Claude what to change** if the plan exceeds the stated scope.
5. Once the plan matches the phase, select **Yes, manually approve edits**.
6. Review every proposed edit.
7. Let Claude run only the tests named in the phase.
8. Confirm the diff and commit locally.
9. Do not push until the phase report looks correct.
10. Paste Claude's completion report back into ChatGPT for review before the
    next phase.

Do not use auto mode for schema migrations, run-lifecycle changes or the final
integration phase.

---

## 8. Phase 0 — Baseline assessment and documentation correction

### Goal

Verify the v1.1 branch, identify exact integration points, and correct stale
documentation that still calls schema-version enforcement informational.

### Copy-paste prompt for Claude Code

```text
You are working in the local-aidp-poc repository on branch
feat/aidp-v1-2-control-plane.

This is Phase 0 of the AiDP v1.2 control-plane work. Do not implement the
control plane yet.

First inspect:
- README.md
- ARCHITECTURE.md
- HARDENING_LOG.md
- RUNBOOK.md
- config/settings.yaml
- infrastructure/postgres/lib/schema.sql
- src/processing/pipeline.py
- src/ml/train.py
- src/ingestion/consumer.py
- src/common/config.py
- all current unit tests related to these modules

Tasks:
1. Verify that the working tree is clean and that the current branch is
   feat/aidp-v1-2-control-plane.
2. Run the existing safe unit-test suite and report the exact result.
3. Produce a concise integration map showing where typed run configuration,
   a central run lifecycle service and a unified CLI should connect.
4. Find documentation that incorrectly describes schema_version as
   informational or unenforced. Correct documentation only where it is stale;
   do not change runtime behaviour.
5. Add a short ARCHITECTURE.md section describing the planned v1.2 control
   plane and explicitly marking it as in progress.

Constraints:
- Do not modify application runtime code in this phase.
- Do not modify database schema.
- Do not start, stop or reset Docker infrastructure.
- Do not regenerate data or retrain models.
- Do not modify generated artifacts.
- Do not change Git configuration.
- Do not add Co-authored-by or AI attribution.
- Do not push.

After edits:
- Review the final diff.
- Run any documentation/static checks already present; do not introduce a new
  documentation tool.
- Commit locally with: docs: prepare v1.2 control-plane work

Finish by reporting files changed, tests executed, pass/fail counts, commit SHA
and final git status.
```

### Acceptance checklist

- Existing unit tests pass.
- No Python runtime code changed.
- Stale schema-version wording is corrected.
- The integration map is clear.
- One clean local commit exists.

---

## 9. Phase 1 — Typed run configuration

### Goal

Add validated configuration objects without changing current pipeline results.

### Required design

- Use the repository's installed Pydantic version.
- Create separate types for batch, training and stream runs.
- Defaults must preserve existing behaviour.
- Existing YAML/environment configuration remains the source for platform
  settings; run models describe one invocation and its safe overrides.
- Reject unknown fields to catch spelling mistakes.
- Paths must be represented consistently.
- Provide a redacted, JSON-serialisable snapshot method.
- Provide a stable configuration hash based on canonical JSON.
- Do not include secrets in either output.

### Copy-paste prompt for Claude Code

```text
Implement only Phase 1: typed run configuration for AiDP v1.2.

Read the Phase 0 integration map and inspect existing configuration and CLI
argument handling before editing.

Create a focused control-plane configuration module, preferably
src/control_plane/config.py, containing validated models for:
- BatchRunConfig
- TrainingRunConfig
- StreamRunConfig

Requirements:
1. Preserve all existing defaults and behaviour.
2. Use the installed Pydantic version and forbid unknown fields.
3. Validate numeric bounds such as positive stream rate/duration and positive
   retry/count values where applicable.
4. Keep connection strings, credentials and secrets out of run models unless
   absolutely required. If any sensitive value must be represented, redact it
   in every snapshot.
5. Add a method or shared helper that returns a deterministic,
   JSON-serialisable redacted snapshot.
6. Add a stable SHA-256 config hash computed from canonical JSON.
7. Do not yet refactor batch, training or streaming execution to consume the
   models; that belongs to later phases.
8. Export the intended public types from src/control_plane/__init__.py.

Tests must cover:
- default configurations
- valid overrides
- unknown-field rejection
- invalid numeric bounds
- deterministic snapshots and hashes
- secret redaction or proof that secrets cannot enter the models
- JSON serialisation of Path values or other non-primitive values

Constraints:
- Do not change database schema.
- Do not add a third-party CLI framework.
- Do not touch pipeline execution logic.
- Do not start/reset infrastructure, regenerate data or retrain.
- Do not modify Git configuration.
- Do not add Co-authored-by or AI attribution.
- Do not push.

Run the new tests and the complete safe unit-test suite. Review the final diff.
Commit locally with:
feat(control-plane): add typed run configuration

Finish by reporting exact behaviour, files changed, tests and counts, commit
SHA and final git status.
```

### Acceptance checklist

- All three configuration models exist.
- Invalid input fails before pipeline work starts.
- Snapshots and hashes are stable.
- Secrets cannot leak.
- Existing tests still pass.

---

## 10. Phase 2 — Provenance schema and lifecycle service

### Goal

Create one reliable service for recording all run states and metadata.

### Database approach

Update the fresh-install schema and add an explicit migration, for example:

```text
infrastructure/postgres/migrations/002_pipeline_run_provenance.sql
```

The migration must be safe for an existing v1.1 database. It must not be
automatically applied to the user's live database during this phase.

### Copy-paste prompt for Claude Code

```text
Implement only Phase 2: the provenance database schema and central run
lifecycle service.

Before editing, inspect the existing pipeline_runs schema and all batch/stream
functions that currently insert or update it. Preserve v1.1 failure-recording
correctness.

Tasks:
1. Extend the fresh-install pipeline_runs schema for the agreed v1.2 provenance
   contract: pipeline name including train; status including PENDING, RUNNING,
   SUCCESS, FAILED and CANCELLED; trigger source; git SHA; redacted config JSON;
   config hash; dataset version; model version; artifact JSON; safe error type
   and message; heartbeat timestamp; existing record counts and timestamps.
2. Add a numbered SQL migration for existing installations. It must be
   idempotent where practical and must preserve existing rows.
3. Create a central lifecycle service, preferably src/control_plane/runs.py,
   that can begin, heartbeat, succeed, fail and query a run.
4. Add provenance helpers, preferably src/control_plane/provenance.py, for Git
   SHA detection, safe exception summaries and metadata normalisation.
5. Enforce allowed state transitions. Terminal states must not be overwritten,
   and duplicate finalisation must be safe.
6. Use parameterised SQL and transactions.
7. If provenance finalisation fails while handling another exception, preserve
   and re-raise the original exception.
8. Do not integrate the service into batch, training or streaming yet.

Security requirements:
- Never store DSNs, passwords, API tokens, secret keys or credential-bearing
  URLs.
- Limit stored error length and keep the full traceback only in logs.
- Git SHA lookup must have a timeout and a safe fallback.

Tests must cover:
- successful lifecycle
- failed lifecycle with original exception preserved
- allowed and forbidden state transitions
- duplicate finalisation
- heartbeat behaviour
- redaction
- Git SHA present and unavailable cases
- migration/schema expectations without touching the live aidp database

Constraints:
- Do not apply the migration to aidp or aidp_test yet.
- Do not reset Docker or delete volumes.
- Do not modify runtime pipeline callers in this phase.
- Do not change Git configuration.
- Do not add Co-authored-by or AI attribution.
- Do not push.

Run the new tests and complete safe unit-test suite. Review both SQL files and
the final diff. Commit locally with:
feat(control-plane): add run provenance lifecycle

Finish by reporting files changed, migration behaviour, tests/counts, anything
not executed and why, commit SHA and final git status.
```

### Acceptance checklist

- Fresh schema and migration agree.
- Existing records are preserved.
- Transitions are enforced.
- Secrets and raw tracebacks are not stored.
- Migration was not applied without approval.

---

## 11. Phase 3 — Integrate batch, training and streaming

### Goal

Route all three workloads through the common lifecycle service while retaining
their current direct entry points.

### Copy-paste prompt for Claude Code

```text
Implement only Phase 3: integrate typed configuration and run provenance into
batch, training and streaming execution.

Start by reviewing the Phase 1 and Phase 2 public interfaces and the existing
tests for pipeline and consumer failure status. Do not weaken v1.1 guarantees.

Requirements:
1. Batch, training and streaming entry points accept their typed run config,
   with backward-compatible defaults for existing callers.
2. Every workload creates its run record before main work begins.
3. Each success records status, counts and relevant dataset/model/artifact
   metadata.
4. Each unhandled exception records FAILED and then re-raises the original
   exception.
5. Stream graceful shutdown records SUCCESS; unexpected poll/model/DB failures
   record FAILED.
6. Training records dataset version and resulting MLflow run/model version.
7. Avoid double bookkeeping: remove or route old direct pipeline_runs SQL
   through the lifecycle service.
8. Keep business logic and output results backward compatible.
9. Do not automatically promote a different model or regenerate a dataset.

Testing:
- Update focused unit tests for batch, training and stream lifecycle behaviour.
- Prove existing failure cases still record FAILED.
- Prove success is never recorded after an exception.
- Prove legacy entry points still work with default configs.
- Run the complete unit-test suite.
- If infrastructure is already running and the migration has been explicitly
  approved/applied to the isolated aidp_test database, run only the relevant
  integration tests. Otherwise report the exact commands required and stop;
  do not start or reset infrastructure.

Constraints:
- Do not modify the CLI in this phase.
- Do not apply migrations to the live aidp database without explicit approval.
- Do not reset Docker, delete volumes, regenerate data or retrain the champion.
- Do not change Git configuration.
- Do not add Co-authored-by or AI attribution.
- Do not push.

Review the final diff and commit locally with:
refactor(control-plane): unify workload run tracking

Finish by reporting files changed, lifecycle behaviour for each workload,
tests/counts, anything not executed and why, commit SHA and final git status.
```

### Migration approval checkpoint

Before integration tests, ask Claude to print the exact migration command and
SQL target. Apply first to `aidp_test`, inspect the table, and only then consider
the demo `aidp` database. Never use `reset_demo.sh` as a migration mechanism.

---

## 12. Phase 4 — Unified CLI

### Goal

Add one dependency-light command surface that calls existing application
functions.

### Copy-paste prompt for Claude Code

```text
Implement only Phase 4: a unified AiDP CLI.

Use Python's standard-library argparse unless the repository already contains
an appropriate CLI dependency. Do not add Typer or Click merely for this task.

Create python -m src.cli and a thin scripts/aidp.sh wrapper. Support:
- config validate
- pipeline run batch
- train run with an optional features path
- stream run with supported safe options
- run show RUN_ID
- run list with a bounded limit
- model show champion

Requirements:
1. Commands must invoke existing typed configs and workload functions. Do not
   copy batch, training, streaming, model or SQL business logic into the CLI.
2. Human-readable output is default.
3. --json must emit valid JSON on stdout with no logs or banners mixed into it.
4. Operational logs may go to stderr.
5. Success exits 0; validation/user errors use a consistent non-zero exit;
   unexpected operational failures use a different consistent non-zero exit.
6. Never print secrets or credential-bearing URLs.
7. Help text must show meaningful examples.
8. Existing scripts and module entry points must continue to work.
9. scripts/aidp.sh may activate .venv when present but must work when Python is
   already active.

Tests must cover:
- command help
- valid and invalid configuration
- dispatch without executing real infrastructure
- JSON parseability
- human output
- exit codes
- run-show not-found behaviour
- secret absence

Constraints:
- Do not start/reset infrastructure.
- Do not run a real training job.
- Do not change database schema.
- Do not change Git configuration.
- Do not add Co-authored-by or AI attribution.
- Do not push.

Run focused CLI tests and the complete unit-test suite. Review the final diff.
Commit locally with:
feat(cli): add unified aidp command interface

Finish by reporting supported commands, files changed, tests/counts, sample
output, commit SHA and final git status.
```

### Acceptance checklist

- `python -m src.cli --help` works.
- JSON output parses successfully.
- CLI contains no duplicated business logic.
- Existing entry points remain valid.

---

## 13. Phase 5 — CI and documentation

### Goal

Make every pull request verify the safe test suite and document the new
operator workflow.

### Copy-paste prompt for Claude Code

```text
Implement only Phase 5: safe CI and final v1.2 documentation.

Tasks:
1. Add a GitHub Actions workflow for pull requests and pushes that installs the
   supported Python version and runs the complete unit-test suite.
2. Use dependency caching where straightforward.
3. Do not run destructive scripts, reset databases, publish artifacts, train a
   champion model or require repository secrets.
4. Add a small workflow_dispatch option only if it remains safe and requires no
   external credentials. Prefer unit tests and configuration validation over a
   remote training workflow.
5. Update README.md, ARCHITECTURE.md and RUNBOOK.md with the unified CLI,
   configuration examples, provenance fields, migration instructions and
   backward-compatible legacy commands.
6. Update TROUBLESHOOTING.md with migration mismatch, invalid config, JSON
   output contamination and unavailable Git SHA cases.
7. Mark the v1.2 architecture section complete only after tests pass.

Validation:
- Validate workflow YAML locally using an available safe parser or focused
  test; do not install a global tool.
- Run the complete unit-test suite.
- Run existing safe static checks.
- Review the final diff for secrets, machine-specific paths and stale claims.

Constraints:
- Do not push or open a pull request.
- Do not start/reset infrastructure.
- Do not retrain or promote a model.
- Do not modify Git configuration.
- Do not add Co-authored-by or AI attribution.

Commit locally with:
ci: validate aidp control-plane changes

Finish by reporting workflow triggers, files changed, tests/counts, commit SHA
and final git status.
```

---

## 14. Phase 6 — Final verification

### Goal

Verify the complete feature safely before pushing or opening a pull request.

### Copy-paste prompt for Claude Code

```text
Perform the final read-only-first verification of AiDP v1.2 control-plane work.
Do not make edits until you have reported any findings.

Review all commits on feat/aidp-v1-2-control-plane compared with its base.
Verify:
- typed config defaults preserve v1.1 behaviour
- unknown/invalid input is rejected before work starts
- config snapshots and hashes are deterministic and secret-free
- batch, train and stream use one lifecycle implementation
- errors become FAILED and original exceptions are re-raised
- terminal statuses cannot be overwritten
- CLI JSON output is valid and contains no logs
- legacy scripts still work
- fresh schema and migration agree
- documentation matches implementation
- CI is safe and non-destructive

Run:
- complete unit-test suite
- CLI help and config-validation smoke commands
- JSON-output parse checks
- migration syntax/schema tests that do not modify the live database

If infrastructure is already running and explicit approval was previously
given, run relevant tests against aidp_test only. Do not start/reset services,
apply anything to aidp, regenerate data or retrain without stopping and asking.

Also inspect:
- git diff from the base branch
- git log for the phase commits
- git status
- commit authors and messages
- absence of Co-authored-by and AI attribution
- absence of secrets and machine-specific absolute paths

Report findings by severity with file references. If no blocker exists, state
that the branch is ready to push and open as a pull request. Do not push and do
not open the pull request.
```

### Expected commit sequence

```text
docs: prepare v1.2 control-plane work
feat(control-plane): add typed run configuration
feat(control-plane): add run provenance lifecycle
refactor(control-plane): unify workload run tracking
feat(cli): add unified aidp command interface
ci: validate aidp control-plane changes
```

---

## 15. Push and pull request

Only after Phase 6 reports no blocker:

```bash
git status
git log --oneline --decorate -10
git push -u origin feat/aidp-v1-2-control-plane
```

Suggested pull-request title:

```text
feat: add typed configuration, unified CLI and run provenance
```

Suggested pull-request summary:

```text
## Summary
- adds validated batch, training and streaming run configuration
- adds a unified AiDP command-line interface with JSON output
- records Git, config, dataset, model, artifact and failure provenance
- centralises run lifecycle handling across all workloads
- adds safe unit-test CI and operator documentation

## Safety
- preserves existing v1.1 entry points
- does not introduce remote execution or automatic model promotion
- redacts secrets from snapshots, errors and CLI output
- includes an explicit non-destructive database migration

## Validation
- complete unit-test suite
- CLI parse/output tests
- run lifecycle and transition tests
- migration/schema consistency checks
```

Use GitHub's normal merge process only after CI passes. Do not force-push over
reviewed work.

---

## 16. Stop conditions

Stop Claude and request review if any phase proposes:

- resetting Docker or deleting volumes
- applying a migration to the demo database without approval
- regenerating the main dataset
- retraining or promoting the champion model without approval
- putting credentials into config snapshots or workflow files
- replacing existing business logic inside the CLI
- changing Git identity
- force-pushing
- combining multiple phases into one large commit
- adding GPU, Kubernetes or cloud infrastructure
- silently changing fraud thresholds, model features or training splits

---

## 17. Definition of done

AiDP v1.2 is complete when:

- One validated configuration model exists for each workload.
- One lifecycle service owns run status transitions.
- Every batch, training and stream run has traceable provenance.
- One CLI invokes all supported workloads and queries.
- JSON mode is reliable for future automation.
- Existing scripts remain backward compatible.
- The database migration preserves existing rows.
- No secrets appear in stored metadata, logs, output or GitHub Actions.
- All unit tests and approved integration tests pass.
- Documentation reflects actual behaviour.
- The branch is clean, pushed and reviewed through a pull request.


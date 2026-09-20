# Current Build Checkpoint

Last completed phase:
v1.1 hardening pass (branch `feat/aidp-v1-1-hardening`), on top of
Phase 10 — Final Demo Packaging.

Current phase:
**None — the build is complete, hardened.** All 10 build phases finished
and runtime-verified (see BUILD_LOG.md), plus 5 post-completion
correctness fixes reviewed, tested, committed, and verified end to end
(see HARDENING_LOG.md). README.md/RUNBOOK.md/DEMO_SCRIPT.md/
ARCHITECTURE.md/TROUBLESHOOTING.md still describe how to operate the
platform day to day.

Working (all runtime-verified this session, hardening pass):
- Full infra stack healthy (Postgres, MinIO, Redpanda, Redpanda Console,
  MLflow) — 5/5 containers `Up (healthy)`, none restarted or reset during
  this pass.
- Five hardening fixes committed individually, each with new/updated
  tests: schema_version validation (`1228abe`), flavor-aware champion
  model loading (`b427e0d`), pipeline FAILED-status recording
  (`2079f7f`), idempotent `recent_events` on replay (`9f032c0`), and
  train-split-only risk lookups to eliminate target leakage (`f67c0a1`).
  See HARDENING_LOG.md for the detail on each.
- Batch pipeline and model retrained end to end after the leakage fix:
  `python -m src.processing.pipeline` (50,000 rows, 0 rejected, `split`
  column stratified exactly 70/15/15), `./scripts/train_model.sh`
  (model version 6, registered `champion`). FastAPI restarted to pick up
  v6; `/health` and `/score` re-verified live.
- Full test suite: **79/79 passing** (up from 59 pre-hardening — 20 new
  tests across the five fixes, unit + integration + smoke).
- Documentation complete: README.md, ARCHITECTURE.md, RUNBOOK.md,
  DEMO_SCRIPT.md, TROUBLESHOOTING.md, BUILD_LOG.md (original build
  history), HARDENING_LOG.md (this pass), and this file.
- All guide section 7 required scripts still present and unchanged:
  `bootstrap/start/stop/reset_demo/seed_data/train_model/run_stream/healthcheck.sh`,
  plus the build's own additions (`run_api.sh`, `run_consumer.sh`,
  `run_dashboard.sh`) and a `Makefile`.

Known, documented limitations (not blocking, disclosed honestly rather
than hidden):
- No browser tool was available this session (user declined the Chrome
  extension) — the dashboard's data correctness was thoroughly verified
  via Streamlit's headless `AppTest` framework, but its visual
  layout/styling was never actually seen rendered. The user should open
  http://127.0.0.1:8501 themselves to confirm.
- The API (`:8000`), dashboard (`:8501`), and streaming consumer are
  processes run outside docker-compose (per the guide's own design — see
  guide section 57) — they do not survive a terminal/session reset and
  need to be restarted via their `scripts/run_*.sh` wrappers. This is
  expected/by-design, not a gap.

If resuming a future session: there is no "next phase" — this file's job
is done. Use `git status`-equivalent awareness of BUILD_LOG.md's Phase 10
entry and HARDENING_LOG.md's v1.1 entry as the closing record, and
RUNBOOK.md for "how do I start this back up."

-- Standalone migration — NOT applied automatically by schema.sql /
-- 00_bootstrap.sql (those only run against a fresh, empty data directory).
-- Run this manually, once, against each already-running database that needs
-- the v1.2 control-plane provenance columns on pipeline_runs.
--
-- v1.2 Phase 2: extends pipeline_runs with typed-run provenance (trigger
-- source, Git SHA, redacted config snapshot/hash, dataset/model version,
-- artifacts, safe error type/message, heartbeat) and widens pipeline_name to
-- include 'train' and status to include 'PENDING'/'CANCELLED'. Preserves all
-- existing rows: every current pipeline_name ('batch'/'stream') and status
-- ('RUNNING'/'SUCCESS'/'FAILED') value is a strict subset of the new allowed
-- sets, so no row is invalidated when the CHECK constraints are replaced.
--
-- Idempotent: every ADD COLUMN uses IF NOT EXISTS, both CHECK constraints
-- are replaced by explicit, stable name (DROP CONSTRAINT IF EXISTS + ADD
-- CONSTRAINT — not the unsupported "ADD CONSTRAINT IF NOT EXISTS" form), and
-- indexes use IF NOT EXISTS. Re-running this file is safe. The whole file is
-- one transaction: if any statement fails, every prior statement in this
-- run (including a DROP CONSTRAINT) is rolled back, so the table is never
-- left without its constraints.
--
-- Usage (adjust -d for the target database, e.g. aidp or aidp_test):
--   docker exec -i aidp-postgres psql -U aidp -d aidp \
--     -f - < infrastructure/postgres/migrations/002_pipeline_run_provenance.sql

BEGIN;

ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS trigger_source TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS git_sha TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS config_snapshot JSONB;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS config_hash TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS dataset_version TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS artifacts JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS error_type TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ;

-- Widen pipeline_name to add 'train'.
ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_pipeline_name_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_pipeline_name_check
    CHECK (pipeline_name IN ('batch', 'train', 'stream'));

-- Widen status to add 'PENDING' and 'CANCELLED'.
ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_status_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_status_check
    CHECK (status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED'));

-- New: cap stored error_message length. Explicit, stable constraint name so
-- rerunning this file doesn't fail on "constraint already exists".
ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_error_message_length_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_error_message_length_check
    CHECK (error_message IS NULL OR char_length(error_message) <= 2000);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pipeline_name ON pipeline_runs (pipeline_name);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs (status);

COMMIT;

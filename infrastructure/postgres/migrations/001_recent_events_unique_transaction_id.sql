-- Standalone migration — NOT applied automatically by schema.sql /
-- 00_bootstrap.sql (those only run against a fresh, empty data directory).
-- Run this manually, once, against each already-running database that
-- needs recent_events to be idempotent per transaction_id.
--
-- fix: recent_events could receive duplicate rows when the same
-- transaction was replayed (no unique constraint on transaction_id),
-- which inflated the live velocity features read by fetch_recent_events.
--
-- Usage (adjust -d for the target database, e.g. aidp or aidp_test):
--   docker exec -i aidp-postgres psql -U aidp -d aidp \
--     -f - < infrastructure/postgres/migrations/001_recent_events_unique_transaction_id.sql

BEGIN;

-- Step 1: de-duplicate existing rows. Keep only the lowest event_id per
-- non-null transaction_id (the first-ever recorded event for that
-- transaction); delete every later duplicate a replayed/redelivered
-- message produced.
DELETE FROM recent_events r
WHERE r.transaction_id IS NOT NULL
  AND r.event_id > (
    SELECT min(r2.event_id)
    FROM recent_events r2
    WHERE r2.transaction_id = r.transaction_id
  );

-- Step 2: enforce uniqueness going forward. Partial index keeps
-- transaction_id nullable for any future non-transaction-linked event
-- while enforcing uniqueness for current usage (score_and_persist always
-- passes a non-null transaction_id). Matches the definition added to
-- infrastructure/postgres/lib/schema.sql for fresh installs.
CREATE UNIQUE INDEX IF NOT EXISTS uq_recent_events_transaction_id
    ON recent_events (transaction_id) WHERE transaction_id IS NOT NULL;

COMMIT;

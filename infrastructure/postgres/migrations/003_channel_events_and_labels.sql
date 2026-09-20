-- Standalone migration -- NOT applied automatically by schema.sql /
-- 00_bootstrap.sql (those only run against a fresh, empty data directory).
-- Run this manually, once, against each already-running database that
-- needs the v1.3 fraud-intelligence channel-event/source-alert/label/
-- bundle tables.
--
-- v1.3 Phase 1: adds four brand-new tables -- channel_events,
-- source_alerts, synthetic_event_labels, channel_model_bundles -- plus the
-- partial unique index enforcing at most one OPERATIONAL bundle per
-- channel (guide section 22). Unlike migration 002, none of these tables
-- already exist on any prior install, so this migration is a
-- straightforward idempotent CREATE, not an ALTER of existing constraints
-- -- there is nothing to DROP CONSTRAINT / ADD CONSTRAINT here. This SQL
-- is intentionally IDENTICAL to the new-table section of
-- infrastructure/postgres/lib/schema.sql; the two must stay in agreement.
--
-- Idempotent: every CREATE TABLE uses IF NOT EXISTS, every index uses IF
-- NOT EXISTS. Re-running this file is safe. The whole file is one
-- transaction: if any statement fails, every prior statement in this run
-- is rolled back, so no table is left half-created.
--
-- Not applied to any database by this guide's Phase 1 -- reviewed only.
-- When approved (optional at Phase 1, required before Phase 7B), apply to
-- aidp_test ONLY:
--   docker exec -i aidp-postgres psql -v ON_ERROR_STOP=1 -U aidp -d aidp_test \
--     -f - < infrastructure/postgres/migrations/003_channel_events_and_labels.sql

BEGIN;

CREATE TABLE IF NOT EXISTS channel_events (
    event_id            UUID PRIMARY KEY,
    channel              TEXT NOT NULL CHECK (channel IN ('ach', 'wire', 'mobile_deposit', 'online_banking', 'atm', 'debit_card', 'p2p')),
    customer_id           TEXT NOT NULL,
    account_id            TEXT NOT NULL,
    event_timestamp       TIMESTAMPTZ NOT NULL,
    amount_minor_units    BIGINT NOT NULL CHECK (amount_minor_units > 0),
    direction             TEXT NOT NULL CHECK (direction IN ('debit', 'credit')),
    device_id             TEXT,
    ip_address            TEXT,
    channel_payload       JSONB NOT NULL,
    scenario_id           TEXT,
    schema_version        INT NOT NULL DEFAULT 1,
    generation_run_id     TEXT,
    dataset_version       TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_channel_events_customer ON channel_events (customer_id);
CREATE INDEX IF NOT EXISTS idx_channel_events_timestamp ON channel_events (event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_channel_events_generation_run ON channel_events (generation_run_id);

CREATE TABLE IF NOT EXISTS source_alerts (
    source_alert_id           UUID PRIMARY KEY,
    source_system              TEXT NOT NULL,
    event_id                   UUID NOT NULL REFERENCES channel_events (event_id),
    source_alert_created_at    TIMESTAMPTZ NOT NULL,
    source_rule_ids            JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_rule_version        TEXT NOT NULL,
    source_alert_score         NUMERIC(7, 6),
    source_alert_reason_codes  JSONB NOT NULL DEFAULT '[]'::jsonb,
    generation_run_id          TEXT,
    dataset_version             TEXT,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_source_alerts_event_id ON source_alerts (event_id);
CREATE INDEX IF NOT EXISTS idx_source_alerts_generation_run ON source_alerts (generation_run_id);

CREATE TABLE IF NOT EXISTS synthetic_event_labels (
    event_id                  UUID PRIMARY KEY REFERENCES channel_events (event_id),
    scenario_id                TEXT,
    synthetic_scenario_label   BOOLEAN NOT NULL,
    scenario_type               TEXT NOT NULL,
    label_source                 TEXT NOT NULL DEFAULT 'SYNTHETIC_GENERATOR' CHECK (label_source = 'SYNTHETIC_GENERATOR'),
    generation_run_id            TEXT,
    dataset_version               TEXT,
    generated_at                   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS channel_model_bundles (
    bundle_id                       BIGSERIAL PRIMARY KEY,
    channel                          TEXT NOT NULL CHECK (channel IN ('ach', 'wire', 'mobile_deposit', 'online_banking', 'atm', 'debit_card', 'p2p')),
    bundle_version                   INT NOT NULL,
    gbm_model_version                TEXT,
    lr_model_version                 TEXT,
    anomaly_model_version            TEXT,
    preprocessing_artifact_version   TEXT,
    feature_schema_version           TEXT,
    training_run_id                  BIGINT,
    dataset_version                  TEXT,
    evaluation_report_ref            TEXT,
    status                           TEXT NOT NULL DEFAULT 'CANDIDATE' CHECK (status IN ('CANDIDATE', 'OPERATIONAL', 'RETIRED')),
    created_at                       TIMESTAMPTZ NOT NULL DEFAULT now(),
    promoted_at                      TIMESTAMPTZ,
    promoted_by                      TEXT,
    UNIQUE (channel, bundle_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_one_operational_bundle_per_channel
    ON channel_model_bundles (channel) WHERE status = 'OPERATIONAL';

COMMIT;

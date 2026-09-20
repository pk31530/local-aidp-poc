-- Standalone migration -- NOT applied automatically by schema.sql /
-- 00_bootstrap.sql (those only run against a fresh, empty data directory).
-- Run this manually, once, against each already-running database that
-- needs the v1.3 fraud-intelligence alert queue, evidence, disposition,
-- and label-assessment tables, PLUS the pipeline_runs.pipeline_name widen.
--
-- v1.3 Phase 6. Depends on migration 003 already being applied
-- (channel_events/source_alerts/channel_model_bundles must already exist)
-- -- apply 003 before 004, in that order, never the reverse.
--
-- Idempotent: every CREATE TABLE/INDEX uses IF NOT EXISTS, every ALTER
-- TABLE ADD COLUMN uses IF NOT EXISTS, the pipeline_name CHECK widen uses
-- the same DROP CONSTRAINT IF EXISTS / ADD CONSTRAINT pattern as migration
-- 002 (never the unsupported "ADD CONSTRAINT IF NOT EXISTS"). The one
-- exception is source_alerts' new UNIQUE(source_system, source_alert_id)
-- constraint: DROP+ADD is unsafe for it specifically, because fraud_alerts
-- (created later in this same file) composite-FKs against it -- a blind
-- DROP on a re-run would fail once that dependent FK exists. That
-- constraint instead uses an existence-checked DO block, scoped to this
-- exact table (conrelid = 'source_alerts'::regclass), not just the
-- constraint name, so a same-named constraint on an unrelated table can
-- never cause a false match.
--
-- The whole file is one transaction: if any statement fails, every prior
-- statement in this run is rolled back, so no table/column/constraint is
-- ever left half-applied.
--
-- Not applied to any database by this guide's Phase 6 -- reviewed only.
-- When separately approved, apply to aidp_test ONLY:
--   docker exec -i aidp-postgres psql -v ON_ERROR_STOP=1 -U aidp -d aidp_test \
--     -f - < infrastructure/postgres/migrations/004_fraud_alerts_evidence_and_lifecycle.sql

BEGIN;

-- ---- source_alerts: add the composite unique constraint fraud_alerts needs ----

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'source_alerts'::regclass
          AND conname = 'uq_source_alerts_system_id'
    ) THEN
        ALTER TABLE source_alerts ADD CONSTRAINT uq_source_alerts_system_id UNIQUE (source_system, source_alert_id);
    END IF;
END $$;

-- ---- channel_model_bundles: four new columns (Phase 5 decision 3) ----

ALTER TABLE channel_model_bundles ADD COLUMN IF NOT EXISTS rule_set_version TEXT;
ALTER TABLE channel_model_bundles ADD COLUMN IF NOT EXISTS graph_policy_version TEXT;
ALTER TABLE channel_model_bundles ADD COLUMN IF NOT EXISTS ensemble_policy_version TEXT;
ALTER TABLE channel_model_bundles ADD COLUMN IF NOT EXISTS reason_code_version TEXT;

-- ---- fraud_alerts ----

CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id                             BIGSERIAL PRIMARY KEY,
    event_id                             UUID NOT NULL REFERENCES channel_events (event_id),
    source_alert_id                      UUID NOT NULL,
    source_system                        TEXT NOT NULL,
    channel                              TEXT NOT NULL CHECK (channel IN ('ach', 'wire', 'mobile_deposit', 'online_banking', 'atm', 'debit_card', 'p2p')),
    customer_id                          TEXT NOT NULL,
    account_id                           TEXT NOT NULL,
    amount_minor_units                   BIGINT NOT NULL CHECK (amount_minor_units > 0),
    initial_operational_priority_score   NUMERIC(7, 6) NOT NULL CHECK (initial_operational_priority_score BETWEEN 0 AND 1),
    initial_priority_band                TEXT NOT NULL CHECK (initial_priority_band IN ('LOW', 'MEDIUM', 'HIGH')),
    initial_ensemble_policy_version      TEXT NOT NULL,
    status                               TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'IN_REVIEW', 'CLOSED')),
    created_at                           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_system, source_alert_id),
    FOREIGN KEY (source_system, source_alert_id) REFERENCES source_alerts (source_system, source_alert_id)
);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_event_id ON fraud_alerts (event_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts (status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_priority_band ON fraud_alerts (initial_priority_band);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_created_at ON fraud_alerts (created_at DESC);

-- ---- alert_evidence ----

CREATE TABLE IF NOT EXISTS alert_evidence (
    evidence_id                      UUID PRIMARY KEY,
    alert_id                         BIGINT NOT NULL REFERENCES fraud_alerts (alert_id),
    score_execution_id               UUID NOT NULL,
    rule_result                      JSONB,
    gbm_probability                  NUMERIC(7, 6),
    lr_probability                   NUMERIC(7, 6),
    anomaly_score                    NUMERIC(7, 6),
    graph_risk_score                 NUMERIC(7, 6),
    operational_priority_score       NUMERIC(7, 6) CHECK (operational_priority_score IS NULL OR operational_priority_score BETWEEN 0 AND 1),
    priority_band                    TEXT CHECK (priority_band IS NULL OR priority_band IN ('LOW', 'MEDIUM', 'HIGH')),
    degraded                         BOOLEAN NOT NULL DEFAULT false,
    component_statuses               JSONB NOT NULL CHECK (jsonb_typeof(component_statuses) = 'object'),
    reason_codes                     JSONB NOT NULL CHECK (jsonb_typeof(reason_codes) = 'array'),
    channel_model_bundle_id           BIGINT NOT NULL REFERENCES channel_model_bundles (bundle_id),
    gbm_model_version                 TEXT NOT NULL,
    lr_model_version                   TEXT NOT NULL,
    anomaly_model_version               TEXT NOT NULL,
    preprocessing_artifact_version        TEXT NOT NULL,
    feature_schema_version                TEXT NOT NULL,
    rule_set_version                      TEXT,
    graph_policy_version                  TEXT NOT NULL,
    ensemble_policy_version               TEXT NOT NULL,
    reason_code_version                   TEXT NOT NULL,
    config_hash                           TEXT NOT NULL,
    git_sha                               TEXT,
    event_time                            TIMESTAMPTZ NOT NULL,
    scored_at                             TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (alert_id, score_execution_id)
);
CREATE INDEX IF NOT EXISTS idx_alert_evidence_alert_latest ON alert_evidence (alert_id, scored_at DESC, evidence_id DESC);

-- ---- analyst_dispositions ----

CREATE TABLE IF NOT EXISTS analyst_dispositions (
    disposition_id     BIGSERIAL PRIMARY KEY,
    alert_id            BIGINT NOT NULL REFERENCES fraud_alerts (alert_id),
    analyst_id           TEXT NOT NULL,
    disposition            TEXT NOT NULL CHECK (disposition IN ('CONFIRMED_FRAUD', 'CONFIRMED_LEGITIMATE', 'NEEDS_MORE_INFO', 'ESCALATED')),
    notes                    TEXT CHECK (notes IS NULL OR char_length(notes) <= 2000),
    disposed_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_analyst_dispositions_alert_id ON analyst_dispositions (alert_id);

-- ---- label_assessments ----

CREATE TABLE IF NOT EXISTS label_assessments (
    assessment_id           BIGSERIAL PRIMARY KEY,
    alert_id                 BIGINT NOT NULL REFERENCES fraud_alerts (alert_id),
    policy_version              TEXT NOT NULL,
    evaluated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_disposition_id           BIGINT REFERENCES analyst_dispositions (disposition_id),
    basis_timestamp                   TIMESTAMPTZ NOT NULL,
    maturity_due_at                     TIMESTAMPTZ NOT NULL,
    maturity_status                       TEXT NOT NULL CHECK (maturity_status IN ('IMMATURE', 'MATURE')),
    eligibility_result                      BOOLEAN NOT NULL,
    eligibility_reason_code                   TEXT NOT NULL,
    resolved_label                              TEXT CHECK (resolved_label IS NULL OR resolved_label IN ('RESOLVED_FRAUD', 'RESOLVED_LEGITIMATE', 'UNRESOLVED')),
    resolved_label_source                         TEXT CHECK (resolved_label_source IS NULL OR resolved_label_source IN ('SYNTHETIC_GENERATOR', 'ANALYST_DISPOSITION', 'EXTERNAL_CONFIRMATION'))
);
CREATE INDEX IF NOT EXISTS idx_label_assessments_alert_latest ON label_assessments (alert_id, evaluated_at DESC, assessment_id DESC);

-- ---- pipeline_runs.pipeline_name widen (same pattern as migration 002) ----

ALTER TABLE pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_pipeline_name_check;
ALTER TABLE pipeline_runs ADD CONSTRAINT pipeline_runs_pipeline_name_check
    CHECK (pipeline_name IN ('batch', 'train', 'stream', 'fraud_score', 'label_eligibility', 'model_promotion', 'fraud_evaluation'));

COMMIT;

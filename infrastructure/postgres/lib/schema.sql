-- Shared schema, applied to both the demo database (aidp) and the isolated
-- test database (aidp_test, fix H4) by 00_bootstrap.sql.

CREATE TABLE IF NOT EXISTS customers (
    customer_id     TEXT PRIMARY KEY,
    full_name       TEXT,
    home_country    TEXT,
    signup_date     DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Online feature/profile store (fix C1). Queried live at scoring time by both
-- the batch pipeline and the real-time serving path via src/common/features.py.
CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id                TEXT PRIMARY KEY,
    avg_transaction_amount     NUMERIC(14, 2) NOT NULL DEFAULT 0,
    stddev_transaction_amount  NUMERIC(14, 2) NOT NULL DEFAULT 0,
    known_devices               JSONB NOT NULL DEFAULT '[]'::jsonb,
    known_countries             JSONB NOT NULL DEFAULT '[]'::jsonb,
    home_country                TEXT,
    total_transactions          INT NOT NULL DEFAULT 0,
    last_transaction_at         TIMESTAMPTZ,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Velocity/recency feature source (fix C1): transaction_count_1h,
-- failed_attempts_1h, transactions_last_10m are computed from this table at
-- scoring time, not stubbed or precomputed.
CREATE TABLE IF NOT EXISTS recent_events (
    event_id        BIGSERIAL PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    transaction_id  TEXT,
    event_type      TEXT NOT NULL CHECK (event_type IN ('transaction_success', 'transaction_failed')),
    amount          NUMERIC(14, 2),
    country         TEXT,
    device_id       TEXT,
    occurred_at     TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_recent_events_customer_time
    ON recent_events (customer_id, occurred_at DESC);

-- Idempotency for the one write path (score_and_persist -> record_event),
-- which always passes the scored transaction's own transaction_id. A
-- partial index keeps transaction_id nullable for any future
-- non-transaction-linked event while enforcing uniqueness for current
-- usage, so a replayed message can't insert a duplicate velocity-feature
-- row (fix: recent_events not idempotent on replay).
CREATE UNIQUE INDEX IF NOT EXISTS uq_recent_events_transaction_id
    ON recent_events (transaction_id) WHERE transaction_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id          TEXT PRIMARY KEY,
    customer_id              TEXT NOT NULL,
    amount                   NUMERIC(14, 2) NOT NULL CHECK (amount > 0),
    merchant                 TEXT,
    country                  TEXT,
    device_id                TEXT,
    payment_method           TEXT,
    transaction_timestamp    TIMESTAMPTZ NOT NULL,
    source                   TEXT NOT NULL CHECK (source IN ('batch', 'stream', 'api')),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions (customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions (transaction_timestamp DESC);

CREATE TABLE IF NOT EXISTS fraud_scores (
    transaction_id     TEXT PRIMARY KEY REFERENCES transactions (transaction_id),
    fraud_probability  NUMERIC(7, 6) NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    model_version      TEXT NOT NULL,
    scored_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fraud_decisions (
    transaction_id     TEXT PRIMARY KEY REFERENCES transactions (transaction_id),
    customer_id        TEXT NOT NULL,
    fraud_probability  NUMERIC(7, 6) NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    risk_level         TEXT NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    decision           TEXT NOT NULL CHECK (decision IN ('APPROVE', 'MONITOR', 'REVIEW', 'BLOCK')),
    reason_codes       JSONB NOT NULL DEFAULT '[]'::jsonb,
    model_version      TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fraud_decisions_created ON fraud_decisions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fraud_decisions_risk ON fraud_decisions (risk_level);

CREATE TABLE IF NOT EXISTS model_versions (
    id                BIGSERIAL PRIMARY KEY,
    model_version     TEXT NOT NULL,
    model_name        TEXT NOT NULL,
    mlflow_run_id     TEXT,
    precision_score   NUMERIC(7, 6),
    recall_score      NUMERIC(7, 6),
    f1_score          NUMERIC(7, 6),
    roc_auc_score     NUMERIC(7, 6),
    registered_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active         BOOLEAN NOT NULL DEFAULT false
);

-- v1.2 control-plane provenance (see
-- infrastructure/postgres/migrations/002_pipeline_run_provenance.sql for the
-- existing-install migration; this CREATE TABLE only ever runs against a
-- fresh, empty data directory, so the two must be kept in agreement).
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              BIGSERIAL PRIMARY KEY,
    pipeline_name       TEXT NOT NULL CHECK (pipeline_name IN ('batch', 'train', 'stream')),
    status              TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED', 'CANCELLED')),
    trigger_source      TEXT CHECK (trigger_source IS NULL OR trigger_source IN ('cli', 'legacy', 'github_actions', 'api', 'test')),
    git_sha             TEXT,
    config_snapshot     JSONB,
    config_hash         TEXT,
    dataset_version     TEXT,
    model_version       TEXT,
    records_processed   INT NOT NULL DEFAULT 0,
    records_rejected    INT NOT NULL DEFAULT 0,
    artifacts           JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_type          TEXT,
    error_message       TEXT CHECK (error_message IS NULL OR char_length(error_message) <= 2000),
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    heartbeat_at        TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pipeline_name ON pipeline_runs (pipeline_name);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs (status);

-- ============================================================
-- AiDP v1.3 fraud-intelligence: channel events, source alerts, synthetic
-- labels, and channel model bundles (v1.3 Phase 1). See
-- infrastructure/postgres/migrations/003_channel_events_and_labels.sql for
-- the matching existing-install migration; the two must stay in agreement,
-- same discipline as pipeline_runs above.
-- ============================================================

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

-- One row per source alert (guide section 6). event_id is intentionally
-- NOT unique here -- one event can have more than one source alert.
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

-- Ground truth from the synthetic generator only (guide section 6/8) --
-- one row per event, never read by any scoring/feature code path.
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

-- Channel model bundle registry (guide section 22). Not populated until
-- Phase 4 -- its shape is defined now so later phases don't need a schema
-- change. status widens to RETIRED alongside CANDIDATE/OPERATIONAL so a
-- promoted bundle's predecessor can be demoted rather than deleted.
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
-- Guide section 22: at most one OPERATIONAL bundle per channel, enforced
-- at the database level, independent of application logic.
CREATE UNIQUE INDEX IF NOT EXISTS uq_one_operational_bundle_per_channel
    ON channel_model_bundles (channel) WHERE status = 'OPERATIONAL';

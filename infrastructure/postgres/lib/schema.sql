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
    -- v1.3 Phase 6 additive widen: fraud_score/label_eligibility/
    -- model_promotion/fraud_evaluation, alongside the original three.
    pipeline_name       TEXT NOT NULL CHECK (pipeline_name IN ('batch', 'train', 'stream', 'fraud_score', 'label_eligibility', 'model_promotion', 'fraud_evaluation')),
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
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- v1.3 Phase 6: fraud_alerts composite-FKs against (source_system,
    -- source_alert_id) -- source_alert_id alone being a PK does not
    -- satisfy Postgres's requirement that a composite FK's referenced
    -- columns have a unique constraint on that EXACT column pair.
    UNIQUE (source_system, source_alert_id)
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
    -- Phase 5 decision 3 -- required for promotion eligibility (see
    -- src.fraud_intel.models.bundle.REQUIRED_OPERATIONAL_COMPONENTS) but
    -- nullable here so Phase 4's already-registered, immutable candidate
    -- bundle (which predates these columns) stays valid unchanged.
    rule_set_version                 TEXT,
    graph_policy_version              TEXT,
    ensemble_policy_version            TEXT,
    reason_code_version                 TEXT,
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

-- ============================================================
-- AiDP v1.3 fraud-intelligence: alert queue, evidence, analyst
-- dispositions, label assessments (v1.3 Phase 6). See
-- infrastructure/postgres/migrations/004_fraud_alerts_evidence_and_lifecycle.sql
-- for the matching existing-install migration; the two must stay in
-- agreement, same discipline as pipeline_runs/channel_events above.
-- ============================================================

-- One row per source alert (guide section 18), never per event -- the
-- same event_id can appear on more than one row (guide section 6).
-- initial_operational_priority_score/initial_priority_band/
-- initial_ensemble_policy_version are set ONCE, at first-scoring time,
-- and never updated by a later rescore (Phase 6 decision 2) -- the
-- current/latest scoring result always comes from the latest
-- alert_evidence row instead, never from these columns.
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

-- Insert-only, immutable. One row per logical scoring execution
-- (score_execution_id) -- a same-execution retry is a safe no-op, a
-- deliberate re-score always adds a new row (guide section 18/20).
-- Component/policy/bundle-provenance columns are NOT NULL wherever they
-- are derivable from the scoring CALL's own arguments (bundle/policy
-- objects), which are known even when scoring itself fails catastrophically
-- (Phase 6 decision 3) -- only genuine scoring OUTPUTS (rule_result,
-- operational_priority_score, priority_band, rule_set_version) are
-- nullable, since those alone are unrecoverable from a catastrophic
-- failure.
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

-- Append-only analyst action log -- never updated or deleted. Multiple
-- dispositions per alert over time are legitimate (guide section 18).
CREATE TABLE IF NOT EXISTS analyst_dispositions (
    disposition_id     BIGSERIAL PRIMARY KEY,
    alert_id            BIGINT NOT NULL REFERENCES fraud_alerts (alert_id),
    analyst_id           TEXT NOT NULL,
    disposition            TEXT NOT NULL CHECK (disposition IN ('CONFIRMED_FRAUD', 'CONFIRMED_LEGITIMATE', 'NEEDS_MORE_INFO', 'ESCALATED')),
    notes                    TEXT CHECK (notes IS NULL OR char_length(notes) <= 2000),
    disposed_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_analyst_dispositions_alert_id ON analyst_dispositions (alert_id);

-- Append-only -- replaces any single mutable "current training label" row
-- (guide section 18/19). source_disposition_id traces an analyst-derived
-- assessment back to the exact disposition it is based on (NULL for a
-- synthetic-only assessment, per Phase 6 decision 5's application-level
-- rule, enforced in src.fraud_intel.labels.eligibility).
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

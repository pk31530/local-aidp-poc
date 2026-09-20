"""Phase 6: static text-only consistency checks between fresh-install
schema.sql and migration 004 -- no SQL is ever executed, no database is
touched. Migration 004 itself is applied to no database in Phase 6.
"""
from __future__ import annotations

import re

from src.common.config import PROJECT_ROOT

SCHEMA_SQL = (PROJECT_ROOT / "infrastructure" / "postgres" / "lib" / "schema.sql").read_text()
MIGRATION_SQL = (
    PROJECT_ROOT / "infrastructure" / "postgres" / "migrations" / "004_fraud_alerts_evidence_and_lifecycle.sql"
).read_text()

EXPECTED_NEW_TABLES = {"fraud_alerts", "alert_evidence", "analyst_dispositions", "label_assessments"}


def _table_names(sql_text: str) -> set[str]:
    return set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", sql_text))


def _strip_sql_comments(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))


MIGRATION_EXECUTABLE_SQL = _strip_sql_comments(MIGRATION_SQL)


def test_migration_004_file_exists_and_is_one_transaction():
    assert "BEGIN;" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("COMMIT;")


def test_migration_never_uses_unsupported_add_constraint_if_not_exists():
    assert "ADD CONSTRAINT IF NOT EXISTS" not in MIGRATION_EXECUTABLE_SQL


def test_schema_sql_and_migration_004_declare_the_same_new_tables():
    schema_tables = _table_names(SCHEMA_SQL)
    migration_tables = _table_names(MIGRATION_SQL)
    assert EXPECTED_NEW_TABLES <= schema_tables
    assert EXPECTED_NEW_TABLES <= migration_tables


def test_source_alerts_composite_unique_constraint_present_in_both_files():
    assert "UNIQUE (source_system, source_alert_id)" in SCHEMA_SQL
    assert "uq_source_alerts_system_id" in MIGRATION_SQL
    assert "conrelid = 'source_alerts'::regclass" in MIGRATION_SQL
    assert "ALTER TABLE source_alerts ADD CONSTRAINT uq_source_alerts_system_id UNIQUE (source_system, source_alert_id)" in MIGRATION_SQL


def test_fraud_alerts_has_composite_fk_to_source_alerts_in_both_files():
    expected = "FOREIGN KEY (source_system, source_alert_id) REFERENCES source_alerts (source_system, source_alert_id)"
    assert expected in SCHEMA_SQL
    assert expected in MIGRATION_SQL


def test_fraud_alerts_uses_amount_minor_units_not_dollar_amount():
    for text in (SCHEMA_SQL, MIGRATION_SQL):
        assert "amount_minor_units" in text
        assert "dollar_amount" not in text


def test_fraud_alerts_event_id_has_no_standalone_unique_constraint():
    for text, path_label in ((SCHEMA_SQL, "schema.sql"), (MIGRATION_SQL, "migration 004")):
        match = re.search(r"CREATE TABLE IF NOT EXISTS fraud_alerts \((.*?)\n\);", text, re.DOTALL)
        assert match is not None, f"fraud_alerts table not found in {path_label}"
        body = match.group(1)
        for line in body.splitlines():
            if line.strip().startswith("event_id"):
                assert "UNIQUE" not in line
                assert "PRIMARY KEY" not in line


def test_alert_evidence_jsonb_shape_checks_present_in_both_files():
    for text in (SCHEMA_SQL, MIGRATION_SQL):
        assert "jsonb_typeof(component_statuses) = 'object'" in text
        assert "jsonb_typeof(reason_codes) = 'array'" in text


def test_alert_evidence_has_channel_model_bundle_id_fk_in_both_files():
    for text in (SCHEMA_SQL, MIGRATION_SQL):
        assert "channel_model_bundle_id" in text
        assert "REFERENCES channel_model_bundles (bundle_id)" in text


def test_alert_evidence_unique_on_alert_and_score_execution_id():
    for text in (SCHEMA_SQL, MIGRATION_SQL):
        assert "UNIQUE (alert_id, score_execution_id)" in text


def test_alert_evidence_latest_index_has_deterministic_tie_break():
    expected = "ON alert_evidence (alert_id, scored_at DESC, evidence_id DESC)"
    assert expected in SCHEMA_SQL
    assert expected in MIGRATION_SQL


def test_label_assessments_latest_index_has_deterministic_tie_break():
    expected = "ON label_assessments (alert_id, evaluated_at DESC, assessment_id DESC)"
    assert expected in SCHEMA_SQL
    assert expected in MIGRATION_SQL


def test_label_assessments_has_source_disposition_id_and_basis_columns():
    for text in (SCHEMA_SQL, MIGRATION_SQL):
        assert "source_disposition_id" in text
        assert "basis_timestamp" in text
        assert "maturity_due_at" in text
        assert "eligibility_reason_code" in text


def test_analyst_dispositions_check_values_match_between_files():
    schema_values = set(re.findall(r"'(\w+)'", re.search(r"CHECK \(disposition IN \(([^)]*)\)\)", SCHEMA_SQL).group(1)))
    migration_values = set(re.findall(r"'(\w+)'", re.search(r"CHECK \(disposition IN \(([^)]*)\)\)", MIGRATION_SQL).group(1)))
    assert schema_values == migration_values == {"CONFIRMED_FRAUD", "CONFIRMED_LEGITIMATE", "NEEDS_MORE_INFO", "ESCALATED"}


def test_channel_model_bundles_new_version_columns_present_in_both_files():
    for text in (SCHEMA_SQL, MIGRATION_SQL):
        assert "rule_set_version" in text
        assert "graph_policy_version" in text
        assert "ensemble_policy_version" in text
        assert "reason_code_version" in text


def test_channel_model_bundles_partial_unique_index_untouched_by_migration_004():
    assert "uq_one_operational_bundle_per_channel" not in MIGRATION_SQL


def test_pipeline_name_values_consistent_between_schema_and_migration_004():
    schema_match = re.search(r"CHECK \(pipeline_name IN \(([^)]*)\)\)", SCHEMA_SQL)
    migration_match = re.search(r"CHECK \(pipeline_name IN \(([^)]*)\)\)", MIGRATION_SQL)
    assert schema_match and migration_match
    schema_values = set(re.findall(r"'(\w+)'", schema_match.group(1)))
    migration_values = set(re.findall(r"'(\w+)'", migration_match.group(1)))
    expected = {"batch", "train", "stream", "fraud_score", "label_eligibility", "model_promotion", "fraud_evaluation"}
    assert schema_values == migration_values == expected


def test_migration_004_drops_before_adding_the_pipeline_name_constraint():
    assert "DROP CONSTRAINT IF EXISTS pipeline_runs_pipeline_name_check" in MIGRATION_SQL
    assert "ADD CONSTRAINT pipeline_runs_pipeline_name_check" in MIGRATION_SQL

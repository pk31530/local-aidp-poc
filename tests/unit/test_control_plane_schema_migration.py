"""Static consistency checks between the fresh-install schema and migration
002 — text-only, no SQL is ever executed and no database is touched. The
migration itself is applied to no database in Phase 2 (see the guide)."""
import re
from pathlib import Path

from src.common.config import PROJECT_ROOT

def _strip_sql_comments(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))


SCHEMA_SQL = (PROJECT_ROOT / "infrastructure" / "postgres" / "lib" / "schema.sql").read_text()
MIGRATION_SQL = (
    PROJECT_ROOT / "infrastructure" / "postgres" / "migrations" / "002_pipeline_run_provenance.sql"
).read_text()
MIGRATION_EXECUTABLE_SQL = _strip_sql_comments(MIGRATION_SQL)

NEW_COLUMNS = [
    "trigger_source",
    "git_sha",
    "config_snapshot",
    "config_hash",
    "dataset_version",
    "model_version",
    "artifacts",
    "error_type",
    "error_message",
    "heartbeat_at",
]


def test_migration_file_exists_with_next_unused_number():
    migrations_dir = PROJECT_ROOT / "infrastructure" / "postgres" / "migrations"
    numbers = sorted(int(p.name[:3]) for p in migrations_dir.glob("*.sql"))
    # v1.3 Phase 1 adds migration 003 (channel_events/source_alerts/
    # synthetic_event_labels/channel_model_bundles) -- this file otherwise
    # covers migration 002 specifically and is unaffected by that addition.
    assert numbers == [1, 2, 3]


def test_every_new_column_appears_in_schema_sql():
    for column in NEW_COLUMNS:
        assert re.search(rf"\b{column}\b", SCHEMA_SQL), f"{column!r} missing from schema.sql"


def test_every_new_column_added_idempotently_in_migration():
    for column in NEW_COLUMNS:
        assert re.search(
            rf"ADD COLUMN IF NOT EXISTS {column}\b", MIGRATION_SQL
        ), f"{column!r} not added idempotently in the migration"


def test_migration_never_uses_unsupported_add_constraint_if_not_exists():
    assert "ADD CONSTRAINT IF NOT EXISTS" not in MIGRATION_EXECUTABLE_SQL


def test_migration_drops_before_adding_each_named_constraint():
    for constraint in (
        "pipeline_runs_pipeline_name_check",
        "pipeline_runs_status_check",
        "pipeline_runs_error_message_length_check",
    ):
        assert f"DROP CONSTRAINT IF EXISTS {constraint}" in MIGRATION_SQL
        assert f"ADD CONSTRAINT {constraint}" in MIGRATION_SQL


def test_migration_is_wrapped_in_a_single_transaction():
    assert MIGRATION_SQL.strip().startswith("BEGIN;") or "\nBEGIN;" in MIGRATION_SQL
    assert MIGRATION_SQL.rstrip().endswith("COMMIT;")


def _extract_check_values(sql: str, constraint_pattern: str) -> set[str]:
    match = re.search(constraint_pattern + r"\s*\(([^)]*)\)", sql)
    assert match, f"pattern {constraint_pattern!r} not found"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def test_pipeline_name_allowed_values_match_between_schema_and_migration():
    schema_values = _extract_check_values(SCHEMA_SQL, r"CHECK \(pipeline_name IN")
    migration_values = _extract_check_values(MIGRATION_SQL, r"CHECK \(pipeline_name IN")
    assert schema_values == migration_values == {"batch", "train", "stream"}


def test_status_allowed_values_match_between_schema_and_migration():
    schema_values = _extract_check_values(SCHEMA_SQL, r"CHECK \(status IN")
    migration_values = _extract_check_values(MIGRATION_SQL, r"CHECK \(status IN")
    assert schema_values == migration_values == {"PENDING", "RUNNING", "SUCCESS", "FAILED", "CANCELLED"}


def test_error_message_length_limit_matches_runs_module_constant():
    from src.control_plane.runs import MAX_ERROR_MESSAGE_LENGTH

    assert str(MAX_ERROR_MESSAGE_LENGTH) in SCHEMA_SQL
    assert str(MAX_ERROR_MESSAGE_LENGTH) in MIGRATION_SQL

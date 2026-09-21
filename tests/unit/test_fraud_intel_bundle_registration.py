"""Phase 4: ChannelModelBundleStore -- injectable pattern, candidate-bundle
shape, and the incomplete-bundle (missing anomaly component) promotion
safeguard. No database, Docker, or network access anywhere in this file
-- every test uses _FakeChannelModelBundleStore.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.fraud_intel.models.bundle import (
    REQUIRED_OPERATIONAL_COMPONENTS,
    IncompleteBundleError,
    _BUNDLE_VERSION_LOCK_NAMESPACE,
    _FakeChannelModelBundleStore,
    _PostgresChannelModelBundleStore,
    validate_promotion_eligible,
)


def _complete_fields(**overrides) -> dict:
    # Phase 5 decision 3: four additional fields (rule/graph/ensemble/
    # reason-code versions) are now part of REQUIRED_OPERATIONAL_COMPONENTS.
    base = dict(
        channel="online_banking",
        gbm_model_version="1",
        lr_model_version="1",
        anomaly_model_version="1",
        preprocessing_artifact_version="pp-abc123",
        feature_schema_version="v1",
        rule_set_version="v1",
        graph_policy_version="v1",
        ensemble_policy_version="v1",
        reason_code_version="v1",
        training_run_id=42,
        dataset_version="ds-abc123",
        evaluation_report_ref="{}",
    )
    base.update(overrides)
    return base


def test_register_candidate_starts_at_version_one():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    assert bundle.bundle_version == 1
    assert bundle.status == "CANDIDATE"
    assert bundle.bundle_id == 1


def test_register_candidate_increments_version_per_channel():
    store = _FakeChannelModelBundleStore()
    first = store.register_candidate(**_complete_fields())
    second = store.register_candidate(**_complete_fields())
    assert first.bundle_version == 1
    assert second.bundle_version == 2
    assert first.bundle_id != second.bundle_id


def test_register_candidate_versions_are_independent_per_channel():
    store = _FakeChannelModelBundleStore()
    ob_1 = store.register_candidate(**_complete_fields(channel="online_banking"))
    ach_1 = store.register_candidate(**_complete_fields(channel="ach"))
    ob_2 = store.register_candidate(**_complete_fields(channel="online_banking"))
    assert ob_1.bundle_version == 1
    assert ach_1.bundle_version == 1
    assert ob_2.bundle_version == 2


def test_bundle_versions_are_immutable_once_registered():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    with pytest.raises(Exception):
        bundle.bundle_version = 99  # frozen-like pydantic model rejects mutation attempts differently per config


# ---- Phase 4's incomplete (anomaly-less) candidate bundle -------------------------


def test_phase4_candidate_bundle_has_anomaly_model_version_null():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields(anomaly_model_version=None))
    assert bundle.anomaly_model_version is None
    assert bundle.status == "CANDIDATE"


def test_incomplete_bundle_is_not_promotion_eligible():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields(anomaly_model_version=None))
    assert bundle.is_promotion_eligible() is False
    with pytest.raises(IncompleteBundleError, match="anomaly_model_version"):
        validate_promotion_eligible(bundle)


def test_complete_bundle_is_promotion_eligible():
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields())
    assert bundle.is_promotion_eligible() is True
    validate_promotion_eligible(bundle)  # must not raise


@pytest.mark.parametrize("missing_field", REQUIRED_OPERATIONAL_COMPONENTS)
def test_promotion_rejects_a_bundle_missing_any_single_required_component(missing_field):
    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(**_complete_fields(**{missing_field: None}))
    with pytest.raises(IncompleteBundleError, match=missing_field):
        validate_promotion_eligible(bundle)


def test_phase5_registers_a_new_bundle_version_never_mutates_phase4s_row():
    """Phase 4's candidate bundle is never updated in place -- a later
    "Phase 5" registration (simulated here as a second register_candidate
    call with a complete component set) must produce a NEW row, leaving
    the first bundle's own fields exactly as they were."""
    store = _FakeChannelModelBundleStore()
    phase4_bundle = store.register_candidate(**_complete_fields(anomaly_model_version=None))
    phase4_bundle_snapshot = phase4_bundle.model_copy(deep=True)

    phase5_bundle = store.register_candidate(**_complete_fields(anomaly_model_version="1"))

    assert phase5_bundle.bundle_id != phase4_bundle.bundle_id
    assert phase5_bundle.bundle_version == phase4_bundle.bundle_version + 1
    assert phase4_bundle == phase4_bundle_snapshot  # untouched
    assert phase4_bundle.anomaly_model_version is None  # still incomplete, still CANDIDATE
    assert phase4_bundle.status == "CANDIDATE"
    assert len(store.rows) == 2


# ============================ Phase 7B Stage 3: real-SQL-shape bundle-version locking ==
#
# The previous SQL (`SELECT ... FOR UPDATE` combined with `MAX(...)`) was
# rejected outright by real Postgres on the very first real training run:
# `psycopg2.errors.FeatureNotSupported: FOR UPDATE is not allowed with
# aggregate functions`. This section exercises _PostgresChannelModelBundleStore
# (never exercised before this corrective pass) against a small, purpose-
# built in-memory fake connection/cursor that models exactly the three SQL
# statements register_candidate() now issues -- proving the real fix
# without a real database.


class _FakeBundleStoreConnection:
    def __init__(self):
        self.rows: list[dict] = []
        self._next_id = 1
        self.calls: list[tuple] = []
        self.raise_on_insert: Exception | None = None

    def cursor(self, cursor_factory=None):
        return _FakeBundleStoreCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


class _FakeBundleStoreCursor:
    def __init__(self, conn: _FakeBundleStoreConnection):
        self._conn = conn
        self._result: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        import re

        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT pg_advisory_xact_lock"):
            self._conn.calls.append(("lock", tuple(params)))
            self._result = [{"pg_advisory_xact_lock": None}]
        elif normalized.startswith("SELECT COALESCE(MAX(bundle_version)"):
            self._conn.calls.append(("max", tuple(params)))
            (channel,) = params
            existing = [r["bundle_version"] for r in self._conn.rows if r["channel"] == channel]
            self._result = [{"max_version": max(existing, default=0)}]
        elif normalized.startswith("INSERT INTO channel_model_bundles"):
            self._conn.calls.append(("insert", tuple(params)))
            if self._conn.raise_on_insert is not None:
                raise self._conn.raise_on_insert
            match = re.search(r"\(([^)]+)\)\s+VALUES", sql)
            columns = [c.strip() for c in match.group(1).split(",")]
            row = dict(zip(columns, params))
            row.setdefault("bundle_id", self._conn._next_id)
            self._conn._next_id += 1
            row.setdefault("created_at", datetime(2026, 1, 1, tzinfo=timezone.utc))
            self._conn.rows.append(row)
            self._result = [row]
        else:
            raise AssertionError(f"unexpected SQL in fake: {normalized[:100]}")

    def fetchone(self):
        return self._result[0] if self._result else None


def _patch_bundle_connection(monkeypatch, conn, *, count_calls: list | None = None):
    def _get_connection(database=None):
        if count_calls is not None:
            count_calls.append(database)
        return conn

    monkeypatch.setattr("src.fraud_intel.models.bundle.get_connection", _get_connection)


def test_the_invalid_for_update_aggregate_query_no_longer_exists():
    """AST-based (Phase 7A convention, avoids docstring-substring false
    positives -- the method's own docstring legitimately explains why
    FOR UPDATE is no longer used): no .execute(...) call's SQL string
    literal contains 'FOR UPDATE' anywhere in this method's source."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(_PostgresChannelModelBundleStore.register_candidate)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            sql_arg = node.args[0]
            if isinstance(sql_arg, ast.Constant) and isinstance(sql_arg.value, str):
                assert "FOR UPDATE" not in sql_arg.value


def test_advisory_lock_executes_before_the_max_query(monkeypatch):
    conn = _FakeBundleStoreConnection()
    _patch_bundle_connection(monkeypatch, conn)
    store = _PostgresChannelModelBundleStore(database="aidp_test")

    store.register_candidate(**_complete_fields())

    kinds = [call[0] for call in conn.calls]
    assert kinds.index("lock") < kinds.index("max")
    # the lock call is parameterized with the documented namespace + the channel value
    lock_call = next(call for call in conn.calls if call[0] == "lock")
    assert lock_call[1] == (_BUNDLE_VERSION_LOCK_NAMESPACE, "online_banking")


def test_lock_max_and_insert_use_the_same_connection_and_transaction(monkeypatch):
    conn = _FakeBundleStoreConnection()
    connection_requests: list = []
    _patch_bundle_connection(monkeypatch, conn, count_calls=connection_requests)
    store = _PostgresChannelModelBundleStore(database="aidp_test")

    store.register_candidate(**_complete_fields())

    assert len(connection_requests) == 1  # exactly one get_connection() call for the whole operation
    kinds = [call[0] for call in conn.calls]
    assert kinds == ["lock", "max", "insert"]  # all three against the one fake connection's shared call log


def test_first_candidate_for_a_channel_gets_bundle_version_one(monkeypatch):
    conn = _FakeBundleStoreConnection()
    _patch_bundle_connection(monkeypatch, conn)
    store = _PostgresChannelModelBundleStore(database="aidp_test")

    bundle = store.register_candidate(**_complete_fields(channel="wire"))
    assert bundle.bundle_version == 1
    assert bundle.status == "CANDIDATE"


def test_a_subsequent_candidate_for_the_same_channel_gets_bundle_version_two(monkeypatch):
    conn = _FakeBundleStoreConnection()
    _patch_bundle_connection(monkeypatch, conn)
    store = _PostgresChannelModelBundleStore(database="aidp_test")

    first = store.register_candidate(**_complete_fields(channel="wire"))
    second = store.register_candidate(**_complete_fields(channel="wire"))
    assert first.bundle_version == 1
    assert second.bundle_version == 2


def test_different_channels_maintain_independent_version_sequences_against_real_sql_shape(monkeypatch):
    conn = _FakeBundleStoreConnection()
    _patch_bundle_connection(monkeypatch, conn)
    store = _PostgresChannelModelBundleStore(database="aidp_test")

    ob_1 = store.register_candidate(**_complete_fields(channel="online_banking"))
    ach_1 = store.register_candidate(**_complete_fields(channel="ach"))
    ob_2 = store.register_candidate(**_complete_fields(channel="online_banking"))
    assert ob_1.bundle_version == 1
    assert ach_1.bundle_version == 1
    assert ob_2.bundle_version == 2


def test_inserted_status_remains_candidate_against_real_sql_shape(monkeypatch):
    conn = _FakeBundleStoreConnection()
    _patch_bundle_connection(monkeypatch, conn)
    store = _PostgresChannelModelBundleStore(database="aidp_test")

    bundle = store.register_candidate(**_complete_fields())
    assert bundle.status == "CANDIDATE"


def test_register_candidate_never_writes_operational_status():
    """AST-based (Phase 7A convention): the only status string literal
    anywhere in register_candidate()'s own source is 'CANDIDATE' -- the
    partial-unique-OPERATIONAL-per-channel constraint and all promotion
    logic (src.fraud_intel.models.promotion) are untouched by this
    corrective pass."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(_PostgresChannelModelBundleStore.register_candidate)))
    string_literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "OPERATIONAL" not in string_literals
    assert "CANDIDATE" in string_literals


def test_lock_and_max_queries_are_parameterized_string_literals():
    """AST-based: the advisory-lock and MAX SQL text are plain string
    literals (never an f-string/concatenation built from `fields`), and
    both pass their values through a separate params argument -- never
    string-interpolated into the SQL itself."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(_PostgresChannelModelBundleStore.register_candidate)))
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            sql_arg = node.args[0]
            if isinstance(sql_arg, ast.Constant) and isinstance(sql_arg.value, str) and (
                "pg_advisory_xact_lock" in sql_arg.value or "MAX(bundle_version)" in sql_arg.value
            ):
                checked += 1
                assert "%s" in sql_arg.value
                assert len(node.args) == 2, "lock/MAX query must pass a separate params argument"
    assert checked == 2  # both the lock call and the MAX call were found and checked


def test_a_database_exception_during_insert_still_propagates_and_returns_no_bundle(monkeypatch):
    """psycopg2's own `with conn:` context manager rolls back on any
    exception raised inside it -- unchanged by this fix, and not
    reimplemented by this fake; what IS proven here is that
    register_candidate() itself never swallows the exception or returns
    a partial/fabricated bundle record when the INSERT fails."""
    conn = _FakeBundleStoreConnection()
    conn.raise_on_insert = RuntimeError("simulated unique constraint violation")
    _patch_bundle_connection(monkeypatch, conn)
    store = _PostgresChannelModelBundleStore(database="aidp_test")

    with pytest.raises(RuntimeError, match="simulated unique constraint violation"):
        store.register_candidate(**_complete_fields())

    assert conn.rows == []

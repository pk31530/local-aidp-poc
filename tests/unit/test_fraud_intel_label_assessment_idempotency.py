"""Phase 7B Stage 7 corrective pass: `load_alert_label_bases()`'s real
generation-run validation/scoping SQL, and `_PostgresLabelAssessmentStore.
append_if_changed()`'s real advisory-lock/compare/insert SQL, exercised
against small, purpose-built in-memory fake Postgres connection/cursor
doubles -- proving the fix for the two gaps discovered during Stage 7's
read-only preflight (load_alert_label_bases() used to be scoped only by
channel, and label_assessments had zero retry/idempotency protection)
without touching real infrastructure. No database, Docker, MLflow, or
network access anywhere in this file.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

import pytest

from src.fraud_intel.cli_data_access import (
    GenerationRunChannelMismatchError,
    GenerationRunDatasetVersionError,
    UnknownGenerationRunError,
    load_alert_label_bases,
)
from src.fraud_intel.labels.eligibility import (
    _LABEL_ASSESSMENT_LOCK_NAMESPACE,
    _PostgresLabelAssessmentStore,
    assess_label,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---- load_alert_label_bases(): generation-run validation and scoping -------------------


class _FakeLabelBasesConnection:
    def __init__(self):
        self.channel_events: list[dict] = []
        self.fraud_alerts: list[dict] = []
        self.synthetic_event_labels: list[dict] = []
        self.analyst_dispositions: list[dict] = []

    def cursor(self, cursor_factory=None):
        return _FakeLabelBasesCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


class _FakeLabelBasesCursor:
    def __init__(self, conn: _FakeLabelBasesConnection):
        self._conn = conn
        self._result: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT DISTINCT channel FROM channel_events WHERE generation_run_id"):
            (run_id,) = params
            channels = {r["channel"] for r in self._conn.channel_events if r["generation_run_id"] == run_id}
            self._result = [{"channel": c} for c in channels]
        elif normalized.startswith("SELECT DISTINCT dataset_version FROM channel_events"):
            channel, run_id = params
            versions = {
                r["dataset_version"] for r in self._conn.channel_events
                if r["channel"] == channel and r["generation_run_id"] == run_id
            }
            self._result = [{"dataset_version": v} for v in versions]
        elif normalized.startswith("SELECT fa.alert_id, ce.event_timestamp, sel.synthetic_scenario_label"):
            channel, generation_run_id = params
            rows = []
            for fa in self._conn.fraud_alerts:
                if fa["channel"] != channel:
                    continue
                ce = next((r for r in self._conn.channel_events if r["event_id"] == fa["event_id"]), None)
                if ce is None or ce["generation_run_id"] != generation_run_id:
                    continue
                sel = next((r for r in self._conn.synthetic_event_labels if r["event_id"] == fa["event_id"]), None)
                if sel is None:
                    continue
                dispositions = sorted(
                    (d for d in self._conn.analyst_dispositions if d["alert_id"] == fa["alert_id"]),
                    key=lambda d: (d["disposed_at"], d["disposition_id"]), reverse=True,
                )
                latest_disposition = dispositions[0] if dispositions else None
                rows.append({
                    "alert_id": fa["alert_id"],
                    "event_timestamp": ce["event_timestamp"],
                    "synthetic_scenario_label": sel["synthetic_scenario_label"],
                    "disposition_id": latest_disposition["disposition_id"] if latest_disposition else None,
                    "disposition": latest_disposition["disposition"] if latest_disposition else None,
                    "disposed_at": latest_disposition["disposed_at"] if latest_disposition else None,
                })
            self._result = rows
        else:
            raise AssertionError(f"unexpected SQL in fake: {normalized[:120]}")

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result[0] if self._result else None


def _patch(monkeypatch, conn):
    monkeypatch.setattr("src.fraud_intel.cli_data_access.get_connection", lambda database=None: conn)


def _add_event_and_alert(conn, alert_id, *, channel="online_banking", generation_run_id, dataset_version, fraud=True, ts=T0):
    event_id = uuid.uuid4()
    conn.channel_events.append({"event_id": event_id, "channel": channel, "generation_run_id": generation_run_id, "dataset_version": dataset_version, "event_timestamp": ts})
    conn.fraud_alerts.append({"alert_id": alert_id, "channel": channel, "event_id": event_id})
    conn.synthetic_event_labels.append({"event_id": event_id, "synthetic_scenario_label": fraud})


def test_unknown_generation_run_raises(monkeypatch):
    conn = _FakeLabelBasesConnection()
    _patch(monkeypatch, conn)
    with pytest.raises(UnknownGenerationRunError):
        load_alert_label_bases("online_banking", "aidp_test", generation_run_id="genrun-does-not-exist")


def test_generation_channel_mismatch_raises(monkeypatch):
    conn = _FakeLabelBasesConnection()
    _patch(monkeypatch, conn)
    _add_event_and_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")  # channel=online_banking
    with pytest.raises(GenerationRunChannelMismatchError):
        load_alert_label_bases("wire", "aidp_test", generation_run_id="genrun-A")


def test_ambiguous_dataset_version_raises(monkeypatch):
    conn = _FakeLabelBasesConnection()
    _patch(monkeypatch, conn)
    _add_event_and_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_event_and_alert(conn, 2, generation_run_id="genrun-A", dataset_version="dsv-A-DIFFERENT")
    with pytest.raises(GenerationRunDatasetVersionError):
        load_alert_label_bases("online_banking", "aidp_test", generation_run_id="genrun-A")


def test_two_generations_for_one_channel_never_mix(monkeypatch):
    conn = _FakeLabelBasesConnection()
    _patch(monkeypatch, conn)
    _add_event_and_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_event_and_alert(conn, 2, generation_run_id="genrun-B", dataset_version="dsv-B")

    dataset_version, bases = load_alert_label_bases("online_banking", "aidp_test", generation_run_id="genrun-A")
    assert dataset_version == "dsv-A"
    assert [b.alert_id for b in bases] == [1]


def test_returns_dataset_version_and_synthetic_bases(monkeypatch):
    conn = _FakeLabelBasesConnection()
    _patch(monkeypatch, conn)
    _add_event_and_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A", fraud=True)
    _add_event_and_alert(conn, 2, generation_run_id="genrun-A", dataset_version="dsv-A", fraud=False)

    dataset_version, bases = load_alert_label_bases("online_banking", "aidp_test", generation_run_id="genrun-A")
    assert dataset_version == "dsv-A"
    by_id = {b.alert_id: b for b in bases}
    assert by_id[1].label_source == "SYNTHETIC_GENERATOR"
    assert by_id[1].resolved_label == "RESOLVED_FRAUD"
    assert by_id[1].source_disposition_id is None
    assert by_id[2].resolved_label == "RESOLVED_LEGITIMATE"


def test_analyst_disposition_takes_priority_over_synthetic_label(monkeypatch):
    conn = _FakeLabelBasesConnection()
    _patch(monkeypatch, conn)
    _add_event_and_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A", fraud=True)
    disposed_at = T0
    conn.analyst_dispositions.append({"alert_id": 1, "disposition_id": 99, "disposition": "CONFIRMED_LEGITIMATE", "disposed_at": disposed_at})

    _, bases = load_alert_label_bases("online_banking", "aidp_test", generation_run_id="genrun-A")
    (basis,) = bases
    assert basis.label_source == "ANALYST_DISPOSITION"
    assert basis.source_disposition_id == 99
    assert basis.resolved_label == "RESOLVED_LEGITIMATE"
    assert basis.basis_timestamp == disposed_at


# ---- append_if_changed(): advisory lock ordering, single transaction, lock keys --------


class _FakeAssessmentConnection:
    def __init__(self):
        self.rows: list[dict] = []
        self._next_id = 1
        self.calls: list[tuple] = []

    def cursor(self, cursor_factory=None):
        return _FakeAssessmentCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


class _FakeAssessmentCursor:
    def __init__(self, conn: _FakeAssessmentConnection):
        self._conn = conn
        self._result: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT pg_advisory_xact_lock"):
            self._conn.calls.append(("lock", tuple(params)))
            self._result = [{"pg_advisory_xact_lock": None}]
        elif normalized.startswith("SELECT * FROM label_assessments WHERE alert_id"):
            self._conn.calls.append(("select_latest", tuple(params)))
            (alert_id,) = params
            matching = sorted(
                (r for r in self._conn.rows if r["alert_id"] == alert_id),
                key=lambda r: (r["evaluated_at"], r["assessment_id"]), reverse=True,
            )
            self._result = matching[:1]
        elif normalized.startswith("INSERT INTO label_assessments"):
            self._conn.calls.append(("insert", tuple(params)))
            match = re.search(r"\(([^)]+)\)\s+VALUES", sql)
            columns = [c.strip() for c in match.group(1).split(",")]
            row = dict(zip(columns, params))
            row["assessment_id"] = self._conn._next_id
            self._conn._next_id += 1
            row["evaluated_at"] = datetime.now(timezone.utc)
            self._conn.rows.append(row)
            self._result = [row]
        else:
            raise AssertionError(f"unexpected SQL in fake: {normalized[:120]}")

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result[0] if self._result else None


def _patch_assessment_connection(monkeypatch, conn, *, count_calls: list | None = None):
    def _get_connection(database=None):
        if count_calls is not None:
            count_calls.append(database)
        return conn

    monkeypatch.setattr("src.fraud_intel.labels.eligibility.get_connection", _get_connection)


def _fields(alert_id, **overrides):
    base = assess_label(alert_id=alert_id, label_source="SYNTHETIC_GENERATOR", basis_timestamp=T0, source_disposition_id=None, resolved_label="RESOLVED_FRAUD", now=T0)
    base.update(overrides)
    return base


def test_lock_executes_before_select_and_insert(monkeypatch):
    conn = _FakeAssessmentConnection()
    _patch_assessment_connection(monkeypatch, conn)
    store = _PostgresLabelAssessmentStore(database="aidp_test")

    store.append_if_changed(**_fields(1))

    kinds = [call[0] for call in conn.calls]
    assert kinds.index("lock") < kinds.index("select_latest") < kinds.index("insert")


def test_lock_select_and_insert_use_the_same_connection_and_transaction(monkeypatch):
    conn = _FakeAssessmentConnection()
    connection_requests: list = []
    _patch_assessment_connection(monkeypatch, conn, count_calls=connection_requests)
    store = _PostgresLabelAssessmentStore(database="aidp_test")

    store.append_if_changed(**_fields(1))

    assert len(connection_requests) == 1  # exactly one get_connection() call for the whole operation
    kinds = [call[0] for call in conn.calls]
    assert kinds == ["lock", "select_latest", "insert"]


def test_same_alert_attempts_use_the_same_lock_key(monkeypatch):
    conn = _FakeAssessmentConnection()
    _patch_assessment_connection(monkeypatch, conn)
    store = _PostgresLabelAssessmentStore(database="aidp_test")

    store.append_if_changed(**_fields(1))
    store.append_if_changed(**_fields(1, resolved_label="RESOLVED_LEGITIMATE"))

    lock_calls = [call[1] for call in conn.calls if call[0] == "lock"]
    assert len(lock_calls) == 2
    assert lock_calls[0] == lock_calls[1]
    assert lock_calls[0] == ((_LABEL_ASSESSMENT_LOCK_NAMESPACE, 1))


def test_different_alerts_use_independent_lock_keys(monkeypatch):
    conn = _FakeAssessmentConnection()
    _patch_assessment_connection(monkeypatch, conn)
    store = _PostgresLabelAssessmentStore(database="aidp_test")

    store.append_if_changed(**_fields(1))
    store.append_if_changed(**_fields(2))

    lock_calls = [call[1] for call in conn.calls if call[0] == "lock"]
    assert lock_calls[0] != lock_calls[1]
    assert lock_calls[0] == (_LABEL_ASSESSMENT_LOCK_NAMESPACE, 1)
    assert lock_calls[1] == (_LABEL_ASSESSMENT_LOCK_NAMESPACE, 2)


def test_exact_retry_against_real_sql_shape_skips_the_insert(monkeypatch):
    conn = _FakeAssessmentConnection()
    _patch_assessment_connection(monkeypatch, conn)
    store = _PostgresLabelAssessmentStore(database="aidp_test")

    fields = _fields(1)
    first = store.append_if_changed(**fields)
    second = store.append_if_changed(**fields)

    assert first.inserted is True
    assert second.inserted is False
    assert len(conn.rows) == 1
    kinds = [call[0] for call in conn.calls]
    assert kinds.count("insert") == 1


def test_changed_input_against_real_sql_shape_still_inserts(monkeypatch):
    conn = _FakeAssessmentConnection()
    _patch_assessment_connection(monkeypatch, conn)
    store = _PostgresLabelAssessmentStore(database="aidp_test")

    store.append_if_changed(**_fields(1))
    result = store.append_if_changed(**_fields(1, resolved_label="RESOLVED_LEGITIMATE"))

    assert result.inserted is True
    assert len(conn.rows) == 2


def test_never_updates_or_deletes_existing_rows_against_real_sql_shape(monkeypatch):
    """AST-based: append_if_changed()'s own source never calls an UPDATE
    or DELETE statement -- the only mutating SQL is the plain INSERT."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(_PostgresLabelAssessmentStore.append_if_changed)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            sql_arg = node.args[0]
            if isinstance(sql_arg, ast.Constant) and isinstance(sql_arg.value, str):
                upper = sql_arg.value.upper()
                assert "UPDATE " not in upper
                assert "DELETE " not in upper

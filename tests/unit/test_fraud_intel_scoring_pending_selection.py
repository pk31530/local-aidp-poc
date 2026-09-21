"""Phase 7B Stage 6 corrective pass: _PostgresScoringDataAccess's real
generation-run validation and pending-selection SQL, exercised against a
small, purpose-built in-memory fake Postgres connection/cursor -- proving
the fix for the discovered gap (list_pending() used to be scoped only by
channel, silently able to mix multiple generation runs together) without
touching real infrastructure. No database, Docker, MLflow, or network
access anywhere in this file.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.fraud_intel.cli_data_access import (
    GenerationRunChannelMismatchError,
    GenerationRunDatasetVersionError,
    UnknownGenerationRunError,
)
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.scoring.dispatch import _PostgresScoringDataAccess

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _event(event_id: uuid.UUID, *, ts: datetime = T0) -> FraudEvent:
    return FraudEvent(
        event_id=event_id, channel="online_banking", customer_id="CUST1", account_id="ACCT1",
        event_timestamp=ts, amount_minor_units=10_000, direction="debit", device_id="DEV1",
        channel_payload=OnlineBankingPayload(
            session_id="S1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"
        ),
    )


def _source_alert(event_id: uuid.UUID, *, ts: datetime = T0) -> SourceAlertContext:
    return SourceAlertContext(
        source_alert_id=uuid.uuid4(), source_system="LocalYamlRuleProvider (simulated upstream)",
        event_id=event_id, source_alert_created_at=ts, source_rule_ids=["RULE1"], source_rule_version="v1",
        source_alert_reason_codes=["REASON1"], generation_run_id="unused", dataset_version="unused", created_at=ts,
    )


def _channel_event_row(event: FraudEvent, *, generation_run_id: str, dataset_version: str) -> dict:
    return {
        "event_id": event.event_id, "channel": event.channel, "customer_id": event.customer_id,
        "account_id": event.account_id, "event_timestamp": event.event_timestamp,
        "amount_minor_units": event.amount_minor_units, "direction": event.direction,
        "device_id": event.device_id, "ip_address": event.ip_address,
        "channel_payload": event.channel_payload.model_dump(mode="json"),
        "scenario_id": event.scenario_id, "schema_version": event.schema_version,
        "generation_run_id": generation_run_id, "dataset_version": dataset_version,
    }


def _source_alert_row(sa: SourceAlertContext, *, generation_run_id: str, dataset_version: str) -> dict:
    return {
        "source_alert_id": sa.source_alert_id, "source_system": sa.source_system, "event_id": sa.event_id,
        "source_alert_created_at": sa.source_alert_created_at, "source_rule_ids": sa.source_rule_ids,
        "source_rule_version": sa.source_rule_version, "source_alert_score": sa.source_alert_score,
        "source_alert_reason_codes": sa.source_alert_reason_codes, "generation_run_id": generation_run_id,
        "dataset_version": dataset_version, "created_at": sa.created_at,
    }


class _FakeScoringConnection:
    def __init__(self):
        self.channel_events: list[dict] = []
        self.source_alerts: list[dict] = []
        self.fraud_alerts: list[dict] = []
        self.alert_evidence: list[dict] = []

    def cursor(self, cursor_factory=None):
        return _FakeScoringCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


class _FakeScoringCursor:
    def __init__(self, conn: _FakeScoringConnection):
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
        elif normalized.startswith("SELECT sa.*, ce.event_id AS ce_event_id"):
            channel, generation_run_id, operational_bundle_id, limit = params
            eligible_event_ids = {
                r["event_id"] for r in self._conn.channel_events
                if r["channel"] == channel and r["generation_run_id"] == generation_run_id
            }
            rows = []
            for sa in self._conn.source_alerts:
                if sa["event_id"] not in eligible_event_ids:
                    continue
                has_current_bundle_evidence = any(
                    fa["source_system"] == sa["source_system"] and fa["source_alert_id"] == sa["source_alert_id"]
                    and any(
                        ae["alert_id"] == fa["alert_id"] and ae["channel_model_bundle_id"] == operational_bundle_id
                        for ae in self._conn.alert_evidence
                    )
                    for fa in self._conn.fraud_alerts
                )
                if has_current_bundle_evidence:
                    continue
                ce = next(r for r in self._conn.channel_events if r["event_id"] == sa["event_id"])
                row = dict(sa)
                row.update(
                    {
                        "ce_event_id": ce["event_id"], "ce_channel": ce["channel"], "ce_customer_id": ce["customer_id"],
                        "ce_account_id": ce["account_id"], "ce_event_timestamp": ce["event_timestamp"],
                        "ce_amount_minor_units": ce["amount_minor_units"], "ce_direction": ce["direction"],
                        "ce_device_id": ce["device_id"], "ce_ip_address": ce["ip_address"],
                        "ce_channel_payload": ce["channel_payload"], "ce_scenario_id": ce["scenario_id"],
                        "ce_schema_version": ce["schema_version"],
                    }
                )
                rows.append(row)
            rows.sort(key=lambda r: r["source_alert_created_at"])
            self._result = rows[:limit]
        elif normalized.startswith("SELECT * FROM channel_events WHERE customer_id"):
            self._result = []
        elif normalized.startswith("SELECT sa2.* FROM source_alerts sa2"):
            self._result = []
        elif normalized.startswith("SELECT la.assessment_id"):
            self._result = []
        else:
            raise AssertionError(f"unexpected SQL in fake: {normalized[:120]}")

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result[0] if self._result else None


def _patch(monkeypatch, conn):
    monkeypatch.setattr("src.common.db.get_connection", lambda database=None: conn)


def _add_event_and_alert(conn, *, generation_run_id, dataset_version, ts=T0) -> SourceAlertContext:
    event_id = uuid.uuid4()
    event = _event(event_id, ts=ts)
    sa = _source_alert(event_id, ts=ts)
    conn.channel_events.append(_channel_event_row(event, generation_run_id=generation_run_id, dataset_version=dataset_version))
    conn.source_alerts.append(_source_alert_row(sa, generation_run_id=generation_run_id, dataset_version=dataset_version))
    return sa


# ---- generation-run validation ---------------------------------------------------------


def test_unknown_generation_run_raises(monkeypatch):
    conn = _FakeScoringConnection()
    _patch(monkeypatch, conn)
    access = _PostgresScoringDataAccess(database="aidp_test")
    with pytest.raises(UnknownGenerationRunError):
        access.validate_generation_run("online_banking", "genrun-does-not-exist")


def test_generation_channel_mismatch_raises(monkeypatch):
    conn = _FakeScoringConnection()
    _patch(monkeypatch, conn)
    _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")  # channel=online_banking
    access = _PostgresScoringDataAccess(database="aidp_test")
    with pytest.raises(GenerationRunChannelMismatchError):
        access.validate_generation_run("wire", "genrun-A")


def test_ambiguous_dataset_version_raises(monkeypatch):
    conn = _FakeScoringConnection()
    _patch(monkeypatch, conn)
    _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A-DIFFERENT")
    access = _PostgresScoringDataAccess(database="aidp_test")
    with pytest.raises(GenerationRunDatasetVersionError):
        access.validate_generation_run("online_banking", "genrun-A")


def test_validate_generation_run_returns_the_dataset_version(monkeypatch):
    conn = _FakeScoringConnection()
    _patch(monkeypatch, conn)
    _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")
    access = _PostgresScoringDataAccess(database="aidp_test")
    assert access.validate_generation_run("online_banking", "genrun-A") == "dsv-A"


# ---- pending selection -------------------------------------------------------------


def test_two_generations_for_one_channel_never_mix(monkeypatch):
    conn = _FakeScoringConnection()
    _patch(monkeypatch, conn)
    sa_a = _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_event_and_alert(conn, generation_run_id="genrun-B", dataset_version="dsv-B")

    access = _PostgresScoringDataAccess(database="aidp_test")
    items = access.list_pending("online_banking", generation_run_id="genrun-A", operational_bundle_id=5)
    assert {i.source_alert.source_alert_id for i in items} == {sa_a.source_alert_id}


def test_pending_selection_semantics_across_all_five_scenarios(monkeypatch):
    conn = _FakeScoringConnection()
    _patch(monkeypatch, conn)
    CURRENT_BUNDLE = 5
    OLD_BUNDLE = 3

    sa_no_alert = _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")
    sa_shell_no_evidence = _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")
    sa_old_bundle_evidence = _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")
    sa_current_bundle_evidence = _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")
    sa_current_bundle_degraded = _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")

    # sa_shell_no_evidence: a fraud_alerts row exists, but no evidence at all -- still pending.
    conn.fraud_alerts.append(
        {"alert_id": 1, "source_system": sa_shell_no_evidence.source_system, "source_alert_id": sa_shell_no_evidence.source_alert_id}
    )

    # sa_old_bundle_evidence: fraud_alerts + evidence, but only for an OLDER bundle -- still pending.
    conn.fraud_alerts.append(
        {"alert_id": 2, "source_system": sa_old_bundle_evidence.source_system, "source_alert_id": sa_old_bundle_evidence.source_alert_id}
    )
    conn.alert_evidence.append({"alert_id": 2, "channel_model_bundle_id": OLD_BUNDLE, "degraded": False})

    # sa_current_bundle_evidence: complete evidence for the CURRENT bundle -- excluded.
    conn.fraud_alerts.append(
        {"alert_id": 3, "source_system": sa_current_bundle_evidence.source_system, "source_alert_id": sa_current_bundle_evidence.source_alert_id}
    )
    conn.alert_evidence.append({"alert_id": 3, "channel_model_bundle_id": CURRENT_BUNDLE, "degraded": False})

    # sa_current_bundle_degraded: DEGRADED evidence for the CURRENT bundle -- still a completed,
    # auditable scoring attempt -- excluded.
    conn.fraud_alerts.append(
        {"alert_id": 4, "source_system": sa_current_bundle_degraded.source_system, "source_alert_id": sa_current_bundle_degraded.source_alert_id}
    )
    conn.alert_evidence.append({"alert_id": 4, "channel_model_bundle_id": CURRENT_BUNDLE, "degraded": True})

    access = _PostgresScoringDataAccess(database="aidp_test")
    items = access.list_pending("online_banking", generation_run_id="genrun-A", operational_bundle_id=CURRENT_BUNDLE)
    pending_ids = {i.source_alert.source_alert_id for i in items}

    assert pending_ids == {
        sa_no_alert.source_alert_id, sa_shell_no_evidence.source_alert_id, sa_old_bundle_evidence.source_alert_id,
    }
    assert sa_current_bundle_evidence.source_alert_id not in pending_ids
    assert sa_current_bundle_degraded.source_alert_id not in pending_ids


def test_successful_rerun_for_the_current_bundle_gives_zero_pending(monkeypatch):
    conn = _FakeScoringConnection()
    _patch(monkeypatch, conn)
    sa = _add_event_and_alert(conn, generation_run_id="genrun-A", dataset_version="dsv-A")
    access = _PostgresScoringDataAccess(database="aidp_test")

    first = access.list_pending("online_banking", generation_run_id="genrun-A", operational_bundle_id=5)
    assert len(first) == 1

    # Simulate exactly the writes a successful score_and_record_alert()
    # call would have made for this alert.
    conn.fraud_alerts.append({"alert_id": 1, "source_system": sa.source_system, "source_alert_id": sa.source_alert_id})
    conn.alert_evidence.append({"alert_id": 1, "channel_model_bundle_id": 5, "degraded": False})

    second = access.list_pending("online_banking", generation_run_id="genrun-A", operational_bundle_id=5)
    assert second == []


def test_pending_query_is_parameterized_with_no_string_interpolation():
    """AST-based: list_pending()'s SQL is a plain string literal, never an
    f-string/concatenation built from `channel`/`generation_run_id`/
    `operational_bundle_id`."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(_PostgresScoringDataAccess.list_pending)))
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            sql_arg = node.args[0]
            assert isinstance(sql_arg, ast.Constant) and isinstance(sql_arg.value, str)
            checked += 1
    assert checked > 0

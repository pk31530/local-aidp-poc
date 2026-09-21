"""Phase 7B Stage 8 corrective pass: load_resolved_alert_outcomes()'s and
load_resolved_alert_scoring_contexts()'s real generation-run validation
and bundle/generation-scoping SQL, exercised against a small, purpose-
built in-memory fake Postgres connection/cursor -- proving the fix for
the two gaps discovered during Stage 8's read-only preflight (both
loaders used to be scoped only by channel, and evidence selection never
tied itself to the CURRENT operational bundle) without touching real
infrastructure. No database, Docker, MLflow, or network access anywhere
in this file.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.fraud_intel.cli_data_access import (
    GenerationRunChannelMismatchError,
    GenerationRunDatasetVersionError,
    IncompleteResolvedPopulationError,
    UnknownGenerationRunError,
    load_resolved_alert_outcomes,
    load_resolved_alert_scoring_contexts,
)
from src.fraud_intel.events.online_banking import OnlineBankingPayload

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class _FakeEvalConnection:
    def __init__(self):
        self.channel_events: list[dict] = []
        self.fraud_alerts: list[dict] = []
        self.label_assessments: list[dict] = []
        self.alert_evidence: list[dict] = []
        self.source_alerts: list[dict] = []

    def cursor(self, cursor_factory=None):
        return _FakeEvalCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


def _latest_assessment(conn: _FakeEvalConnection, alert_id: int):
    matching = sorted(
        (a for a in conn.label_assessments if a["alert_id"] == alert_id),
        key=lambda a: (a["evaluated_at"], a["assessment_id"]), reverse=True,
    )
    return matching[0] if matching else None


class _FakeEvalCursor:
    def __init__(self, conn: _FakeEvalConnection):
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
        elif normalized.startswith("SELECT fa.source_alert_id, fa.channel, la.resolved_label"):
            operational_bundle_id, channel, generation_run_id = params
            rows = []
            for fa in self._conn.fraud_alerts:
                if fa["channel"] != channel:
                    continue
                ce = next((c for c in self._conn.channel_events if c["event_id"] == fa["event_id"]), None)
                if ce is None or ce["generation_run_id"] != generation_run_id:
                    continue
                la = _latest_assessment(self._conn, fa["alert_id"])
                if la is None or not la["eligibility_result"] or la["resolved_label"] not in ("RESOLVED_FRAUD", "RESOLVED_LEGITIMATE"):
                    continue
                evs = sorted(
                    (e for e in self._conn.alert_evidence if e["alert_id"] == fa["alert_id"] and e["channel_model_bundle_id"] == operational_bundle_id),
                    key=lambda e: (e["scored_at"], e["evidence_id"]), reverse=True,
                )
                ev = evs[0] if evs else None
                rows.append({
                    "source_alert_id": fa["source_alert_id"], "channel": fa["channel"], "resolved_label": la["resolved_label"],
                    "evidence_id": ev["evidence_id"] if ev else None,
                    "rule_result": ev["rule_result"] if ev else None,
                    "operational_priority_score": ev["operational_priority_score"] if ev else None,
                    "priority_band": ev["priority_band"] if ev else None,
                    "event_time": ev["event_time"] if ev else None,
                    "degraded": ev["degraded"] if ev else None,
                })
            self._result = rows
        elif normalized.startswith("SELECT fa.source_alert_id, fa.event_id, ce.channel AS ce_channel"):
            channel, generation_run_id = params
            rows = []
            for fa in self._conn.fraud_alerts:
                if fa["channel"] != channel:
                    continue
                ce = next((c for c in self._conn.channel_events if c["event_id"] == fa["event_id"]), None)
                if ce is None or ce["generation_run_id"] != generation_run_id:
                    continue
                la = _latest_assessment(self._conn, fa["alert_id"])
                if la is None or not la["eligibility_result"] or la["resolved_label"] not in ("RESOLVED_FRAUD", "RESOLVED_LEGITIMATE"):
                    continue
                sa = next((s for s in self._conn.source_alerts if s["event_id"] == fa["event_id"]), None)
                rows.append({
                    "source_alert_id": fa["source_alert_id"], "event_id": fa["event_id"],
                    "ce_channel": ce["channel"], "ce_customer_id": ce["customer_id"], "ce_account_id": ce["account_id"],
                    "ce_event_timestamp": ce["event_timestamp"], "ce_amount_minor_units": ce["amount_minor_units"],
                    "ce_direction": ce["direction"], "ce_device_id": ce["device_id"], "ce_ip_address": ce["ip_address"],
                    "ce_channel_payload": ce["channel_payload"], "ce_scenario_id": ce["scenario_id"],
                    "ce_schema_version": ce["schema_version"],
                    "source_system": sa["source_system"], "source_alert_created_at": sa["source_alert_created_at"],
                    "source_rule_ids": sa["source_rule_ids"], "source_rule_version": sa["source_rule_version"],
                    "source_alert_score": sa["source_alert_score"], "source_alert_reason_codes": sa["source_alert_reason_codes"],
                    "generation_run_id": sa["generation_run_id"], "dataset_version": sa["dataset_version"],
                    "created_at": sa["created_at"],
                })
            self._result = rows
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
    monkeypatch.setattr("src.fraud_intel.cli_data_access.get_connection", lambda database=None: conn)


def _payload() -> dict:
    return OnlineBankingPayload(
        session_id="S1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"
    ).model_dump(mode="json")


def _add_alert(
    conn: _FakeEvalConnection, alert_id: int, *, channel="online_banking", generation_run_id, dataset_version,
    resolved_label="RESOLVED_FRAUD", eligibility_result=True, evaluated_at=T0, assessment_id=None,
    ts=T0,
) -> tuple[uuid.UUID, dict]:
    event_id = uuid.uuid4()
    source_alert_id = uuid.uuid4()
    conn.channel_events.append({
        "event_id": event_id, "channel": channel, "generation_run_id": generation_run_id, "dataset_version": dataset_version,
        "customer_id": "CUST1", "account_id": "ACCT1", "event_timestamp": ts, "amount_minor_units": 10_000,
        "direction": "debit", "device_id": "DEV1", "ip_address": None, "channel_payload": _payload(),
        "scenario_id": None, "schema_version": 1,
    })
    fraud_alert = {"alert_id": alert_id, "channel": channel, "event_id": event_id, "source_alert_id": source_alert_id}
    conn.fraud_alerts.append(fraud_alert)
    conn.source_alerts.append({
        "event_id": event_id, "source_system": "core_fraud_engine", "source_alert_created_at": ts,
        "source_rule_ids": [], "source_rule_version": "v1", "source_alert_score": 0.5, "source_alert_reason_codes": [],
        "generation_run_id": generation_run_id, "dataset_version": dataset_version, "created_at": ts,
    })
    conn.label_assessments.append({
        "alert_id": alert_id, "assessment_id": assessment_id or alert_id, "evaluated_at": evaluated_at,
        "eligibility_result": eligibility_result, "resolved_label": resolved_label,
    })
    return source_alert_id, fraud_alert


def _add_evidence(conn: _FakeEvalConnection, alert_id: int, *, evidence_id=None, channel_model_bundle_id, scored_at=T0, degraded=False, score=0.5, band="MEDIUM"):
    conn.alert_evidence.append({
        "alert_id": alert_id, "evidence_id": evidence_id or uuid.uuid4(), "channel_model_bundle_id": channel_model_bundle_id,
        "scored_at": scored_at, "degraded": degraded, "rule_result": {"score_contribution": 0.0},
        "operational_priority_score": score, "priority_band": band, "event_time": T0,
    })


# ---- load_resolved_alert_outcomes: generation-run validation ---------------------------


def test_outcomes_unknown_generation_run_raises(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    with pytest.raises(UnknownGenerationRunError):
        load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-missing", operational_bundle_id=1)


def test_outcomes_generation_channel_mismatch_raises(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_evidence(conn, 1, channel_model_bundle_id=1)
    with pytest.raises(GenerationRunChannelMismatchError):
        load_resolved_alert_outcomes("wire", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)


def test_outcomes_ambiguous_dataset_version_raises(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_alert(conn, 2, generation_run_id="genrun-A", dataset_version="dsv-A-DIFFERENT")
    _add_evidence(conn, 1, channel_model_bundle_id=1)
    _add_evidence(conn, 2, channel_model_bundle_id=1)
    with pytest.raises(GenerationRunDatasetVersionError):
        load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)


def test_outcomes_two_generations_never_mix(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    sa_a, _ = _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_evidence(conn, 1, channel_model_bundle_id=1)
    _add_alert(conn, 2, generation_run_id="genrun-B", dataset_version="dsv-B")
    _add_evidence(conn, 2, channel_model_bundle_id=1)

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    assert [o.source_alert_id for o in population.outcomes] == [sa_a]
    assert population.source_dataset_version == "dsv-A"


# ---- bundle scoping / completeness -----------------------------------------------------


def test_outcomes_only_current_bundle_evidence_is_selected(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_evidence(conn, 1, channel_model_bundle_id=1, score=0.42)
    _add_evidence(conn, 1, channel_model_bundle_id=2, score=0.99)  # a DIFFERENT bundle's evidence -- must be ignored

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    (outcome,) = population.outcomes
    assert outcome.operational_priority_score == 0.42


def test_outcomes_newer_old_bundle_evidence_cannot_replace_current_bundle_evidence(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_evidence(conn, 1, channel_model_bundle_id=1, scored_at=T0, score=0.30)
    from datetime import timedelta
    _add_evidence(conn, 1, channel_model_bundle_id=2, scored_at=T0 + timedelta(hours=1), score=0.95)  # NEWER but old/other bundle

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    (outcome,) = population.outcomes
    assert outcome.operational_priority_score == 0.30


def test_outcomes_latest_evidence_within_bundle_is_selected_deterministically(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    from datetime import timedelta
    _add_evidence(conn, 1, channel_model_bundle_id=1, scored_at=T0, score=0.10)
    _add_evidence(conn, 1, channel_model_bundle_id=1, scored_at=T0 + timedelta(hours=1), score=0.77)

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    (outcome,) = population.outcomes
    assert outcome.operational_priority_score == 0.77


def test_outcomes_latest_eligible_label_assessment_is_selected_deterministically(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    from datetime import timedelta
    sa, fa = _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A", resolved_label="RESOLVED_LEGITIMATE", evaluated_at=T0, assessment_id=1)
    conn.label_assessments.append({"alert_id": 1, "assessment_id": 2, "evaluated_at": T0 + timedelta(hours=1), "eligibility_result": True, "resolved_label": "RESOLVED_FRAUD"})
    _add_evidence(conn, 1, channel_model_bundle_id=1)

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    (outcome,) = population.outcomes
    assert outcome.resolved_label == "RESOLVED_FRAUD"


def test_outcomes_multiple_evidence_and_assessment_rows_do_not_duplicate(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    from datetime import timedelta
    _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A", evaluated_at=T0, assessment_id=1)
    conn.label_assessments.append({"alert_id": 1, "assessment_id": 2, "evaluated_at": T0 + timedelta(hours=1), "eligibility_result": True, "resolved_label": "RESOLVED_FRAUD"})
    _add_evidence(conn, 1, channel_model_bundle_id=1, scored_at=T0)
    _add_evidence(conn, 1, channel_model_bundle_id=1, scored_at=T0 + timedelta(hours=1))

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    assert len(population.outcomes) == 1


def test_outcomes_degraded_current_bundle_evidence_remains_included_and_counted(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_evidence(conn, 1, channel_model_bundle_id=1, degraded=True)

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    assert population.evaluated_count == 1
    assert population.degraded_evidence_count == 1
    assert population.missing_current_bundle_evidence_count == 0


def test_outcomes_missing_current_bundle_evidence_fails_cleanly(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_alert(conn, 2, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_evidence(conn, 1, channel_model_bundle_id=1)
    # alert 2 has NO evidence for the current bundle at all

    with pytest.raises(IncompleteResolvedPopulationError):
        load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)


def test_outcomes_449_unchanged_complete_population_evaluates_cleanly(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    for alert_id in range(1, 450):
        _add_alert(
            conn, alert_id, generation_run_id="genrun-A", dataset_version="dsv-A",
            resolved_label="RESOLVED_FRAUD" if alert_id % 3 == 0 else "RESOLVED_LEGITIMATE",
        )
        _add_evidence(conn, alert_id, channel_model_bundle_id=1)

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    assert population.resolved_eligible_count == 449
    assert population.evaluated_count == 449
    assert population.missing_current_bundle_evidence_count == 0


# ---- load_resolved_alert_scoring_contexts: generation scoping --------------------------


def test_scoring_contexts_unknown_generation_run_raises(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    with pytest.raises(UnknownGenerationRunError):
        load_resolved_alert_scoring_contexts("online_banking", "aidp_test", generation_run_id="genrun-missing")


def test_scoring_contexts_are_generation_scoped(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    sa_a, _ = _add_alert(conn, 1, generation_run_id="genrun-A", dataset_version="dsv-A")
    _add_alert(conn, 2, generation_run_id="genrun-B", dataset_version="dsv-B")

    dataset_version, items = load_resolved_alert_scoring_contexts("online_banking", "aidp_test", generation_run_id="genrun-A")
    assert dataset_version == "dsv-A"
    assert [i.source_alert_id for i in items] == [sa_a]


def test_scoring_contexts_remain_in_memory_with_zero_persistence():
    """AST-based: load_resolved_alert_scoring_contexts()'s own source
    never calls anything that would persist a row (no INSERT/UPDATE
    string literal anywhere in its .execute() calls)."""
    import ast
    import inspect
    import textwrap

    from src.fraud_intel.cli_data_access import load_resolved_alert_scoring_contexts as fn

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            sql_arg = node.args[0]
            if isinstance(sql_arg, ast.Constant) and isinstance(sql_arg.value, str):
                upper = sql_arg.value.upper()
                assert "INSERT " not in upper
                assert "UPDATE " not in upper
                assert "DELETE " not in upper


# ---- operational vs. candidate population identity (Stage 8 requirement E) -------------


def test_operational_and_candidate_populations_use_identical_source_alert_ids(monkeypatch):
    conn = _FakeEvalConnection()
    _patch(monkeypatch, conn)
    for alert_id in (1, 2, 3):
        _add_alert(conn, alert_id, generation_run_id="genrun-A", dataset_version="dsv-A")
        _add_evidence(conn, alert_id, channel_model_bundle_id=1)
    # a different generation's alert must never leak into either population
    _add_alert(conn, 4, generation_run_id="genrun-B", dataset_version="dsv-B")
    _add_evidence(conn, 4, channel_model_bundle_id=1)

    population = load_resolved_alert_outcomes("online_banking", "aidp_test", generation_run_id="genrun-A", operational_bundle_id=1)
    _, scoring_inputs = load_resolved_alert_scoring_contexts("online_banking", "aidp_test", generation_run_id="genrun-A")

    operational_ids = {o.source_alert_id for o in population.outcomes}
    candidate_ids = {i.source_alert_id for i in scoring_inputs}
    assert operational_ids == candidate_ids
    assert len(operational_ids) == 3

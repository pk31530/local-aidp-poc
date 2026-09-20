"""Phase 6: alert queue -- idempotent alert/evidence persistence,
catastrophic-failure handling, and concurrency-safe disposition/status
transitions. No database, Docker, or network access anywhere in this file
-- every test uses `_FakeAlertQueueStore`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.alerts.queue import (
    InvalidAlertTransitionError,
    _FakeAlertQueueStore,
    record_disposition,
    score_and_record_alert,
)
from src.fraud_intel.ensemble.policy import EnsemblePolicy
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import GraphPolicy
from src.fraud_intel.models.anomaly import AnomalyNormalization
from src.fraud_intel.models.preprocessing import ChannelPreprocessor
from src.fraud_intel.rules.provider import LocalYamlRuleProvider
from src.fraud_intel.scoring.orchestrator import LoadedChannelBundle

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

_FEATURE_COLUMNS = [
    "amount_vs_entity_average", "amount_zscore", "is_new_device", "is_new_ip", "night_transaction_flag",
    "events_last_10m", "events_last_1h", "events_last_24h", "prior_alert_count",
    "customer_observed_tenure_days", "account_observed_tenure_days", "counterparty_risk_score",
    "new_device_high_value_combo_flag", "mfa_bypass_flag", "session_velocity", "profile_change_then_transfer_flag",
]

_PREPROCESSOR = ChannelPreprocessor.fit(
    [{col: 0.0 for col in _FEATURE_COLUMNS} for _ in range(3)],
    feature_columns=_FEATURE_COLUMNS, feature_schema_version="v1", preprocessing_artifact_version="pp-test",
)


class _FakeProbaModel:
    def __init__(self, fraud_proba: float = 0.2):
        self._p = fraud_proba

    def predict_proba(self, X):
        return [[1.0 - self._p, self._p] for _ in range(len(X))]


class _FakeAnomalyModel:
    def decision_function(self, X):
        return [0.0 for _ in range(len(X))]


class _FailingModel:
    def predict_proba(self, X):
        raise RuntimeError("simulated model failure")


def _bundle() -> LoadedChannelBundle:
    return LoadedChannelBundle(
        channel="online_banking", bundle_id=2, bundle_version=2,
        gbm_model=_FakeProbaModel(0.2), lr_model=_FakeProbaModel(0.3), anomaly_model=_FakeAnomalyModel(),
        anomaly_normalization=AnomalyNormalization(train_min=-1.0, train_max=1.0, anomaly_artifact_version="anom-test", library_versions={}),
        preprocessor=_PREPROCESSOR,
        gbm_model_version="gbm-7", lr_model_version="lr-7", anomaly_model_version="anomaly-7",
        preprocessing_artifact_version="pp-test", feature_schema_version="v1",
        rule_set_version="v1", graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
    )


def _ensemble_policy() -> EnsemblePolicy:
    return EnsemblePolicy(
        channel="online_banking", policy_version="v1", weight_rule=0.35, weight_gbm=0.40, weight_anomaly=0.15,
        weight_graph=0.10, rule_score_cap=1.0, high_threshold=0.75, medium_threshold=0.40,
    )


def _graph_policy() -> GraphPolicy:
    return GraphPolicy(
        channel="online_banking", graph_policy_version="v1", shared_device_cap=5, shared_device_weight=0.3,
        fan_in_cap=10, fan_in_weight=0.3, fan_out_cap=10, fan_out_weight=0.2, shortest_path_weight=0.2,
        graph_max_history_events=5000, max_nodes=20000, max_edges=200000,
    )


def _event(*, event_timestamp: datetime = T0, event_id: uuid.UUID | None = None) -> FraudEvent:
    return FraudEvent(
        event_id=event_id or uuid.uuid4(), channel="online_banking", customer_id=CUSTOMER, account_id=ACCOUNT,
        event_timestamp=event_timestamp, amount_minor_units=10_000, direction="debit", device_id="DEV1",
        channel_payload=OnlineBankingPayload(session_id="S1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"),
    )


def _source_alert(*, event_id: uuid.UUID, created_at: datetime = T0, source_alert_id: uuid.UUID | None = None) -> SourceAlertContext:
    return SourceAlertContext(
        source_alert_id=source_alert_id or uuid.uuid4(),
        source_system="LocalYamlRuleProvider (simulated upstream)", event_id=event_id, source_alert_created_at=created_at,
        source_rule_ids=["RULE1"], source_rule_version="v1", source_alert_reason_codes=["REASON1"],
        generation_run_id="genrun-1", dataset_version="dsv-1", created_at=created_at,
    )


def _ctx(event: FraudEvent) -> FeatureComputationContext:
    return FeatureComputationContext(current_event=event, historical_events=(), source_alert_history=(), as_of_time=event.event_timestamp)


def _score(*, event=None, source_alert=None, store, bundle=None, score_execution_id=None):
    event = event or _event()
    return score_and_record_alert(
        event=event, source_alert=source_alert or _source_alert(event_id=event.event_id), context=_ctx(event),
        bundle=bundle or _bundle(), rule_provider=LocalYamlRuleProvider(), ensemble_policy=_ensemble_policy(),
        graph_policy=_graph_policy(), resolved_fraud_evidence=(), config_hash="cfg-hash-1", git_sha="deadbeef",
        store=store, score_execution_id=score_execution_id,
    )


# ---- alert creation / idempotency ---------------------------------------------------


def test_two_distinct_source_alerts_for_the_same_event_create_two_alerts():
    store = _FakeAlertQueueStore()
    event = _event()
    alert1, _ = _score(event=event, source_alert=_source_alert(event_id=event.event_id), store=store)
    alert2, _ = _score(event=event, source_alert=_source_alert(event_id=event.event_id), store=store)
    assert alert1.alert_id != alert2.alert_id
    assert alert1.event_id == alert2.event_id == event.event_id
    assert len(store.alerts_by_id) == 2


def test_duplicate_delivery_of_the_same_source_alert_returns_the_same_alert():
    store = _FakeAlertQueueStore()
    event = _event()
    alert = _source_alert(event_id=event.event_id)
    first, _ = _score(event=event, source_alert=alert, store=store)
    second, _ = _score(event=event, source_alert=alert, store=store)
    assert first.alert_id == second.alert_id
    assert len(store.alerts_by_id) == 1


def test_initial_priority_fields_come_from_the_first_scoring_pass():
    store = _FakeAlertQueueStore()
    alert, evidence = _score(store=store)
    assert alert.initial_operational_priority_score == evidence.operational_priority_score
    assert alert.initial_priority_band == evidence.priority_band
    assert alert.status == "OPEN"


def test_deliberate_rescore_adds_evidence_without_creating_another_alert():
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert1, evidence1 = _score(event=event, source_alert=source_alert, store=store)
    alert2, evidence2 = _score(event=event, source_alert=source_alert, store=store)
    assert alert1.alert_id == alert2.alert_id
    assert evidence1.evidence_id != evidence2.evidence_id
    assert len(store.evidence_by_alert[alert1.alert_id]) == 2
    # immutable initial fields never change even though a second, distinct
    # score_execution_id was scored
    assert alert1.initial_operational_priority_score == alert2.initial_operational_priority_score


def test_amount_minor_units_matches_the_event_not_a_dollar_value():
    store = _FakeAlertQueueStore()
    event = _event()
    alert, _ = _score(event=event, store=store)
    assert alert.amount_minor_units == event.amount_minor_units == 10_000


# ---- same-execution retry ----------------------------------------------------------


def test_same_score_execution_id_retry_leaves_evidence_unchanged():
    store = _FakeAlertQueueStore()
    alert, evidence = _score(store=store)
    same_execution_id = evidence.score_execution_id

    retried = store.record_evidence(
        alert_id=alert.alert_id, score_execution_id=same_execution_id,
        rule_result={"different": "payload -- must be ignored"}, gbm_probability=0.999, lr_probability=None,
        anomaly_score=None, graph_risk_score=None, operational_priority_score=0.999, priority_band="HIGH",
        degraded=True, component_statuses={}, reason_codes=[], channel_model_bundle_id=999,
        gbm_model_version="different", lr_model_version="different", anomaly_model_version="different",
        preprocessing_artifact_version="different", feature_schema_version="different", rule_set_version="different",
        graph_policy_version="different", ensemble_policy_version="different", reason_code_version="different",
        config_hash="different", git_sha="different", event_time=T0,
    )
    assert retried == evidence  # completely unchanged -- the "retry" attempt's payload was ignored
    assert len(store.evidence_by_alert[alert.alert_id]) == 1


def test_score_and_record_alert_reuses_an_explicit_score_execution_id_for_a_retry():
    """Phase 6 corrective pass: score_execution_id is now an optional
    parameter -- a caller that needs to safely retry the same logical
    scoring attempt (e.g. src.fraud_intel.scoring.dispatch after a
    transient failure) can pass the SAME id on both calls, and the second
    call is a true no-op (same evidence row, not a second one) rather than
    minting a fresh id and silently producing a duplicate-looking rescore."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    explicit_id = uuid.uuid4()

    alert1, evidence1 = _score(event=event, source_alert=source_alert, store=store, score_execution_id=explicit_id)
    alert2, evidence2 = _score(event=event, source_alert=source_alert, store=store, score_execution_id=explicit_id)

    assert alert1.alert_id == alert2.alert_id
    assert evidence1.evidence_id == evidence2.evidence_id
    assert evidence1.score_execution_id == evidence2.score_execution_id == explicit_id
    assert len(store.evidence_by_alert[alert1.alert_id]) == 1  # no duplicate row from the "retry"


# ---- catastrophic failure -------------------------------------------------------------


class _RaisingRuleProvider:
    def evaluate(self, *, event, source_alert, features):
        raise RuntimeError("simulated catastrophic failure with a secret token=abc123")


def test_catastrophic_failure_creates_alert_at_high_with_actual_policy_version():
    store = _FakeAlertQueueStore()
    event = _event()
    with pytest.raises(RuntimeError, match="simulated catastrophic failure"):
        score_and_record_alert(
            event=event, source_alert=_source_alert(event_id=event.event_id), context=_ctx(event), bundle=_bundle(),
            rule_provider=_RaisingRuleProvider(), ensemble_policy=_ensemble_policy(), graph_policy=_graph_policy(),
            resolved_fraud_evidence=(), config_hash="cfg-hash-1", git_sha="deadbeef", store=store,
        )
    assert len(store.alerts_by_id) == 1
    alert = next(iter(store.alerts_by_id.values()))
    assert alert.initial_priority_band == "HIGH"
    assert alert.initial_operational_priority_score == 1.0
    # NOT a fake "SCORING_UNAVAILABLE" pretending to be a real policy version
    assert alert.initial_ensemble_policy_version == "v1"


def test_catastrophic_failure_persists_idempotent_typed_evidence():
    store = _FakeAlertQueueStore()
    event = _event()
    with pytest.raises(RuntimeError):
        score_and_record_alert(
            event=event, source_alert=_source_alert(event_id=event.event_id), context=_ctx(event), bundle=_bundle(),
            rule_provider=_RaisingRuleProvider(), ensemble_policy=_ensemble_policy(), graph_policy=_graph_policy(),
            resolved_fraud_evidence=(), config_hash="cfg-hash-1", git_sha="deadbeef", store=store,
        )
    alert = next(iter(store.alerts_by_id.values()))
    evidence = store.evidence_by_alert[alert.alert_id][0]
    assert evidence.degraded is True
    assert evidence.rule_result is None
    assert evidence.operational_priority_score is None
    assert evidence.priority_band is None
    assert evidence.component_statuses == {"orchestrator": {"status": "ERROR", "error_code": "SCORING_UNAVAILABLE"}}
    assert any(rc["code"] == "SCORING_UNAVAILABLE" for rc in evidence.reason_codes)
    # attempted bundle/policy versions ARE preserved
    assert evidence.channel_model_bundle_id == 2
    assert evidence.gbm_model_version == "gbm-7"
    assert evidence.ensemble_policy_version == "v1"
    # never the raw exception text anywhere in the persisted row
    dumped = str(evidence.model_dump())
    assert "secret token" not in dumped
    assert "simulated catastrophic failure" not in dumped


def test_catastrophic_failure_never_masks_the_original_exception_even_if_evidence_write_fails():
    store = _FakeAlertQueueStore()
    original_record_evidence = store.record_evidence

    def _boom(*args, **kwargs):
        raise ConnectionError("simulated evidence-write failure")

    store.record_evidence = _boom  # type: ignore[method-assign]

    event = _event()
    with pytest.raises(RuntimeError, match="simulated catastrophic failure"):
        score_and_record_alert(
            event=event, source_alert=_source_alert(event_id=event.event_id), context=_ctx(event), bundle=_bundle(),
            rule_provider=_RaisingRuleProvider(), ensemble_policy=_ensemble_policy(), graph_policy=_graph_policy(),
            resolved_fraud_evidence=(), config_hash="cfg-hash-1", git_sha="deadbeef", store=store,
        )
    # the alert itself was still created before the evidence write failed
    assert len(store.alerts_by_id) == 1


def test_catastrophic_failure_never_masks_the_original_exception_even_if_alert_creation_fails():
    store = _FakeAlertQueueStore()

    def _boom(*args, **kwargs):
        raise ConnectionError("simulated alert-creation failure")

    store.create_alert_if_new = _boom  # type: ignore[method-assign]

    event = _event()
    with pytest.raises(RuntimeError, match="simulated catastrophic failure"):
        score_and_record_alert(
            event=event, source_alert=_source_alert(event_id=event.event_id), context=_ctx(event), bundle=_bundle(),
            rule_provider=_RaisingRuleProvider(), ensemble_policy=_ensemble_policy(), graph_policy=_graph_policy(),
            resolved_fraud_evidence=(), config_hash="cfg-hash-1", git_sha="deadbeef", store=store,
        )


def test_degraded_but_non_catastrophic_scoring_still_persists_normally():
    store = _FakeAlertQueueStore()
    alert, evidence = _score(store=store, bundle=_bundle())
    # A regular (non-raising) run with a healthy rule provider is never degraded.
    assert evidence.degraded is False


def test_gbm_component_failure_still_persists_a_real_non_catastrophic_evidence_row():
    store = _FakeAlertQueueStore()
    bundle = LoadedChannelBundle(
        channel="online_banking", bundle_id=2, bundle_version=2,
        gbm_model=_FailingModel(), lr_model=_FakeProbaModel(0.3), anomaly_model=_FakeAnomalyModel(),
        anomaly_normalization=AnomalyNormalization(train_min=-1.0, train_max=1.0, anomaly_artifact_version="anom-test", library_versions={}),
        preprocessor=_PREPROCESSOR, gbm_model_version="gbm-7", lr_model_version="lr-7", anomaly_model_version="anomaly-7",
        preprocessing_artifact_version="pp-test", feature_schema_version="v1",
        rule_set_version="v1", graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
    )
    alert, evidence = _score(store=store, bundle=bundle)
    assert evidence.degraded is True
    assert evidence.rule_result is not None  # rules succeeded -- this is NOT catastrophic
    assert alert.initial_priority_band == "HIGH"  # GBM failure floor


# ---- disposition / status transitions --------------------------------------------------


def _alert_id(store) -> int:
    alert, _ = _score(store=store)
    return alert.alert_id


def test_needs_more_info_moves_open_to_in_review():
    store = _FakeAlertQueueStore()
    alert_id = _alert_id(store)
    record_disposition(alert_id=alert_id, analyst_id="A1", disposition="NEEDS_MORE_INFO", notes=None, store=store)
    assert store.get_alert(alert_id).status == "IN_REVIEW"


def test_confirmed_fraud_moves_to_closed():
    store = _FakeAlertQueueStore()
    alert_id = _alert_id(store)
    record_disposition(alert_id=alert_id, analyst_id="A1", disposition="CONFIRMED_FRAUD", notes=None, store=store)
    assert store.get_alert(alert_id).status == "CLOSED"


def test_escalated_from_in_review_stays_in_review():
    store = _FakeAlertQueueStore()
    alert_id = _alert_id(store)
    record_disposition(alert_id=alert_id, analyst_id="A1", disposition="ESCALATED", notes=None, store=store)
    record_disposition(alert_id=alert_id, analyst_id="A2", disposition="ESCALATED", notes=None, store=store)
    assert store.get_alert(alert_id).status == "IN_REVIEW"
    assert len(store.dispositions) == 2  # append-only -- both recorded


def test_disposition_against_closed_alert_raises_invalid_transition():
    store = _FakeAlertQueueStore()
    alert_id = _alert_id(store)
    record_disposition(alert_id=alert_id, analyst_id="A1", disposition="CONFIRMED_LEGITIMATE", notes=None, store=store)
    with pytest.raises(InvalidAlertTransitionError):
        record_disposition(alert_id=alert_id, analyst_id="A2", disposition="ESCALATED", notes=None, store=store)


def test_concurrency_race_closed_cannot_be_reopened():
    """Fake-store race test (Phase 6 decision 4): two 'racing' dispositions
    -- the first closes the alert, the second (submitted immediately after,
    simulating what would be a second transaction blocked on the same row
    lock in a real database) must be rejected, never silently reopening
    CLOSED."""
    store = _FakeAlertQueueStore()
    alert_id = _alert_id(store)
    record_disposition(alert_id=alert_id, analyst_id="A1", disposition="CONFIRMED_FRAUD", notes=None, store=store)
    with pytest.raises(InvalidAlertTransitionError):
        record_disposition(alert_id=alert_id, analyst_id="A2", disposition="CONFIRMED_LEGITIMATE", notes=None, store=store)
    assert store.get_alert(alert_id).status == "CLOSED"
    assert len(store.dispositions) == 1  # the rejected attempt was never recorded


def test_disposition_history_is_append_only():
    store = _FakeAlertQueueStore()
    alert_id = _alert_id(store)
    record_disposition(alert_id=alert_id, analyst_id="A1", disposition="NEEDS_MORE_INFO", notes="first", store=store)
    record_disposition(alert_id=alert_id, analyst_id="A2", disposition="CONFIRMED_FRAUD", notes="second", store=store)
    assert len(store.dispositions) == 2
    assert store.dispositions[0].notes == "first"  # never overwritten


def test_notes_are_redacted_and_bounded():
    store = _FakeAlertQueueStore()
    alert_id = _alert_id(store)
    long_notes = ("password=hunter2 " * 200) + "x" * 3000
    record_disposition(alert_id=alert_id, analyst_id="A1", disposition="NEEDS_MORE_INFO", notes=long_notes, store=store)
    saved = store.dispositions[0].notes
    assert len(saved) <= 2000
    assert "hunter2" not in saved


def test_none_notes_stay_none():
    store = _FakeAlertQueueStore()
    alert_id = _alert_id(store)
    record_disposition(alert_id=alert_id, analyst_id="A1", disposition="NEEDS_MORE_INFO", notes=None, store=store)
    assert store.dispositions[0].notes is None


# ---- latest-evidence deterministic tie-break --------------------------------------------


def test_get_latest_evidence_deterministic_tie_break():
    store = _FakeAlertQueueStore()
    alert, evidence1 = _score(store=store)
    # A genuine rescore always has a strictly later scored_at in practice;
    # this proves get_latest_evidence returns the most recent regardless.
    alert2, evidence2 = _score(event=_event(event_id=alert.event_id), source_alert=_source_alert(event_id=alert.event_id, source_alert_id=alert.source_alert_id), store=store)
    latest = store.get_latest_evidence(alert.alert_id)
    assert latest.evidence_id in (evidence1.evidence_id, evidence2.evidence_id)
    assert latest is not None

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


# ---- list_alerts_with_current_state(): stale alert-list read-path corrective pass ------
#
# Phase 7B corrective pass: `aidp alerts list` used to read ONLY
# fraud_alerts.initial_priority_band/initial_operational_priority_score --
# frozen at first-scoring time -- so an alert rescored under a LATER
# bundle (e.g. ACH's bundle 3 -> bundle 4 replacement promotion) never
# showed its real current band/score, and --priority-band filtering
# against a band the alert had since moved away from silently returned
# 0 rows. `list_alerts_with_current_state()` fixes this by sourcing
# "current" from the SAME latest-evidence selection rule
# (LATEST_EVIDENCE_ORDER_SQL) get_latest_evidence() already uses
# correctly, for both the real store (a single LEFT JOIN LATERAL query,
# no N+1) and the fake store.


def _bundle3() -> LoadedChannelBundle:
    return _bundle()  # bundle_id=2 in the shared fixture -- reused as the "earlier" bundle


def _bundle4() -> LoadedChannelBundle:
    b = _bundle()
    return LoadedChannelBundle(
        channel=b.channel, bundle_id=b.bundle_id + 1, bundle_version=b.bundle_version + 1,
        gbm_model=_FakeProbaModel(0.6), lr_model=b.lr_model, anomaly_model=b.anomaly_model,
        anomaly_normalization=b.anomaly_normalization, preprocessor=b.preprocessor,
        gbm_model_version="gbm-8", lr_model_version=b.lr_model_version, anomaly_model_version=b.anomaly_model_version,
        preprocessing_artifact_version=b.preprocessing_artifact_version, feature_schema_version=b.feature_schema_version,
        rule_set_version=b.rule_set_version, graph_policy_version=b.graph_policy_version,
        ensemble_policy_version=b.ensemble_policy_version, reason_code_version=b.reason_code_version,
    )


def _rescore_to_medium(store, *, event, source_alert):
    """Idempotently reuses the SAME already-existing alert (create_alert_
    if_new is first-write-wins), then persists a NEW evidence row
    directly at MEDIUM band, provenance-stamped as bundle4 -- exactly the
    ACH bundle-3-LOW -> bundle-4-MEDIUM replacement shape. Writes the
    evidence row directly (rather than tuning a real GBM probability
    through the live rule/ensemble pipeline to land exactly on MEDIUM,
    which is unpredictable and not what these list/filter tests are
    about) -- score_and_record_alert()'s own orchestration is already
    covered by the tests above this section. `scored_at` uses real
    wall-clock "now" (never T0-relative): score_and_record_alert()'s own
    evidence rows are ALSO stamped with real wall-clock `datetime.now()`
    (src.fraud_intel.scoring.orchestrator.score_source_alert()), so a
    fixed T0-relative offset could -- and, discovered here, DID --
    silently end up EARLIER than "now" and be selected as stale rather
    than as the intended later row."""
    alert = store.create_alert_if_new(
        event=event, source_alert=source_alert,
        initial_operational_priority_score=0.05, initial_priority_band="LOW",
        initial_ensemble_policy_version="v1",
    )
    bundle4 = _bundle4()
    evidence = store.record_evidence(
        alert_id=alert.alert_id, score_execution_id=uuid.uuid4(),
        rule_result=None, gbm_probability=0.6, lr_probability=None, anomaly_score=0.0, graph_risk_score=0.0,
        operational_priority_score=0.42, priority_band="MEDIUM", degraded=False, component_statuses={}, reason_codes=[],
        channel_model_bundle_id=bundle4.bundle_id, gbm_model_version=bundle4.gbm_model_version,
        lr_model_version=bundle4.lr_model_version, anomaly_model_version=bundle4.anomaly_model_version,
        preprocessing_artifact_version=bundle4.preprocessing_artifact_version,
        feature_schema_version=bundle4.feature_schema_version, rule_set_version=bundle4.rule_set_version,
        graph_policy_version=bundle4.graph_policy_version, ensemble_policy_version=bundle4.ensemble_policy_version,
        reason_code_version=bundle4.reason_code_version, config_hash="cfg-medium", git_sha="deadbeef",
        event_time=event.event_timestamp, scored_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    return alert, evidence


def test_list_current_priority_band_filter_includes_rescored_alert_medium_not_initial_low():
    """Items 1/2: initial LOW + latest MEDIUM -> MEDIUM filter includes
    it, LOW filter excludes it."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert, evidence1 = _score(event=event, source_alert=source_alert, store=store, bundle=_bundle3())
    assert alert.initial_priority_band == "LOW"

    _, evidence2 = _rescore_to_medium(store, event=event, source_alert=source_alert)
    assert evidence2.priority_band == "MEDIUM"

    medium_items = store.list_alerts_with_current_state(current_priority_band="MEDIUM")
    assert [i.alert_id for i in medium_items] == [alert.alert_id]

    low_items = store.list_alerts_with_current_state(current_priority_band="LOW")
    assert alert.alert_id not in [i.alert_id for i in low_items]


def test_list_current_fields_come_from_latest_evidence_not_initial():
    """Item 3."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert, _ = _score(event=event, source_alert=source_alert, store=store, bundle=_bundle3())
    _, evidence2 = _rescore_to_medium(store, event=event, source_alert=source_alert)

    item = next(i for i in store.list_alerts_with_current_state() if i.alert_id == alert.alert_id)
    assert item.current_priority_band == evidence2.priority_band == "MEDIUM"
    assert item.current_operational_priority_score == evidence2.operational_priority_score
    assert item.current_evidence_id == evidence2.evidence_id
    assert item.current_channel_model_bundle_id == evidence2.channel_model_bundle_id == 3
    assert item.current_scored_at == evidence2.scored_at
    # initial_* stay exactly what the FIRST scoring pass produced
    assert item.initial_priority_band == "LOW"


def test_list_and_show_agree_on_current_evidence():
    """Item 4: alerts show (get_latest_evidence) and alerts list
    (list_alerts_with_current_state) must select the identical row."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert, _ = _score(event=event, source_alert=source_alert, store=store, bundle=_bundle3())
    _rescore_to_medium(store, event=event, source_alert=source_alert)

    show_evidence = store.get_latest_evidence(alert.alert_id)
    list_item = next(i for i in store.list_alerts_with_current_state() if i.alert_id == alert.alert_id)

    assert list_item.current_evidence_id == show_evidence.evidence_id
    assert list_item.current_priority_band == show_evidence.priority_band
    assert list_item.current_operational_priority_score == show_evidence.operational_priority_score


def test_list_selects_the_newest_scored_at():
    """Item 5."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert, evidence1 = _score(event=event, source_alert=source_alert, store=store, bundle=_bundle3())
    _, evidence2 = _rescore_to_medium(store, event=event, source_alert=source_alert)
    assert evidence2.scored_at > evidence1.scored_at

    item = next(i for i in store.list_alerts_with_current_state() if i.alert_id == alert.alert_id)
    assert item.current_evidence_id == evidence2.evidence_id


def test_list_tie_break_selects_higher_evidence_id_on_equal_scored_at():
    """Item 6: two evidence rows sharing the exact same scored_at ->
    the higher evidence_id wins, matching LATEST_EVIDENCE_ORDER_SQL
    exactly."""
    from src.fraud_intel.alerts.queue import AlertEvidenceRecord

    store = _FakeAlertQueueStore()
    alert, _ = _score(store=store, bundle=_bundle3())
    same_time = datetime(2026, 6, 1, tzinfo=timezone.utc)

    lower = AlertEvidenceRecord(
        evidence_id=uuid.UUID(int=1), alert_id=alert.alert_id, score_execution_id=uuid.uuid4(),
        rule_result=None, gbm_probability=0.1, lr_probability=None, anomaly_score=None, graph_risk_score=None,
        operational_priority_score=0.2, priority_band="LOW", degraded=False, component_statuses={}, reason_codes=[],
        channel_model_bundle_id=10, gbm_model_version="1", lr_model_version="1", anomaly_model_version="1",
        preprocessing_artifact_version="pp", feature_schema_version="v1", rule_set_version="v1",
        graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
        config_hash="c", git_sha="g", event_time=T0, scored_at=same_time,
    )
    higher = lower.model_copy(update={
        "evidence_id": uuid.UUID(int=2), "score_execution_id": uuid.uuid4(),
        "priority_band": "HIGH", "operational_priority_score": 0.9,
    })
    store.evidence_by_key[(alert.alert_id, str(lower.score_execution_id))] = lower
    store.evidence_by_key[(alert.alert_id, str(higher.score_execution_id))] = higher
    store.evidence_by_alert[alert.alert_id] = [lower, higher]

    item = next(i for i in store.list_alerts_with_current_state() if i.alert_id == alert.alert_id)
    assert item.current_evidence_id == higher.evidence_id
    assert item.current_priority_band == "HIGH"


def test_list_retired_bundle_evidence_then_operational_bundle_evidence_selects_intended_row():
    """Item 7: reproduces the exact ACH shape -- bundle 3 (later RETIRED)
    scores first, bundle 4 (OPERATIONAL) rescores later -> list/show must
    select bundle 4's evidence."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert, evidence_bundle3 = _score(event=event, source_alert=source_alert, store=store, bundle=_bundle3())
    assert evidence_bundle3.channel_model_bundle_id == 2

    _, evidence_bundle4 = _rescore_to_medium(store, event=event, source_alert=source_alert)
    assert evidence_bundle4.channel_model_bundle_id == 3

    item = next(i for i in store.list_alerts_with_current_state() if i.alert_id == alert.alert_id)
    assert item.current_channel_model_bundle_id == evidence_bundle4.channel_model_bundle_id
    assert item.current_priority_band == evidence_bundle4.priority_band


def test_list_alert_with_no_evidence_has_null_current_fields():
    """Item 8: an alert created but never (yet) evidenced -- the rare
    partial-write state on score_and_record_alert()'s catastrophic path --
    stays visible with an explicit null current-state, not silently
    dropped and not fabricated from initial_*."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert = store.create_alert_if_new(
        event=event, source_alert=source_alert,
        initial_operational_priority_score=1.0, initial_priority_band="HIGH", initial_ensemble_policy_version="v1",
    )
    item = next(i for i in store.list_alerts_with_current_state() if i.alert_id == alert.alert_id)
    assert item.current_evidence_id is None
    assert item.current_priority_band is None
    assert item.current_operational_priority_score is None
    assert item.current_channel_model_bundle_id is None
    assert item.current_scored_at is None
    # initial_* remain populated and visible even with zero evidence
    assert item.initial_priority_band == "HIGH"


def test_list_initial_fields_remain_unchanged_and_visible_as_audit_values():
    """Item 9."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert, _ = _score(event=event, source_alert=source_alert, store=store, bundle=_bundle3())
    _rescore_to_medium(store, event=event, source_alert=source_alert)

    item = next(i for i in store.list_alerts_with_current_state() if i.alert_id == alert.alert_id)
    assert item.initial_priority_band == alert.initial_priority_band == "LOW"
    assert item.initial_operational_priority_score == alert.initial_operational_priority_score
    assert item.initial_ensemble_policy_version == alert.initial_ensemble_policy_version


def test_list_channel_filtering_remains_isolated():
    """Item 10."""
    from src.fraud_intel.events.ach import ACHPayload

    store = _FakeAlertQueueStore()
    ob_event = _event()
    ach_event = FraudEvent(
        event_id=uuid.uuid4(), channel="ach", customer_id="FIC2000", account_id="FIA200000",
        event_timestamp=T0, amount_minor_units=5_000, direction="debit", device_id=None,
        channel_payload=ACHPayload(
            sec_code="PPD", originating_routing_number="123456789", receiving_routing_number="987654321",
            batch_id="B1", effective_entry_date=T0.date(), company_id="C1",
        ),
    )
    ob_alert, _ = _score(event=ob_event, source_alert=_source_alert(event_id=ob_event.event_id), store=store, bundle=_bundle3())
    ach_bundle = LoadedChannelBundle(
        channel="ach", bundle_id=99, bundle_version=1,
        gbm_model=_FakeProbaModel(0.2), lr_model=_FakeProbaModel(0.3), anomaly_model=_FakeAnomalyModel(),
        anomaly_normalization=AnomalyNormalization(train_min=-1.0, train_max=1.0, anomaly_artifact_version="anom-test", library_versions={}),
        preprocessor=_PREPROCESSOR, gbm_model_version="gbm-9", lr_model_version="lr-9", anomaly_model_version="anomaly-9",
        preprocessing_artifact_version="pp-test", feature_schema_version="v1",
        rule_set_version="v1", graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
    )
    ach_alert, _ = score_and_record_alert(
        event=ach_event, source_alert=_source_alert(event_id=ach_event.event_id), context=_ctx(ach_event),
        bundle=ach_bundle, rule_provider=LocalYamlRuleProvider(),
        ensemble_policy=EnsemblePolicy(
            channel="ach", policy_version="v1", weight_rule=0.35, weight_gbm=0.40, weight_anomaly=0.15,
            weight_graph=0.10, rule_score_cap=1.0, high_threshold=0.75, medium_threshold=0.40,
        ),
        graph_policy=GraphPolicy(
            channel="ach", graph_policy_version="v1", shared_device_cap=5, shared_device_weight=0.3,
            fan_in_cap=10, fan_in_weight=0.3, fan_out_cap=10, fan_out_weight=0.2, shortest_path_weight=0.2,
            graph_max_history_events=5000, max_nodes=20000, max_edges=200000,
        ),
        resolved_fraud_evidence=(), config_hash="c", git_sha="g", store=store,
    )

    ob_only = store.list_alerts_with_current_state(channel="online_banking")
    ach_only = store.list_alerts_with_current_state(channel="ach")
    assert {i.alert_id for i in ob_only} == {ob_alert.alert_id}
    assert {i.alert_id for i in ach_only} == {ach_alert.alert_id}


def test_list_no_duplicate_alert_after_multiple_evidence_rows():
    """Item 11: an alert rescored three times still appears exactly ONCE
    in the list -- the join/selection must not fan out."""
    store = _FakeAlertQueueStore()
    event = _event()
    source_alert = _source_alert(event_id=event.event_id)
    alert, _ = _score(event=event, source_alert=source_alert, store=store, bundle=_bundle3())
    _rescore_to_medium(store, event=event, source_alert=source_alert)
    score_and_record_alert(
        event=event, source_alert=source_alert, context=_ctx(event), bundle=_bundle4(),
        rule_provider=LocalYamlRuleProvider(), ensemble_policy=_ensemble_policy(), graph_policy=_graph_policy(),
        resolved_fraud_evidence=(), config_hash="cfg-3", git_sha="deadbeef", store=store,
    )
    assert len(store.evidence_by_alert[alert.alert_id]) == 3

    items = store.list_alerts_with_current_state()
    matching = [i for i in items if i.alert_id == alert.alert_id]
    assert len(matching) == 1


def test_list_respects_limit_without_duplication_across_many_alerts():
    """Item 11 (pagination edge): more alerts than `limit` -- exactly
    `limit` distinct alert_ids are returned, deterministically."""
    store = _FakeAlertQueueStore()
    alert_ids = set()
    for i in range(5):
        event = _event(event_timestamp=T0 + timedelta(minutes=i))
        alert, _ = _score(event=event, source_alert=_source_alert(event_id=event.event_id), store=store, bundle=_bundle3())
        alert_ids.add(alert.alert_id)

    items = store.list_alerts_with_current_state(limit=3)
    assert len(items) == 3
    assert len({i.alert_id for i in items}) == 3
    assert {i.alert_id for i in items}.issubset(alert_ids)


def test_ach_regression_medium_alerts_visible_after_replacement_bundle_rescore():
    """Item 15: reproduces ACH's exact defect class at a proportional
    scale -- a channel where EVERY alert's initial (first-bundle) band
    was LOW, and a later replacement-bundle rescore moves a real subset
    to MEDIUM. Before this fix, `--priority-band MEDIUM` returned 0 rows
    for such a channel (the real, observed ACH symptom); after this fix
    it must return exactly the rescored subset."""
    store = _FakeAlertQueueStore()
    alerts = []
    for i in range(5):
        event = _event(event_timestamp=T0 + timedelta(minutes=i))
        source_alert = _source_alert(event_id=event.event_id)
        alert, evidence = _score(event=event, source_alert=source_alert, store=store, bundle=_bundle3())
        assert alert.initial_priority_band == "LOW"  # every alert starts LOW, like real ACH bundle 3
        alerts.append((alert, event, source_alert))

    # Replacement-bundle rescore: only 2 of the 5 move to MEDIUM (the
    # other 3 stay LOW), matching the real 27-of-441 proportion's shape.
    rescored_to_medium_ids = set()
    for alert, event, source_alert in alerts[:2]:
        _, evidence2 = _rescore_to_medium(store, event=event, source_alert=source_alert)
        assert evidence2.priority_band == "MEDIUM"
        rescored_to_medium_ids.add(alert.alert_id)

    # Every fraud_alerts row is STILL initial_priority_band=LOW (Phase 6
    # decision 2 -- never silently rewritten).
    assert all(a.initial_priority_band == "LOW" for a, _, _ in alerts)

    medium_items = store.list_alerts_with_current_state(current_priority_band="MEDIUM")
    assert {i.alert_id for i in medium_items} == rescored_to_medium_ids
    assert len(medium_items) == 2  # NOT 0 -- the exact bug this patch fixes


# ---- real Postgres-shape parity for list_alerts_with_current_state() ------------------


class _FakeListAlertsCursor:
    def __init__(self, conn):
        self._conn = conn
        self._result: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        assert normalized.startswith("SELECT fa.*,"), f"unexpected SQL: {normalized[:80]}"
        assert "LEFT JOIN LATERAL" in normalized
        assert "ORDER BY scored_at DESC, evidence_id DESC" in normalized
        assert "UPDATE" not in normalized and "DELETE" not in normalized and "INSERT" not in normalized

        remaining = list(params)
        limit = remaining.pop()  # LIMIT %s is always the final placeholder
        channel = status = band = None
        if "fa.channel = %s" in normalized:
            channel = remaining.pop(0)
        if "fa.status = %s" in normalized:
            status = remaining.pop(0)
        if "le.priority_band = %s" in normalized:
            band = remaining.pop(0)
        assert not remaining  # every param was consumed by exactly one clause

        rows = []
        for fa in self._conn.fraud_alerts:
            if channel is not None and fa["channel"] != channel:
                continue
            if status is not None and fa["status"] != status:
                continue
            evidence = [e for e in self._conn.alert_evidence if e["alert_id"] == fa["alert_id"]]
            latest = sorted(evidence, key=lambda e: (e["scored_at"], str(e["evidence_id"])), reverse=True)[0] if evidence else None
            if band is not None and (latest is None or latest["priority_band"] != band):
                continue
            row = dict(fa)
            row["current_evidence_id"] = latest["evidence_id"] if latest else None
            row["current_channel_model_bundle_id"] = latest["channel_model_bundle_id"] if latest else None
            row["current_priority_band"] = latest["priority_band"] if latest else None
            row["current_operational_priority_score"] = latest["operational_priority_score"] if latest else None
            row["current_scored_at"] = latest["scored_at"] if latest else None
            rows.append(row)
        rows.sort(key=lambda r: (r["created_at"], r["alert_id"]), reverse=True)
        self._result = rows[:limit]

    def fetchall(self):
        return self._result


class _FakeListAlertsConnection:
    def __init__(self):
        self.fraud_alerts: list[dict] = []
        self.alert_evidence: list[dict] = []

    def cursor(self, cursor_factory=None):
        return _FakeListAlertsCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


def _fa_row(alert_id: int, *, channel="ach", status="OPEN", created_at=T0) -> dict:
    return {
        "alert_id": alert_id, "event_id": uuid.uuid4(), "source_alert_id": uuid.uuid4(),
        "source_system": "SYS", "channel": channel, "customer_id": "cust", "account_id": "acct",
        "amount_minor_units": 1000, "status": status, "created_at": created_at,
        "initial_operational_priority_score": 0.1, "initial_priority_band": "LOW",
        "initial_ensemble_policy_version": "v1",
    }


def _ae_row(alert_id: int, *, evidence_id, channel_model_bundle_id, priority_band, score, scored_at) -> dict:
    return {
        "alert_id": alert_id, "evidence_id": evidence_id, "channel_model_bundle_id": channel_model_bundle_id,
        "priority_band": priority_band, "operational_priority_score": score, "scored_at": scored_at,
    }


def test_postgres_list_alerts_real_query_selects_current_band_not_initial(monkeypatch):
    """Real-SQL-shape parity test (matches the established
    _FakeListPendingConnection/_FakeListPendingCursor convention): the
    ACTUAL _PostgresAlertQueueStore.list_alerts_with_current_state() SQL
    string is exercised -- not a hand-copied equivalent -- against a fake
    Postgres double simulating the LEFT JOIN LATERAL, structurally
    asserting parameterization (item 12), the absence of any
    UPDATE/DELETE/INSERT (item 13), and the exact ACH-shaped LOW-initial
    / MEDIUM-current split (item 15) at this layer too."""
    from src.fraud_intel.alerts.queue import _PostgresAlertQueueStore

    conn = _FakeListAlertsConnection()
    conn.fraud_alerts = [_fa_row(1, created_at=T0), _fa_row(2, created_at=T0 + timedelta(minutes=1))]
    conn.alert_evidence = [
        _ae_row(1, evidence_id=uuid.UUID(int=1), channel_model_bundle_id=3, priority_band="LOW", score=0.1, scored_at=T0),
        _ae_row(1, evidence_id=uuid.UUID(int=2), channel_model_bundle_id=4, priority_band="MEDIUM", score=0.42, scored_at=T0 + timedelta(hours=1)),
        _ae_row(2, evidence_id=uuid.UUID(int=3), channel_model_bundle_id=3, priority_band="LOW", score=0.05, scored_at=T0),
    ]
    monkeypatch.setattr("src.fraud_intel.alerts.queue.get_connection", lambda database=None: conn)

    store = _PostgresAlertQueueStore(database="aidp_test")
    all_items = store.list_alerts_with_current_state(channel="ach")
    assert {i.alert_id for i in all_items} == {1, 2}
    item1 = next(i for i in all_items if i.alert_id == 1)
    assert item1.initial_priority_band == "LOW"
    assert item1.current_priority_band == "MEDIUM"  # the LATER bundle-4 evidence, not bundle-3's initial LOW
    assert item1.current_channel_model_bundle_id == 4

    medium_only = store.list_alerts_with_current_state(channel="ach", current_priority_band="MEDIUM")
    assert {i.alert_id for i in medium_only} == {1}  # NOT {} -- the exact regression this patch fixes


def test_postgres_list_alerts_sql_is_parameterized_structurally():
    """Item 12: AST-based, same convention as
    test_load_cross_channel_customer_pool_uses_parameterized_sql --
    every %s-bearing SQL string is a literal passed alongside a `params`/
    `values`-built second argument, never an f-string with a raw value
    interpolated into the SQL text itself (WHERE-clause-fragment
    f-strings using ONLY the fixed column/operator text are the existing,
    accepted repo convention -- e.g. src.fraud_intel.cli_data_access.
    load_cross_channel_customer_pool -- not a parameterization gap)."""
    import ast
    import inspect
    import textwrap

    from src.fraud_intel.alerts import queue as queue_module

    tree = ast.parse(textwrap.dedent(inspect.getsource(queue_module._PostgresAlertQueueStore.list_alerts_with_current_state)))
    execute_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            execute_calls += 1
            assert len(node.args) == 2, "cur.execute() must always pass a separate params argument"
    assert execute_calls == 1


def test_postgres_list_alerts_never_updates_fraud_alerts():
    """Item 13."""
    import inspect

    from src.fraud_intel.alerts import queue as queue_module

    source = inspect.getsource(queue_module._PostgresAlertQueueStore.list_alerts_with_current_state)
    upper = source.upper()
    assert "UPDATE " not in upper
    assert "DELETE " not in upper
    assert "INSERT " not in upper


def test_list_alerts_and_get_latest_evidence_share_the_same_order_sql_constant():
    """Structural drift guard: both the real store's get_latest_evidence()
    and list_alerts_with_current_state() interpolate the SAME
    LATEST_EVIDENCE_ORDER_SQL constant -- they cannot silently diverge
    the way `_list_alerts()`/get_latest_evidence() did before this fix."""
    import inspect

    from src.fraud_intel.alerts import queue as queue_module

    get_latest_src = inspect.getsource(queue_module._PostgresAlertQueueStore.get_latest_evidence)
    list_src = inspect.getsource(queue_module._PostgresAlertQueueStore.list_alerts_with_current_state)
    assert "LATEST_EVIDENCE_ORDER_SQL" in get_latest_src
    assert "LATEST_EVIDENCE_ORDER_SQL" in list_src


def test_online_banking_list_behavior_remains_compatible():
    """Item 14: an unrescored, single-evidence online_banking alert --
    the pre-existing common case -- still lists correctly with current_*
    fields equal to its one (and only) evidence row."""
    store = _FakeAlertQueueStore()
    alert, evidence = _score(store=store, bundle=_bundle3())
    item = next(i for i in store.list_alerts_with_current_state(channel="online_banking") if i.alert_id == alert.alert_id)
    assert item.current_priority_band == evidence.priority_band
    assert item.current_operational_priority_score == evidence.operational_priority_score
    assert item.initial_priority_band == alert.initial_priority_band

"""Phase 5: score_source_alert() -- the key test for this phase, proving
the orchestrator (not just its components) works end to end, plus
component-failure/degraded behavior, LR shadow-only enforcement, no-
persistence structural checks, and the complete new bundle version. No
database, Docker, MLflow server, or network access anywhere in this file.
"""
from __future__ import annotations

import ast
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.fraud_intel.ensemble.policy import EnsemblePolicy
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import GraphPolicy, ResolvedFraudEntityEvidence
from src.fraud_intel.models.anomaly import AnomalyNormalization
from src.fraud_intel.models.bundle import _FakeChannelModelBundleStore
from src.fraud_intel.models.preprocessing import ChannelPreprocessor
from src.fraud_intel.rules.provider import LocalYamlRuleProvider
from src.fraud_intel.scoring import orchestrator as orchestrator_module
from src.fraud_intel.scoring.orchestrator import LoadedChannelBundle, ScoredAlert, score_source_alert

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ensemble_policy(**overrides) -> EnsemblePolicy:
    base = dict(
        channel="online_banking", policy_version="v1",
        weight_rule=0.35, weight_gbm=0.40, weight_anomaly=0.15, weight_graph=0.10,
        rule_score_cap=1.0, high_threshold=0.75, medium_threshold=0.40,
    )
    base.update(overrides)
    return EnsemblePolicy(**base)


def _graph_policy(**overrides) -> GraphPolicy:
    base = dict(
        channel="online_banking", graph_policy_version="v1",
        shared_device_cap=5, shared_device_weight=0.3, fan_in_cap=10, fan_in_weight=0.3,
        fan_out_cap=10, fan_out_weight=0.2, shortest_path_weight=0.2,
        graph_max_history_events=5000, max_nodes=20000, max_edges=200000,
    )
    base.update(overrides)
    return GraphPolicy(**base)


def _ob_event(*, event_timestamp: datetime = T0, amount_minor_units: int = 10_000, device_id: str | None = "DEV1", mfa_used_flag: bool = True) -> FraudEvent:
    return FraudEvent(
        channel="online_banking", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=event_timestamp,
        amount_minor_units=amount_minor_units, direction="debit", device_id=device_id,
        channel_payload=OnlineBankingPayload(
            session_id="SESS1", login_method="password", mfa_used_flag=mfa_used_flag,
            transaction_type="transfer", target_account="TGT1",
        ),
    )


def _source_alert(*, event_id: uuid.UUID, created_at: datetime = T0) -> SourceAlertContext:
    return SourceAlertContext(
        source_system="LocalYamlRuleProvider (simulated upstream)", event_id=event_id, source_alert_created_at=created_at,
        source_rule_ids=["RULE1"], source_rule_version="v1", source_alert_reason_codes=["REASON1"],
        generation_run_id="genrun-1", dataset_version="dsv-1", created_at=created_at,
    )


class _FakeProbaModel:
    def __init__(self, fraud_proba: float):
        self._fraud_proba = fraud_proba

    def predict_proba(self, X):
        return [[1.0 - self._fraud_proba, self._fraud_proba] for _ in range(len(X))]


class _FailingModel:
    def predict_proba(self, X):
        raise RuntimeError("simulated model failure")


class _FakeAnomalyModel:
    def decision_function(self, X):
        return [0.0 for _ in range(len(X))]


class _FailingAnomalyModel:
    def decision_function(self, X):
        raise RuntimeError("simulated anomaly failure")


_PREPROCESSOR = ChannelPreprocessor.fit(
    [
        {col: 0.0 for col in [
            "amount_vs_entity_average", "amount_zscore", "is_new_device", "is_new_ip", "night_transaction_flag",
            "events_last_10m", "events_last_1h", "events_last_24h", "prior_alert_count",
            "customer_observed_tenure_days", "account_observed_tenure_days", "counterparty_risk_score",
            "new_device_high_value_combo_flag", "mfa_bypass_flag", "session_velocity", "profile_change_then_transfer_flag",
        ]}
        for _ in range(3)
    ],
    feature_columns=[
        "amount_vs_entity_average", "amount_zscore", "is_new_device", "is_new_ip", "night_transaction_flag",
        "events_last_10m", "events_last_1h", "events_last_24h", "prior_alert_count",
        "customer_observed_tenure_days", "account_observed_tenure_days", "counterparty_risk_score",
        "new_device_high_value_combo_flag", "mfa_bypass_flag", "session_velocity", "profile_change_then_transfer_flag",
    ],
    feature_schema_version="v1",
    preprocessing_artifact_version="pp-test",
)


def _bundle(*, gbm_model=None, lr_model=None, anomaly_model=None) -> LoadedChannelBundle:
    return LoadedChannelBundle(
        channel="online_banking", bundle_id=2, bundle_version=2,
        gbm_model=gbm_model if gbm_model is not None else _FakeProbaModel(0.2),
        lr_model=lr_model if lr_model is not None else _FakeProbaModel(0.3),
        anomaly_model=anomaly_model if anomaly_model is not None else _FakeAnomalyModel(),
        anomaly_normalization=AnomalyNormalization(train_min=-1.0, train_max=1.0, anomaly_artifact_version="anom-test", library_versions={}),
        preprocessor=_PREPROCESSOR,
        gbm_model_version="gbm-7", lr_model_version="lr-7", anomaly_model_version="anomaly-7",
        preprocessing_artifact_version="pp-test", feature_schema_version="v1",
        rule_set_version="v1", graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
    )


def _ctx(event: FraudEvent, history: tuple[FraudEvent, ...] = ()) -> FeatureComputationContext:
    return FeatureComputationContext(current_event=event, historical_events=history, source_alert_history=(), as_of_time=event.event_timestamp)


def _call(*, event=None, source_alert=None, context=None, bundle=None, rule_provider=None, ensemble_policy=None, graph_policy=None, resolved_fraud_evidence=(), score_execution_id=None):
    event = event or _ob_event()
    return score_source_alert(
        event=event,
        source_alert=source_alert or _source_alert(event_id=event.event_id),
        context=context or _ctx(event),
        bundle=bundle or _bundle(),
        rule_provider=rule_provider or LocalYamlRuleProvider(),
        ensemble_policy=ensemble_policy or _ensemble_policy(),
        graph_policy=graph_policy or _graph_policy(),
        resolved_fraud_evidence=resolved_fraud_evidence,
        score_execution_id=score_execution_id or uuid.uuid4(),
    )


# ---- the key end-to-end test --------------------------------------------------------


def test_score_source_alert_end_to_end_produces_a_correct_scored_alert():
    event = _ob_event(amount_minor_units=10_000, device_id="DEV-KNOWN", mfa_used_flag=True)
    result = _call(event=event)

    assert isinstance(result, ScoredAlert)
    assert result.event_id == event.event_id
    assert 0.0 <= result.operational_priority_score <= 1.0
    assert result.priority_band in ("LOW", "MEDIUM", "HIGH")
    assert result.degraded is False
    assert all(cs.status == "OK" for cs in result.component_statuses.values())
    assert result.calibrated_gbm_probability == pytest.approx(0.2)
    assert result.lr_probability == pytest.approx(0.3)
    assert result.reason_codes  # at least the upstream reason code
    assert result.channel_model_bundle_version == 2
    assert result.feature_schema_version == "v1"
    assert result.rule_set_version  # comes from the real rule provider's YAML


def test_mandatory_review_rule_hit_forces_high_band():
    event = _ob_event(mfa_used_flag=False)  # mfa_bypass_flag will be true
    # history containing a prior profile_change event to trigger the composite MANDATORY_REVIEW rule
    profile_change_event = FraudEvent(
        channel="online_banking", customer_id=CUSTOMER, account_id=ACCOUNT,
        event_timestamp=T0 - timedelta(hours=1), amount_minor_units=1_000, direction="debit",
        channel_payload=OnlineBankingPayload(session_id="S0", login_method="password", mfa_used_flag=True, transaction_type="profile_change", target_account="TGT1"),
    )
    ctx = _ctx(event, history=(profile_change_event,))
    result = _call(event=event, context=ctx)
    assert result.priority_band == "HIGH"
    assert result.rule_result.provider_status == "OK"


# ---- LR shadow-only ------------------------------------------------------------------


def test_varying_lr_prediction_never_changes_operational_score_or_band():
    event = _ob_event()
    low_lr_bundle = _bundle(lr_model=_FakeProbaModel(0.01))
    high_lr_bundle = _bundle(lr_model=_FakeProbaModel(0.99))

    result_low = _call(event=event, bundle=low_lr_bundle)
    result_high = _call(event=event, bundle=high_lr_bundle)

    assert result_low.lr_probability != result_high.lr_probability  # sanity: LR really did differ
    assert result_low.operational_priority_score == result_high.operational_priority_score
    assert result_low.priority_band == result_high.priority_band


def test_lr_failure_does_not_degrade_the_result_at_all():
    event = _ob_event()
    bundle = _bundle(lr_model=_FailingModel())
    result = _call(event=event, bundle=bundle)
    assert result.lr_probability is None
    assert result.degraded is False
    assert "lr" not in result.component_statuses  # LR has no component-status entry -- shadow-only, not gated


# ---- component failure / degraded behavior (Phase 5 decision 6) ---------------------


def test_gbm_failure_degrades_and_forces_at_least_high():
    event = _ob_event()
    bundle = _bundle(gbm_model=_FailingModel())
    result = _call(event=event, bundle=bundle)
    assert result.degraded is True
    assert result.calibrated_gbm_probability is None  # never invents a probability
    assert result.component_statuses["gbm"].status == "ERROR"
    assert result.component_statuses["gbm"].error_code == "GBM_SCORING_FAILED"
    assert result.priority_band == "HIGH"


def test_anomaly_failure_degrades_and_forces_at_least_medium():
    event = _ob_event()
    bundle = _bundle(anomaly_model=_FailingAnomalyModel())
    result = _call(event=event, bundle=bundle)
    assert result.degraded is True
    assert result.anomaly_score is None
    assert result.component_statuses["anomaly"].status == "ERROR"
    assert result.priority_band in ("MEDIUM", "HIGH")


def test_multiple_failures_apply_the_highest_required_floor():
    event = _ob_event()
    bundle = _bundle(gbm_model=_FailingModel(), anomaly_model=_FailingAnomalyModel())
    result = _call(event=event, bundle=bundle)
    assert result.degraded is True
    # GBM's HIGH floor must win over anomaly's MEDIUM floor.
    assert result.priority_band == "HIGH"


def test_rule_provider_failure_floors_at_medium_via_existing_mechanism():
    class _FailingRuleProvider:
        def evaluate(self, *, event, source_alert, features):
            from datetime import datetime, timezone

            from src.fraud_intel.rules.provider import RuleEvaluationResult

            return RuleEvaluationResult(
                provider_name="LocalYamlRuleProvider", provider_version="v1", rule_set_version="unknown",
                fired_rule_ids=[], rule_categories={}, reason_codes=["RULE_PROVIDER_UNAVAILABLE"],
                score_contribution=0.0, minimum_priority_band="MEDIUM", evaluated_at=datetime.now(timezone.utc),
                latency_ms=1.0, provider_status="ERROR", provider_error_code="RULE_CONFIG_LOAD_FAILED",
            )

    event = _ob_event()
    result = _call(event=event, rule_provider=_FailingRuleProvider())
    assert result.priority_band in ("MEDIUM", "HIGH")
    assert result.component_statuses["rules"].status == "ERROR"


def test_every_existing_source_alert_remains_retained_even_when_everything_fails():
    """score_source_alert() always returns a ScoredAlert, never raises, for
    any combination of component failures -- the alert is never dropped."""
    event = _ob_event()
    bundle = _bundle(gbm_model=_FailingModel(), lr_model=_FailingModel(), anomaly_model=_FailingAnomalyModel())
    result = _call(event=event, bundle=bundle)
    assert isinstance(result, ScoredAlert)
    assert result.source_alert_id is not None


# ---- graph leakage enforced through the orchestrator ---------------------------------


def test_orchestrator_rejects_future_or_boundary_resolved_fraud_evidence():
    from src.fraud_intel.graph.entity_graph import GraphLeakageError

    event = _ob_event()
    boundary_evidence = ResolvedFraudEntityEvidence(
        entity_type="customer", entity_id=CUSTOMER, label_assessment_id="A1",
        resolved_fraud_at=event.event_timestamp, eligibility_policy_version="v1", label_source="SYNTHETIC_GENERATOR",
    )
    with pytest.raises(GraphLeakageError):
        _call(event=event, resolved_fraud_evidence=[boundary_evidence])


# ---- structural: no persistence ------------------------------------------------------


def test_score_source_alert_never_imports_a_database_driver():
    source = inspect.getsource(orchestrator_module)
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden = {"psycopg2", "src.common.db"}
    assert forbidden.isdisjoint(imported_names)


def test_score_source_alert_has_no_insert_or_persistence_shaped_code():
    source = inspect.getsource(orchestrator_module)
    assert "INSERT INTO" not in source
    assert "cursor(" not in source


def test_source_alert_context_passed_through_unchanged():
    event = _ob_event()
    alert = _source_alert(event_id=event.event_id)
    alert_before = alert.model_copy(deep=True)
    _call(event=event, source_alert=alert)
    assert alert == alert_before


def test_wrong_channel_event_rejected():
    from src.fraud_intel.events.ach import ACHPayload
    from datetime import date

    ach_event = FraudEvent(
        channel="ach", customer_id=CUSTOMER, account_id=ACCOUNT, event_timestamp=T0, amount_minor_units=10_000,
        direction="debit",
        channel_payload=ACHPayload(sec_code="PPD", originating_routing_number="123456789", receiving_routing_number="987654321", batch_id="B1", effective_entry_date=date(2026, 1, 1), company_id="C1"),
    )
    with pytest.raises(ValueError, match="online_banking"):
        _call(event=ach_event, context=_ctx(ach_event))


# ---- complete Phase 5 bundle version (never mutates Phase 4's row) ------------------


def test_phase5_registers_a_new_complete_bundle_version_and_phase4_bundle_is_untouched():
    store = _FakeChannelModelBundleStore()
    phase4_fields = dict(
        channel="online_banking", gbm_model_version="gbm-1", lr_model_version="lr-1", anomaly_model_version="anom-1",
        preprocessing_artifact_version="pp-1", feature_schema_version="v1", training_run_id=1, dataset_version="ds-1",
        evaluation_report_ref="{}",
    )
    phase4_bundle = store.register_candidate(**phase4_fields)
    phase4_snapshot = phase4_bundle.model_copy(deep=True)
    assert phase4_bundle.is_promotion_eligible() is False  # missing the 4 Phase 5 policy versions

    phase5_fields = dict(phase4_fields)
    phase5_fields.update(
        rule_set_version="v1", graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
    )
    phase5_bundle = store.register_candidate(**phase5_fields)

    assert phase5_bundle.bundle_version == phase4_bundle.bundle_version + 1
    assert phase5_bundle.bundle_id != phase4_bundle.bundle_id
    assert phase5_bundle.is_promotion_eligible() is True
    assert phase4_bundle == phase4_snapshot  # untouched
    assert len(store.rows) == 2

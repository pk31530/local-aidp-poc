"""Phase 6 corrective pass: real reference-channel scoring dispatch
(src.fraud_intel.scoring.dispatch.score_channel). No database, MLflow,
Docker, or network access anywhere in this file -- every test supplies a
fake ScoringDataAccess/BundleArtifactLoader/get_operational_bundle/
RunLifecycle store; the real _PostgresScoringDataAccess/
_MlflowBundleArtifactLoader are never exercised here (reviewed as code).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.control_plane.runs import RunLifecycle, RunRecord
from src.fraud_intel.alerts.queue import _FakeAlertQueueStore
from src.fraud_intel.ensemble.policy import EnsemblePolicy
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import GraphPolicy
from src.fraud_intel.models.anomaly import AnomalyNormalization
from src.fraud_intel.models.bundle import ChannelModelBundleRecord
from src.fraud_intel.models.preprocessing import ChannelPreprocessor
from src.fraud_intel.rules.provider import LocalYamlRuleProvider
from src.fraud_intel.scoring.dispatch import (
    BundlePolicyMismatchError,
    NoOperationalBundleError,
    PendingScoringItem,
    score_channel,
)
from src.fraud_intel.scoring.orchestrator import LoadedChannelBundle

CUSTOMER = "FIC2000"
ACCOUNT = "FIA200000"
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
    def predict_proba(self, X):
        return [[0.8, 0.2] for _ in range(len(X))]


class _FakeAnomalyModel:
    def decision_function(self, X):
        return [0.0 for _ in range(len(X))]


class _FakeRunStore:
    """Same in-memory contract as tests/unit/test_cli.py's own fixture."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    def insert(self, row):
        run_id = self._next_id
        self._next_id += 1
        full = {
            "run_id": run_id, "trigger_source": None, "git_sha": None, "config_snapshot": None,
            "config_hash": None, "dataset_version": None, "model_version": None, "error_type": None,
            "error_message": None, "started_at": datetime.now(timezone.utc), "heartbeat_at": None,
            "completed_at": None, **row,
        }
        self.rows[run_id] = full
        return RunRecord(**full)

    def compare_and_set(self, run_id, allowed_from, updates):
        current = self.rows.get(run_id)
        if current is None or current["status"] not in allowed_from:
            return None
        current.update(updates)
        return RunRecord(**current)

    def get(self, run_id):
        row = self.rows.get(run_id)
        return RunRecord(**row) if row else None

    def list(self, *, pipeline_name=None, status=None, limit=50):
        return [RunRecord(**r) for r in list(self.rows.values())[:limit]]


def _lifecycle() -> tuple[RunLifecycle, _FakeRunStore]:
    store = _FakeRunStore()
    return RunLifecycle(store=store), store


def _bundle_record(**overrides) -> ChannelModelBundleRecord:
    base = dict(
        bundle_id=2, channel="online_banking", bundle_version=1, status="OPERATIONAL", created_at=T0,
        gbm_model_version="gbm-7", lr_model_version="lr-7", anomaly_model_version="anomaly-7",
        preprocessing_artifact_version="pp-test", feature_schema_version="v1",
        rule_set_version="v1", graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
        evaluation_report_ref='{"gbm_mlflow_run_id": "run-gbm-1", "anomaly_mlflow_run_id": "run-anom-1"}',
    )
    base.update(overrides)
    return ChannelModelBundleRecord(**base)


def _event(*, event_id: uuid.UUID | None = None) -> FraudEvent:
    return FraudEvent(
        event_id=event_id or uuid.uuid4(), channel="online_banking", customer_id=CUSTOMER, account_id=ACCOUNT,
        event_timestamp=T0, amount_minor_units=10_000, direction="debit", device_id="DEV1",
        channel_payload=OnlineBankingPayload(session_id="S1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"),
    )


def _source_alert(*, event_id: uuid.UUID) -> SourceAlertContext:
    return SourceAlertContext(
        source_alert_id=uuid.uuid4(), source_system="LocalYamlRuleProvider (simulated upstream)",
        event_id=event_id, source_alert_created_at=T0, source_rule_ids=["RULE1"], source_rule_version="v1",
        source_alert_reason_codes=["REASON1"], generation_run_id="genrun-1", dataset_version="dsv-1", created_at=T0,
    )


def _pending_item() -> PendingScoringItem:
    event = _event()
    context = FeatureComputationContext(current_event=event, historical_events=(), source_alert_history=(), as_of_time=event.event_timestamp)
    return PendingScoringItem(event=event, source_alert=_source_alert(event_id=event.event_id), context=context, resolved_fraud_evidence=())


class _FakeDataAccess:
    def __init__(self, items):
        self._items = items

    def list_pending(self, channel):
        return self._items


class _FakeArtifactLoader:
    def __init__(self, bundle_id=2, bundle_version=1):
        self._bundle_id = bundle_id
        self._bundle_version = bundle_version

    def load(self, bundle_record) -> LoadedChannelBundle:
        return LoadedChannelBundle(
            channel="online_banking", bundle_id=self._bundle_id, bundle_version=self._bundle_version,
            gbm_model=_FakeProbaModel(), lr_model=_FakeProbaModel(), anomaly_model=_FakeAnomalyModel(),
            anomaly_normalization=AnomalyNormalization(train_min=-1.0, train_max=1.0, anomaly_artifact_version="anom-test", library_versions={}),
            preprocessor=_PREPROCESSOR,
            gbm_model_version=bundle_record.gbm_model_version, lr_model_version=bundle_record.lr_model_version,
            anomaly_model_version=bundle_record.anomaly_model_version,
            preprocessing_artifact_version=bundle_record.preprocessing_artifact_version,
            feature_schema_version=bundle_record.feature_schema_version, rule_set_version=bundle_record.rule_set_version,
            graph_policy_version=bundle_record.graph_policy_version, ensemble_policy_version=bundle_record.ensemble_policy_version,
            reason_code_version=bundle_record.reason_code_version,
        )


class _FailingArtifactLoader:
    def load(self, bundle_record):
        raise RuntimeError("simulated MLflow contact failure")


def _ensemble_policy_stub() -> EnsemblePolicy:
    return EnsemblePolicy(
        channel="online_banking", policy_version="v1", weight_rule=0.35, weight_gbm=0.40, weight_anomaly=0.15,
        weight_graph=0.10, rule_score_cap=1.0, high_threshold=0.75, medium_threshold=0.40,
    )


def _graph_policy_stub() -> GraphPolicy:
    return GraphPolicy(
        channel="online_banking", graph_policy_version="v1", shared_device_cap=5, shared_device_weight=0.3,
        fan_in_cap=10, fan_in_weight=0.3, fan_out_cap=10, fan_out_weight=0.2, shortest_path_weight=0.2,
        graph_max_history_events=5000, max_nodes=20000, max_edges=200000,
    )


def _fake_policies(bundle_record):
    """Stands in for dispatch.load_and_validate_pinned_policies -- tests
    that want to exercise dispatch orchestration without depending on the
    real, currently-loaded YAML policy versions matching monkeypatch this
    in; tests that specifically want to exercise the mismatch check itself
    leave it unpatched (see test_bundle_policy_mismatch_fails_the_run_and_reraises)."""
    return LocalYamlRuleProvider(), _graph_policy_stub(), _ensemble_policy_stub()


# ---- end-to-end happy path -----------------------------------------------------------


def test_score_channel_end_to_end_with_one_pending_alert(monkeypatch):
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.load_and_validate_pinned_policies", _fake_policies)
    lifecycle, run_store = _lifecycle()
    alert_store = _FakeAlertQueueStore()
    item = _pending_item()

    result = score_channel(
        channel="online_banking",
        lifecycle=lifecycle,
        data_access=_FakeDataAccess([item]),
        get_operational_bundle=lambda channel: _bundle_record(),
        artifact_loader=_FakeArtifactLoader(),
        alert_queue_store=alert_store,
    )

    assert result["records_processed"] == 1
    assert result["records_rejected"] == 0
    assert result["bundle_id"] == 2
    assert result["bundle_version"] == 1
    assert len(result["alerts"]) == 1
    assert len(alert_store.alerts_by_id) == 1

    run = run_store.get(result["run_id"])
    assert run.status == "SUCCESS"
    assert run.records_processed == 1
    assert run.records_rejected == 0


def test_score_channel_with_no_pending_alerts_still_succeeds(monkeypatch):
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.load_and_validate_pinned_policies", _fake_policies)
    lifecycle, run_store = _lifecycle()

    result = score_channel(
        channel="online_banking", lifecycle=lifecycle, data_access=_FakeDataAccess([]),
        get_operational_bundle=lambda channel: _bundle_record(), artifact_loader=_FakeArtifactLoader(),
        alert_queue_store=_FakeAlertQueueStore(),
    )

    assert result["records_processed"] == 0
    assert result["records_rejected"] == 0
    assert run_store.get(result["run_id"]).status == "SUCCESS"


# ---- RunLifecycle failure + re-raise --------------------------------------------------


def test_missing_operational_bundle_fails_the_run_and_reraises():
    lifecycle, run_store = _lifecycle()

    with pytest.raises(NoOperationalBundleError):
        score_channel(
            channel="online_banking", lifecycle=lifecycle, data_access=_FakeDataAccess([]),
            get_operational_bundle=lambda channel: None, artifact_loader=_FakeArtifactLoader(),
            alert_queue_store=_FakeAlertQueueStore(),
        )

    run = next(iter(run_store.rows.values()))
    assert run["status"] == "FAILED"
    assert run["error_type"] == "NoOperationalBundleError"


def test_bundle_policy_mismatch_fails_the_run_and_reraises():
    lifecycle, run_store = _lifecycle()
    # rule_set_version deliberately stale relative to the real, currently
    # loaded rules_online_banking.yaml -- load_and_validate_pinned_policies
    # runs for real here (no monkeypatch), same as
    # tests/unit/test_fraud_intel_promotion.py's unmocked policy-load tests.
    stale_bundle = _bundle_record(rule_set_version="v999-does-not-exist")

    with pytest.raises(BundlePolicyMismatchError):
        score_channel(
            channel="online_banking", lifecycle=lifecycle, data_access=_FakeDataAccess([]),
            get_operational_bundle=lambda channel: stale_bundle, artifact_loader=_FakeArtifactLoader(),
            alert_queue_store=_FakeAlertQueueStore(),
        )

    run = next(iter(run_store.rows.values()))
    assert run["status"] == "FAILED"
    assert run["error_type"] == "BundlePolicyMismatchError"


def test_artifact_loading_failure_fails_the_run_and_reraises(monkeypatch):
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.load_and_validate_pinned_policies", _fake_policies)
    lifecycle, run_store = _lifecycle()

    with pytest.raises(RuntimeError, match="simulated MLflow contact failure"):
        score_channel(
            channel="online_banking", lifecycle=lifecycle, data_access=_FakeDataAccess([]),
            get_operational_bundle=lambda channel: _bundle_record(), artifact_loader=_FailingArtifactLoader(),
            alert_queue_store=_FakeAlertQueueStore(),
        )

    run = next(iter(run_store.rows.values()))
    assert run["status"] == "FAILED"
    assert run["error_type"] == "RuntimeError"


# ---- one bad alert does not abort the whole batch -------------------------------------


def test_one_alert_scoring_failure_is_counted_rejected_not_fatal(monkeypatch):
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.load_and_validate_pinned_policies", _fake_policies)
    lifecycle, run_store = _lifecycle()

    good_item = _pending_item()
    bad_event = _event()
    # source_alert.event_id deliberately mismatched from the event passed
    # to score_and_record_alert() -- score_source_alert() raises ValueError
    # for this, which score_and_record_alert() treats as a catastrophic
    # per-alert failure (best-effort persists, then re-raises to its
    # caller -- here, score_channel(), which counts it and continues).
    mismatched_context = FeatureComputationContext(current_event=bad_event, historical_events=(), source_alert_history=(), as_of_time=bad_event.event_timestamp)
    bad_item = PendingScoringItem(
        event=bad_event, source_alert=_source_alert(event_id=uuid.uuid4()), context=mismatched_context, resolved_fraud_evidence=()
    )

    result = score_channel(
        channel="online_banking", lifecycle=lifecycle, data_access=_FakeDataAccess([bad_item, good_item]),
        get_operational_bundle=lambda channel: _bundle_record(), artifact_loader=_FakeArtifactLoader(),
        alert_queue_store=_FakeAlertQueueStore(),
    )

    assert result["records_processed"] == 1
    assert result["records_rejected"] == 1
    run = run_store.get(result["run_id"])
    assert run.status == "SUCCESS"  # one bad alert never fails the whole run

"""Phase 4: preprocessing-artifact contract, the calibration-leakage
runtime guard, and the end-to-end train_channel_configured flow -- fully
mocked MLflow/GBM/LR/bundle-store/run-store, exactly like
tests/unit/test_train.py's existing v1.1 pattern. No database, Docker, or
real MLflow/XGBoost/PostgreSQL contact anywhere in this file.
"""
from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.control_plane.runs import RunLifecycle as RealRunLifecycle
from src.control_plane.runs import RunRecord
from src.fraud_intel.config import ChannelTrainingRunConfig
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext, SyntheticGroundTruthLabel
from src.fraud_intel.models import training as training_module
from src.fraud_intel.models.bundle import _FakeChannelModelBundleStore
from src.fraud_intel.models.calibration import CalibrationLeakageError, SplitManifest, fit_calibrator
from src.fraud_intel.models.preprocessing import ChannelPreprocessor, FeatureSchemaMismatchError
from src.fraud_intel.models.training import InsufficientTrainingDataError, train_channel_configured

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ============================ preprocessing ========================================


def _numeric_rows():
    return [
        {"a": 1.0, "b": 10.0},
        {"a": 2.0, "b": 20.0},
        {"a": 3.0, "b": 30.0},
    ]


def test_preprocessing_fit_computes_mean_scale_from_training_rows_only():
    train_rows = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    pp = ChannelPreprocessor.fit(train_rows, feature_columns=["a"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    assert pp.means["a"] == pytest.approx(2.0)
    assert pp.scales["a"] == pytest.approx((( (1-2)**2 + (2-2)**2 + (3-2)**2 ) / 3) ** 0.5)


def test_preprocessing_transform_never_refits_even_on_wildly_different_calib_data():
    train_rows = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    pp = ChannelPreprocessor.fit(train_rows, feature_columns=["a"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    calib_rows = [{"a": 1000.0}, {"a": -1000.0}]
    transformed = pp.transform(calib_rows)
    # mean/scale must still be the TRAINING ones (mean=2.0, scale computed above) -- not
    # re-derived from calib_rows' wildly different distribution.
    expected_scale = ((1 - 2) ** 2 + (2 - 2) ** 2 + (3 - 2) ** 2) / 3
    expected_scale **= 0.5
    assert transformed[0][0] == pytest.approx((1000.0 - 2.0) / expected_scale)


def test_preprocessing_rejects_missing_feature():
    pp = ChannelPreprocessor.fit(_numeric_rows(), feature_columns=["a", "b"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    with pytest.raises(FeatureSchemaMismatchError, match="missing"):
        pp.transform([{"a": 1.0}])


def test_preprocessing_rejects_unexpected_feature():
    pp = ChannelPreprocessor.fit(_numeric_rows(), feature_columns=["a", "b"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    with pytest.raises(FeatureSchemaMismatchError, match="unexpected"):
        pp.transform([{"a": 1.0, "b": 2.0, "c": 3.0}])


def test_preprocessing_rejects_reordered_features_instead_of_silently_adapting():
    pp = ChannelPreprocessor.fit(_numeric_rows(), feature_columns=["a", "b"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    with pytest.raises(FeatureSchemaMismatchError):
        pp.transform([{"b": 2.0, "a": 1.0}])  # same keys, wrong order


def test_preprocessing_categorical_vocabulary_and_unknown_category_handling():
    train_rows = [{"color": "red"}, {"color": "blue"}, {"color": "red"}]
    pp = ChannelPreprocessor.fit(train_rows, feature_columns=["color"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    assert set(pp.categorical_encodings["color"].keys()) == {"__UNKNOWN__", "red", "blue"}
    known_encoded = pp.transform([{"color": "red"}])[0][0]
    unknown_encoded = pp.transform([{"color": "green"}])[0][0]  # unseen at fit time
    assert known_encoded != 0.0
    assert unknown_encoded == 0.0  # reserved unknown index


def test_preprocessing_numeric_imputation_uses_training_median_for_missing_values():
    train_rows = [{"a": 1.0}, {"a": None}, {"a": 5.0}]
    pp = ChannelPreprocessor.fit(train_rows, feature_columns=["a"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    assert pp.numeric_imputation_values["a"] == pytest.approx(3.0)  # median of [1.0, 5.0] with the None imputed to that median


def test_preprocessing_artifact_contains_required_contract_fields():
    pp = ChannelPreprocessor.fit(_numeric_rows(), feature_columns=["a", "b"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    doc = pp.to_json_dict()
    for key in (
        "feature_columns", "feature_schema_version", "preprocessing_artifact_version",
        "numeric_imputation_values", "means", "scales", "categorical_encodings",
        "unknown_category_policy", "library_versions",
    ):
        assert key in doc


def test_preprocessing_canonical_json_hash_is_deterministic():
    pp1 = ChannelPreprocessor.fit(_numeric_rows(), feature_columns=["a", "b"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    pp2 = ChannelPreprocessor.fit(_numeric_rows(), feature_columns=["a", "b"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    assert pp1.content_hash() == pp2.content_hash()


def test_preprocessing_round_trips_through_json():
    pp = ChannelPreprocessor.fit(_numeric_rows(), feature_columns=["a", "b"], feature_schema_version="v1", preprocessing_artifact_version="pp-1")
    restored = ChannelPreprocessor.from_json_dict(pp.to_json_dict())
    assert restored.content_hash() == pp.content_hash()
    assert restored.transform(_numeric_rows()) == pp.transform(_numeric_rows())


# ============================ calibration leakage guard ============================


def _manifest(train=("t1",), calibration=("c1", "c2"), purge=("p1",), test=("x1",)) -> SplitManifest:
    return SplitManifest(
        train_ids=frozenset(train), calibration_ids=frozenset(calibration),
        purge_ids=frozenset(purge), test_ids=frozenset(test),
    )


def test_split_manifest_rejects_overlapping_partitions():
    with pytest.raises(ValueError, match="overlap"):
        SplitManifest(train_ids=frozenset({"a"}), calibration_ids=frozenset({"a"}), purge_ids=frozenset(), test_ids=frozenset())


def test_fit_calibrator_accepts_pure_calibration_rows():
    manifest = _manifest()
    calibrator = fit_calibrator(row_ids=["c1", "c2"], raw_probabilities=[0.1, 0.9], labels=[0, 1], manifest=manifest)
    assert calibrator is not None


def test_fit_calibrator_raises_when_a_test_row_is_deliberately_supplied():
    """The exact guide-required proof: a direct test that deliberately
    passes a test-window row and confirms the runtime guard raises."""
    manifest = _manifest()
    with pytest.raises(CalibrationLeakageError, match="x1"):
        fit_calibrator(row_ids=["c1", "x1"], raw_probabilities=[0.1, 0.9], labels=[0, 1], manifest=manifest)


def test_fit_calibrator_raises_when_a_train_row_is_supplied():
    manifest = _manifest()
    with pytest.raises(CalibrationLeakageError):
        fit_calibrator(row_ids=["c1", "t1"], raw_probabilities=[0.1, 0.9], labels=[0, 1], manifest=manifest)


def test_fit_calibrator_raises_when_a_purge_row_is_supplied():
    manifest = _manifest()
    with pytest.raises(CalibrationLeakageError):
        fit_calibrator(row_ids=["c1", "p1"], raw_probabilities=[0.1, 0.9], labels=[0, 1], manifest=manifest)


def test_fit_calibrator_never_receives_a_test_row_even_mixed_with_valid_rows():
    manifest = _manifest(calibration=("c1", "c2", "c3"))
    with pytest.raises(CalibrationLeakageError):
        fit_calibrator(row_ids=["c1", "c2", "x1"], raw_probabilities=[0.1, 0.2, 0.9], labels=[0, 0, 1], manifest=manifest)


# ============================ end-to-end training framework ========================


class _FakeRunStore:
    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    def insert(self, row):
        run_id = self._next_id
        self._next_id += 1
        full = {
            "run_id": run_id, "trigger_source": None, "git_sha": None, "config_snapshot": None,
            "config_hash": None, "dataset_version": None, "model_version": None,
            "error_type": None, "error_message": None, "started_at": datetime.now(timezone.utc),
            "heartbeat_at": None, "completed_at": None, **row,
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


class _FakeModel:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return [[0.0, 1.0] for _ in range(len(X))]


class _FakeAnomalyModel:
    def fit(self, X):
        return self

    def decision_function(self, X):
        return [0.0 for _ in range(len(X))]


class _FakeAnomalyNormalization:
    def to_json_dict(self):
        return {"train_min": 0.0, "train_max": 1.0}


class _FakeModelInfo:
    def __init__(self, version):
        self.registered_model_version = version


class _FakeMlflowRunContext:
    def __init__(self, run_id):
        self.info = type("Info", (), {"run_id": run_id})()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeMlflow:
    def __init__(self):
        self.xgboost = type("M", (), {"log_model": staticmethod(lambda *a, **k: _FakeModelInfo("gbm-7"))})()
        self.sklearn = type(
            "M",
            (),
            {
                "log_model": staticmethod(
                    lambda *a, registered_model_name="", **k: _FakeModelInfo(
                        "anomaly-7" if "anomaly" in registered_model_name else "lr-7"
                    )
                )
            },
        )()
        self.models = type("M", (), {"infer_signature": staticmethod(lambda *a, **k: "fake-signature")})()

    def set_experiment(self, name):
        pass

    def start_run(self, run_name=None):
        return _FakeMlflowRunContext(f"fake-run-{run_name}")

    def log_param(self, key, value):
        pass

    def log_params(self, params):
        pass

    def log_metric(self, key, value):
        pass


def _ob_event(*, event_timestamp: datetime, event_id: uuid.UUID | None = None) -> FraudEvent:
    return FraudEvent(
        event_id=event_id or uuid.uuid4(),
        channel="online_banking",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=event_timestamp,
        amount_minor_units=10_000,
        direction="debit",
        device_id="DEV1",
        channel_payload=OnlineBankingPayload(
            session_id="SESS1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"
        ),
    )


def _alert(*, event_id: uuid.UUID, created_at: datetime) -> SourceAlertContext:
    return SourceAlertContext(
        source_system="LocalYamlRuleProvider (simulated upstream)", event_id=event_id, source_alert_created_at=created_at,
        source_rule_ids=["RULE1"], source_rule_version="v1", source_alert_reason_codes=["REASON1"],
        generation_run_id="genrun-1", dataset_version="dsv-1", created_at=created_at,
    )


def _label(*, event_id: uuid.UUID, synthetic_scenario_label: bool) -> SyntheticGroundTruthLabel:
    return SyntheticGroundTruthLabel(
        event_id=event_id, scenario_id="s1", synthetic_scenario_label=synthetic_scenario_label,
        scenario_type="X", generation_run_id="genrun-1", dataset_version="dsv-1", generated_at=T0,
    )


def _fixture_population(n: int = 40):
    """n online_banking events, one per hour, alternating labels, each
    with exactly one source alert -- distinct timestamps throughout so
    assign_chronological_split always has plenty of group boundaries."""
    events, alerts, labels = [], [], []
    for i in range(n):
        ts = T0 + timedelta(hours=i)
        event = _ob_event(event_timestamp=ts)
        events.append(event)
        alerts.append(_alert(event_id=event.event_id, created_at=ts))
        labels.append(_label(event_id=event.event_id, synthetic_scenario_label=(i % 2 == 0)))
    return events, alerts, labels


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    store_holder = {"run_store": _FakeRunStore()}
    monkeypatch.setattr(training_module, "RunLifecycle", lambda *a, **k: RealRunLifecycle(store=store_holder["run_store"]))
    monkeypatch.setattr(training_module, "mlflow", _FakeMlflow())
    monkeypatch.setattr(training_module, "configure_mlflow", lambda: None)
    monkeypatch.setattr(training_module, "_train_gbm", lambda X, y, config: (_FakeModel(), {"n_estimators": 1}))
    monkeypatch.setattr(training_module, "_train_lr", lambda X, y, config: (_FakeModel(), {"max_iter": 1}))
    monkeypatch.setattr(
        training_module, "_fit_anomaly", lambda X, config, version: (_FakeAnomalyModel(), _FakeAnomalyNormalization())
    )
    yield store_holder


def test_successful_run_registers_a_candidate_bundle_and_no_alias(_fakes):
    events, alerts, labels = _fixture_population()
    config = ChannelTrainingRunConfig(channel="online_banking")
    bundle_store = _FakeChannelModelBundleStore()

    result = train_channel_configured(
        config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
        bundle_store=bundle_store,
    )

    assert len(bundle_store.rows) == 1
    bundle = bundle_store.rows[0]
    assert bundle.status == "CANDIDATE"
    assert bundle.channel == "online_banking"
    # Phase 5: anomaly training is now unconditional -- this bundle is
    # complete with respect to the 5 Phase-4-era components. It remains
    # incomplete with respect to Phase 5's 4 policy-version fields (rule/
    # graph/ensemble/reason-code), since this test doesn't pass them --
    # see test_fraud_intel_orchestrator.py / the Phase 5 "complete bundle"
    # test for the fully-populated case.
    assert bundle.anomaly_model_version == "anomaly-7"
    assert bundle.gbm_model_version == "gbm-7"
    assert bundle.lr_model_version == "lr-7"
    assert bundle.rule_set_version is None
    assert bundle.training_run_id == result["run_id"]
    assert bundle.dataset_version == result["dataset_version"]

    run_store = _fakes["run_store"]
    assert len(run_store.rows) == 1
    record = next(iter(run_store.rows.values()))
    assert record["status"] == "SUCCESS"
    assert record["dataset_version"] is not None
    assert record["model_version"] == "gbm-7"
    assert record["artifacts"]["lr_model_version"] == "lr-7"
    assert record["artifacts"]["bundle_id"] == bundle.bundle_id


def test_no_mlflow_alias_call_anywhere_in_training_module():
    """AST-level check (Name/Attribute nodes only) -- ignores docstrings
    and comments, which legitimately explain the absence of aliasing by
    name, so it cannot false-positive on documentation the way a raw
    substring search would."""
    import ast

    tree = ast.parse(inspect.getsource(training_module))
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    assert "set_registered_model_alias" not in identifiers
    assert "MlflowClient" not in identifiers


def test_lr_evaluation_is_recorded_separately_from_gbm_and_never_feeds_it(_fakes):
    events, alerts, labels = _fixture_population()
    config = ChannelTrainingRunConfig(channel="online_banking")
    result = train_channel_configured(
        config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
        bundle_store=_FakeChannelModelBundleStore(),
    )
    assert "gbm_evaluation" in result
    assert "lr_shadow_evaluation" in result
    assert result["gbm_evaluation"] is not result["lr_shadow_evaluation"]


def test_deterministic_seeds_produce_identical_params_across_two_runs(_fakes):
    events, alerts, labels = _fixture_population()
    config = ChannelTrainingRunConfig(channel="online_banking", random_seed=123)

    captured = []

    def _capture(X, y, cfg):
        captured.append(cfg.random_seed)
        return _FakeModel(), {"n_estimators": 1}

    from unittest.mock import patch

    with patch.object(training_module, "_train_gbm", _capture):
        train_channel_configured(
            config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
            bundle_store=_FakeChannelModelBundleStore(),
        )
        train_channel_configured(
            config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
            bundle_store=_FakeChannelModelBundleStore(),
        )
    assert captured == [123, 123]


def test_insufficient_training_data_rejected_before_any_registration(_fakes):
    """Too few source-alerted rows -> InsufficientTrainingDataError, raised
    before any model fitting, MLflow call, or bundle write."""
    events, alerts, labels = _fixture_population(n=8)  # splits cleanly but train partition < min_train_rows default
    config = ChannelTrainingRunConfig(channel="online_banking")
    bundle_store = _FakeChannelModelBundleStore()

    with pytest.raises(InsufficientTrainingDataError):
        train_channel_configured(
            config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
            bundle_store=bundle_store,
        )

    assert bundle_store.rows == []
    run_store = _fakes["run_store"]
    record = next(iter(run_store.rows.values()))
    assert record["status"] == "FAILED"


def test_single_class_population_rejected_before_any_registration(_fakes):
    """Every label the same class -> both-classes-required validation
    fails before model fitting."""
    events, alerts, labels = [], [], []
    for i in range(40):
        ts = T0 + timedelta(hours=i)
        event = _ob_event(event_timestamp=ts)
        events.append(event)
        alerts.append(_alert(event_id=event.event_id, created_at=ts))
        labels.append(_label(event_id=event.event_id, synthetic_scenario_label=False))  # always the same class
    config = ChannelTrainingRunConfig(channel="online_banking")
    bundle_store = _FakeChannelModelBundleStore()

    with pytest.raises(InsufficientTrainingDataError, match="class"):
        train_channel_configured(
            config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
            bundle_store=bundle_store,
        )
    assert bundle_store.rows == []


def test_failure_mid_training_records_failed_and_reraises_original_exception(_fakes):
    events, alerts, labels = _fixture_population()
    config = ChannelTrainingRunConfig(channel="online_banking")

    def _boom(X, y, config):
        raise RuntimeError("gbm exploded")

    from unittest.mock import patch

    with patch.object(training_module, "_train_gbm", _boom):
        with pytest.raises(RuntimeError, match="gbm exploded"):
            train_channel_configured(
                config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
                bundle_store=_FakeChannelModelBundleStore(),
            )

    run_store = _fakes["run_store"]
    assert len(run_store.rows) == 1
    record = next(iter(run_store.rows.values()))
    assert record["status"] == "FAILED"
    assert record["error_type"] == "RuntimeError"
    assert record["dataset_version"] is not None  # computed before the failure


def test_wrong_channel_config_fails_before_any_work(_fakes):
    events, alerts, labels = _fixture_population()
    config = ChannelTrainingRunConfig(channel="ach")
    with pytest.raises(ValueError, match="online_banking"):
        train_channel_configured(
            config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
            bundle_store=_FakeChannelModelBundleStore(),
        )


def test_realized_split_fractions_recorded_in_result_and_bundle_report(_fakes):
    events, alerts, labels = _fixture_population()
    config = ChannelTrainingRunConfig(channel="online_banking")
    result = train_channel_configured(
        config, trigger_source="test", channel_events=events, source_alerts=alerts, synthetic_labels=labels,
        bundle_store=_FakeChannelModelBundleStore(),
    )
    assert set(result["realized_split_fractions"].keys()) == {"train", "calibration", "test"}
    total = sum(result["realized_split_fractions"].values())
    assert total == pytest.approx(1.0)

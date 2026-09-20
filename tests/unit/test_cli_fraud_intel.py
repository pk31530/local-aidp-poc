"""Phase 6: CLI surfaces for fraud-intel/alerts. No database, Docker,
Kafka, MinIO, or real MLflow contact anywhere in this file -- every
handler's downstream store/orchestration call is monkeypatched, same
convention as tests/unit/test_cli.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.cli import __main__ as cli_main
from src.fraud_intel.alerts.queue import (
    AnalystDispositionRecord,
    FraudAlertRecord,
    InvalidAlertTransitionError,
    _FakeAlertQueueStore,
)
from src.fraud_intel.models.bundle import ChannelModelBundleRecord, IncompleteBundleError
from src.fraud_intel.models.promotion import BundlePromotionRaceError, BundleVerificationFailedError

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _alert(**overrides) -> FraudAlertRecord:
    import uuid

    base = dict(
        alert_id=1,
        event_id=uuid.uuid4(),
        source_alert_id=uuid.uuid4(),
        source_system="core_fraud_engine",
        channel="online_banking",
        customer_id="cust-1",
        account_id="acct-1",
        amount_minor_units=1000,
        initial_operational_priority_score=0.9,
        initial_priority_band="HIGH",
        initial_ensemble_policy_version="v1",
        status="OPEN",
        created_at=T0,
    )
    base.update(overrides)
    return FraudAlertRecord(**base)


def _bundle(**overrides) -> ChannelModelBundleRecord:
    base = dict(bundle_id=1, channel="online_banking", bundle_version=1, status="OPERATIONAL", created_at=T0)
    base.update(overrides)
    return ChannelModelBundleRecord(**base)


# ---- --database is required for every DB-touching command --------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["fraud-intel", "generate", "--channel", "online_banking", "--count", "5", "--json"],
        ["fraud-intel", "train", "--channel", "online_banking", "--json"],
        ["fraud-intel", "score", "--channel", "online_banking", "--json"],
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--json"],
        ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "a1", "--json"],
        ["fraud-intel", "model", "show", "--channel", "online_banking", "--json"],
        ["alerts", "list", "--json"],
        ["alerts", "show", "1", "--json"],
        ["alerts", "disposition", "1", "--analyst-id", "a1", "--disposition", "CONFIRMED_FRAUD", "--json"],
    ],
)
def test_missing_database_is_a_clean_cli_user_error(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(argv)
    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"
    assert "--database" in result["message"]


# ---- fraud-intel generate ------------------------------------------------------------


def test_generate_dispatches_to_generate_and_write(monkeypatch, capsys):
    captured = {}

    def _spy(*, channel, count, seed, database, reference_date):
        captured.update(channel=channel, count=count, seed=seed, database=database)
        return {"channel": channel, "count": count, "generation_run_id": "genrun-1", "dataset_version": "dsv-1"}

    monkeypatch.setattr(cli_main, "generate_and_write", _spy)

    cli_main.main(
        ["fraud-intel", "generate", "--channel", "online_banking", "--count", "5", "--seed", "7", "--database", "aidp_test", "--json"]
    )

    assert captured == {"channel": "online_banking", "count": 5, "seed": 7, "database": "aidp_test"}
    result = json.loads(capsys.readouterr().out)
    assert result["generation_run_id"] == "genrun-1"


def test_generate_value_error_is_a_cli_user_error(monkeypatch, capsys):
    def _raise(**kwargs):
        raise ValueError("unknown channel")

    monkeypatch.setattr(cli_main, "generate_and_write", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "generate", "--channel", "online_banking", "--count", "5", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- fraud-intel train ----------------------------------------------------------------


def test_train_loads_population_and_dispatches_to_train_channel_configured(monkeypatch, capsys):
    captured = {}

    def _fake_load_population(database):
        captured["database"] = database
        return [], [], []

    def _fake_train_channel_configured(config, *, trigger_source, channel_events, source_alerts, synthetic_labels, bundle_store):
        captured["config"] = config
        captured["trigger_source"] = trigger_source
        return {"bundle_id": 9, "status": "CANDIDATE"}

    fake_bundle_store = object()

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_online_banking_population", _fake_load_population)
    monkeypatch.setattr("src.fraud_intel.models.training.train_channel_configured", _fake_train_channel_configured)
    monkeypatch.setattr(cli_main, "create_default_bundle_store", lambda database: fake_bundle_store)

    cli_main.main(["fraud-intel", "train", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    assert captured["database"] == "aidp_test"
    assert captured["config"].channel == "online_banking"
    assert captured["trigger_source"] == "cli"
    result = json.loads(capsys.readouterr().out)
    assert result == {"bundle_id": 9, "status": "CANDIDATE"}


# ---- fraud-intel score: real reference-channel dispatch (Phase 6 corrective pass) -----


def test_score_dispatches_to_score_channel(monkeypatch, capsys):
    captured = {}

    def _fake_score_channel(*, channel, lifecycle, data_access, get_operational_bundle, artifact_loader, alert_queue_store):
        captured.update(
            channel=channel, lifecycle=lifecycle, data_access=data_access,
            get_operational_bundle=get_operational_bundle, artifact_loader=artifact_loader,
            alert_queue_store=alert_queue_store,
        )
        return {"run_id": 1, "channel": channel, "bundle_id": 2, "bundle_version": 1, "records_processed": 3, "records_rejected": 0, "alerts": []}

    fake_data_access = object()
    fake_artifact_loader = object()
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _fake_score_channel)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", lambda database: fake_data_access)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: fake_artifact_loader)
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(["fraud-intel", "score", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    assert captured["channel"] == "online_banking"
    assert captured["data_access"] is fake_data_access
    assert captured["artifact_loader"] is fake_artifact_loader
    assert captured["get_operational_bundle"]("online_banking") == _bundle()
    result = json.loads(capsys.readouterr().out)
    assert result == {"run_id": 1, "channel": "online_banking", "bundle_id": 2, "bundle_version": 1, "records_processed": 3, "records_rejected": 0, "alerts": []}


def test_score_unsupported_channel_is_a_cli_user_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "score", "--channel", "wire", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"
    assert "Phase 7A" in result["message"]


@pytest.mark.parametrize("exc_cls_path", [
    "src.fraud_intel.scoring.dispatch.NoOperationalBundleError",
    "src.fraud_intel.scoring.dispatch.BundlePolicyMismatchError",
])
def test_score_maps_domain_errors_to_cli_user_error(monkeypatch, capsys, exc_cls_path):
    import importlib

    module_path, cls_name = exc_cls_path.rsplit(".", 1)
    exc_cls = getattr(importlib.import_module(module_path), cls_name)

    def _raise(**kwargs):
        raise exc_cls("refused")

    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _raise)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: object())
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "score", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_score_clean_json_stdout_and_logs_go_to_stderr(monkeypatch, capsys):
    def _fake_score_channel(**kwargs):
        from src.common.logging import get_logger

        get_logger("fake.fraud_score").info("fake_fraud_score_started")
        return {"run_id": 1, "channel": "online_banking", "bundle_id": 2, "bundle_version": 1, "records_processed": 0, "records_rejected": 0, "alerts": []}

    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _fake_score_channel)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: object())
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(["fraud-intel", "score", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    captured = capsys.readouterr()
    stdout_lines = [line for line in captured.out.splitlines() if line]
    assert len(stdout_lines) == 1
    json.loads(stdout_lines[0])  # exactly one clean JSON object, nothing else
    assert "fake_fraud_score_started" not in captured.out
    assert "fake_fraud_score_started" in captured.err


# ---- Phase 7B readiness: no permanent-stub handler remains -------------------------------


def test_no_phase7b_required_handler_raises_not_implemented_error():
    """AST-based (not substring-based) so a docstring/comment that merely
    MENTIONS NotImplementedError -- e.g. explaining that a handler no
    longer raises it -- can never produce a false positive (the same class
    of false positive already hit twice elsewhere in Phase 6:
    tests/unit/test_fraud_intel_migration_004.py)."""
    import ast
    import inspect

    for handler in (
        cli_main._handle_fraud_intel_generate,
        cli_main._handle_fraud_intel_train,
        cli_main._handle_fraud_intel_score,
        cli_main._handle_fraud_intel_promote,
    ):
        tree = ast.parse(inspect.getsource(handler))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                func = node.exc.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                assert name != "NotImplementedError", f"{handler.__name__} still raises NotImplementedError"


# ---- fraud-intel evaluate: explicit Phase 6 scope marker -----------------------------


def test_evaluate_self_identifies_as_phase6_reference_channel_scope(capsys):
    cli_main.main(["fraud-intel", "evaluate", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result["scope"] == "reference_channel_phase6_only"
    assert result["channel"] == "online_banking"


# ---- fraud-intel promote ---------------------------------------------------------------


def test_promote_dispatches_and_returns_the_promoted_bundle(monkeypatch, capsys):
    fake_store = object()
    promoted = _bundle(bundle_id=2, bundle_version=1, promoted_by="analyst1", promoted_at=T0)
    captured = {}

    def _fake_promote_bundle(*, channel, bundle_version, promoted_by, model_version_verifier, store):
        captured.update(channel=channel, bundle_version=bundle_version, promoted_by=promoted_by, store=store)
        return promoted

    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: fake_store)
    monkeypatch.setattr("src.fraud_intel.models.promotion.promote_bundle", _fake_promote_bundle)

    cli_main.main(
        ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "analyst1", "--database", "aidp_test", "--json"]
    )

    assert captured["channel"] == "online_banking"
    assert captured["bundle_version"] == 1
    assert captured["promoted_by"] == "analyst1"
    assert captured["store"] is fake_store
    result = json.loads(capsys.readouterr().out)
    assert result["bundle_id"] == 2
    assert result["status"] == "OPERATIONAL"


@pytest.mark.parametrize("exc_cls", [IncompleteBundleError, BundleVerificationFailedError, BundlePromotionRaceError])
def test_promote_maps_domain_errors_to_cli_user_error(monkeypatch, capsys, exc_cls):
    def _raise(**kwargs):
        raise exc_cls("refused")

    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.models.promotion.promote_bundle", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "analyst1", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- fraud-intel model show -------------------------------------------------------------


def test_model_show_returns_the_operational_bundle(monkeypatch, capsys):
    bundle = _bundle()
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: bundle)

    cli_main.main(["fraud-intel", "model", "show", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["bundle_id"] == 1
    assert result["status"] == "OPERATIONAL"


def test_model_show_no_operational_bundle_is_a_cli_user_error(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: None)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "model", "show", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- alerts list: analyst-facing surface structurally excludes synthetic labels --------


def test_alerts_list_output_has_no_scenario_id_or_synthetic_fields(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_list_alerts", lambda database, *, channel, status, priority_band: [_alert()])

    cli_main.main(["alerts", "list", "--database", "aidp_test", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 1
    assert "scenario_id" not in result["alerts"][0]
    assert "synthetic_scenario_label" not in result["alerts"][0]


def test_alerts_list_passes_filters_through(monkeypatch, capsys):
    captured = {}

    def _fake_list(database, *, channel, status, priority_band):
        captured.update(database=database, channel=channel, status=status, priority_band=priority_band)
        return []

    monkeypatch.setattr(cli_main, "_list_alerts", _fake_list)

    cli_main.main(
        ["alerts", "list", "--channel", "online_banking", "--status", "OPEN", "--priority-band", "HIGH", "--database", "aidp_test", "--json"]
    )

    assert captured == {"database": "aidp_test", "channel": "online_banking", "status": "OPEN", "priority_band": "HIGH"}


# ---- alerts show ------------------------------------------------------------------------


def test_alerts_show_returns_alert_and_latest_evidence(monkeypatch, capsys):
    store = _FakeAlertQueueStore()
    alert = _alert()
    store.alerts_by_id[1] = alert

    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: store)

    cli_main.main(["alerts", "show", "1", "--database", "aidp_test", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["alert"]["alert_id"] == 1
    assert result["latest_evidence"] is None


def test_alerts_show_unknown_alert_is_a_cli_user_error(monkeypatch, capsys):
    store = _FakeAlertQueueStore()
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: store)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["alerts", "show", "999", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- alerts disposition -------------------------------------------------------------------


def test_alerts_disposition_dispatches_to_record_disposition(monkeypatch, capsys):
    fake_store = object()
    captured = {}
    record = AnalystDispositionRecord(
        disposition_id=1, alert_id=1, analyst_id="a1", disposition="CONFIRMED_FRAUD", notes=None, disposed_at=T0
    )

    def _fake_record_disposition(*, alert_id, analyst_id, disposition, notes, store):
        captured.update(alert_id=alert_id, analyst_id=analyst_id, disposition=disposition, notes=notes, store=store)
        return record

    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: fake_store)
    monkeypatch.setattr(cli_main, "record_disposition", _fake_record_disposition)

    cli_main.main(
        ["alerts", "disposition", "1", "--analyst-id", "a1", "--disposition", "CONFIRMED_FRAUD", "--notes", "looks fraudulent", "--database", "aidp_test", "--json"]
    )

    assert captured["alert_id"] == 1
    assert captured["analyst_id"] == "a1"
    assert captured["disposition"] == "CONFIRMED_FRAUD"
    assert captured["notes"] == "looks fraudulent"
    assert captured["store"] is fake_store
    result = json.loads(capsys.readouterr().out)
    assert result["disposition"] == "CONFIRMED_FRAUD"


def test_alerts_disposition_against_a_closed_alert_is_a_cli_user_error(monkeypatch, capsys):
    def _raise(*, alert_id, analyst_id, disposition, notes, store):
        raise InvalidAlertTransitionError(f"alert {alert_id} is CLOSED (terminal); disposition rejected")

    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "record_disposition", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["alerts", "disposition", "1", "--analyst-id", "a1", "--disposition", "CONFIRMED_FRAUD", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"
    assert "CLOSED" in result["message"]

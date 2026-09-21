"""Phase 6: CLI surfaces for fraud-intel/alerts. No database, Docker,
Kafka, MinIO, or real MLflow contact anywhere in this file -- every
handler's downstream store/orchestration call is monkeypatched, same
convention as tests/unit/test_cli.py.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from src.cli import __main__ as cli_main
from src.fraud_intel.alerts.queue import (
    AnalystDispositionRecord,
    FraudAlertRecord,
    InvalidAlertTransitionError,
    _FakeAlertQueueStore,
)
from src.fraud_intel.models.bundle import ChannelModelBundleRecord, IncompleteBundleError
from src.fraud_intel.models.promotion import (
    BundlePromotionRaceError,
    BundleVerificationFailedError,
    ColdStartPromotionGateFailedError,
)

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
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "10", "--recall-target", "0.8", "--json"],
        ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "a1", "--json"],
        ["fraud-intel", "model", "show", "--channel", "online_banking", "--json"],
        ["fraud-intel", "labels", "assess", "--channel", "online_banking", "--json"],
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
        captured.update(channel=channel, count=count, seed=seed, database=database, reference_date=reference_date)
        return {"channel": channel, "requested_count": count, "generation_run_id": "genrun-1", "dataset_version": "dsv-1"}

    monkeypatch.setattr(cli_main, "generate_and_write", _spy)

    cli_main.main(
        ["fraud-intel", "generate", "--channel", "online_banking", "--count", "5", "--seed", "7",
         "--reference-date", "2026-01-01", "--database", "aidp_test", "--json"]
    )

    assert captured == {
        "channel": "online_banking", "count": 5, "seed": 7, "database": "aidp_test",
        "reference_date": date(2026, 1, 1),
    }
    result = json.loads(capsys.readouterr().out)
    assert result["generation_run_id"] == "genrun-1"


def test_generate_reference_date_defaults_to_today_when_omitted(monkeypatch, capsys):
    captured = {}

    def _spy(*, channel, count, seed, database, reference_date):
        captured["reference_date"] = reference_date
        return {"channel": channel, "requested_count": count, "generation_run_id": "genrun-1", "dataset_version": "dsv-1"}

    monkeypatch.setattr(cli_main, "generate_and_write", _spy)

    cli_main.main(["fraud-intel", "generate", "--channel", "online_banking", "--count", "5", "--database", "aidp_test", "--json"])

    assert captured["reference_date"] == date.today()


def test_generate_invalid_reference_date_is_a_cli_user_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "generate", "--channel", "online_banking", "--count", "5",
             "--reference-date", "not-a-date", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"
    assert "--reference-date" in result["message"]


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

    def _fake_load_population(channel, database, *, generation_run_id):
        captured["channel"] = channel
        captured["database"] = database
        captured["generation_run_id"] = generation_run_id
        return [], [], []

    def _fake_train_channel_configured(
        config, *, database, trigger_source, channel_events, source_alerts, synthetic_labels, bundle_store,
        rule_set_version, graph_policy_version, ensemble_policy_version, reason_code_version,
    ):
        captured["config"] = config
        captured["train_database"] = database
        captured["trigger_source"] = trigger_source
        captured["rule_set_version"] = rule_set_version
        captured["graph_policy_version"] = graph_policy_version
        captured["ensemble_policy_version"] = ensemble_policy_version
        captured["reason_code_version"] = reason_code_version
        return {"bundle_id": 9, "status": "CANDIDATE"}

    fake_bundle_store = object()

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_channel_population", _fake_load_population)
    monkeypatch.setattr("src.fraud_intel.models.training.train_channel_configured", _fake_train_channel_configured)
    monkeypatch.setattr(cli_main, "create_default_bundle_store", lambda database: fake_bundle_store)

    cli_main.main(
        ["fraud-intel", "train", "--channel", "online_banking", "--generation-run-id", "genrun-abc",
         "--database", "aidp_test", "--json"]
    )

    assert captured["channel"] == "online_banking"
    assert captured["database"] == "aidp_test"
    assert captured["train_database"] == "aidp_test"  # load_channel_population() and train_channel_configured() share the same database
    assert captured["generation_run_id"] == "genrun-abc"
    assert captured["config"].channel == "online_banking"
    assert captured["trigger_source"] == "cli"
    # Phase 7B Stage 3 corrective pass: the CLI loads the channel's real,
    # current policy versions (config/fraud_intel/*_online_banking.yaml)
    # and passes them all through -- the resulting candidate is complete,
    # not the old Phase-4-style deliberately-incomplete bundle.
    assert captured["rule_set_version"] == "v1"
    assert captured["graph_policy_version"] == "v1"
    assert captured["ensemble_policy_version"] == "v1"
    assert captured["reason_code_version"] == "v1"
    result = json.loads(capsys.readouterr().out)
    assert result == {"bundle_id": 9, "status": "CANDIDATE"}


def test_train_missing_generation_run_id_is_a_cli_user_error(monkeypatch, capsys):
    """Training must never silently fall back to loading every row ever
    generated for a channel -- --generation-run-id is required."""
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_channel_population", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "train", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"
    assert "--generation-run-id" in result["message"]


@pytest.mark.parametrize(
    "exc_cls_path",
    [
        "src.fraud_intel.cli_data_access.UnknownGenerationRunError",
        "src.fraud_intel.cli_data_access.GenerationRunChannelMismatchError",
        "src.fraud_intel.cli_data_access.GenerationRunDatasetVersionError",
    ],
)
def test_train_generation_run_domain_errors_are_cli_user_errors(monkeypatch, capsys, exc_cls_path):
    import importlib

    module_path, _, cls_name = exc_cls_path.rpartition(".")
    exc_cls = getattr(importlib.import_module(module_path), cls_name)

    def _raise(channel, database, *, generation_run_id):
        raise exc_cls("simulated")

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_channel_population", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "train", "--channel", "online_banking", "--generation-run-id", "genrun-abc",
             "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


@pytest.mark.parametrize("channel", sorted({"ach", "wire", "mobile_deposit", "atm", "debit_card", "p2p"}))
def test_train_accepts_every_phase7a_registered_channel(monkeypatch, capsys, channel):
    """Phase 7A: FRAUD_INTEL_IMPLEMENTED_CHANNELS is widened to all 7 --
    every non-reference channel now reaches the shared training path
    instead of being rejected by _require_implemented_channel."""
    def _fake_load_population(channel, database, *, generation_run_id):
        return [], [], []

    def _fake_train_channel_configured(
        config, *, database, trigger_source, channel_events, source_alerts, synthetic_labels, bundle_store,
        rule_set_version, graph_policy_version, ensemble_policy_version, reason_code_version,
    ):
        return {"bundle_id": 9, "status": "CANDIDATE", "channel": config.channel}

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_channel_population", _fake_load_population)
    monkeypatch.setattr("src.fraud_intel.models.training.train_channel_configured", _fake_train_channel_configured)
    monkeypatch.setattr(cli_main, "create_default_bundle_store", lambda database: object())

    cli_main.main(
        ["fraud-intel", "train", "--channel", channel, "--generation-run-id", "genrun-abc", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["channel"] == channel


def test_train_policy_version_load_failure_is_a_cli_user_error(monkeypatch, capsys):
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_channel_population", lambda channel, database, *, generation_run_id: ([], [], []))
    monkeypatch.setattr(
        "src.fraud_intel.rules.provider._load_rule_set_config",
        lambda channel: (_ for _ in ()).throw(FileNotFoundError("no such rules file")),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "train", "--channel", "online_banking", "--generation-run-id", "genrun-abc",
             "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- fraud-intel score: real reference-channel dispatch (Phase 6 corrective pass) -----


def test_score_dispatches_to_score_channel(monkeypatch, capsys):
    captured = {}

    def _fake_score_channel(*, channel, generation_run_id, lifecycle, data_access, get_operational_bundle, artifact_loader, alert_queue_store):
        captured.update(
            channel=channel, generation_run_id=generation_run_id, lifecycle=lifecycle, data_access=data_access,
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

    cli_main.main(
        ["fraud-intel", "score", "--channel", "online_banking", "--generation-run-id", "genrun-1",
         "--database", "aidp_test", "--json"]
    )

    assert captured["channel"] == "online_banking"
    assert captured["generation_run_id"] == "genrun-1"
    assert captured["data_access"] is fake_data_access
    assert captured["artifact_loader"] is fake_artifact_loader
    assert captured["get_operational_bundle"]("online_banking") == _bundle()
    result = json.loads(capsys.readouterr().out)
    assert result == {"run_id": 1, "channel": "online_banking", "bundle_id": 2, "bundle_version": 1, "records_processed": 3, "records_rejected": 0, "alerts": []}


def test_score_missing_generation_run_id_is_a_cli_user_error(monkeypatch, capsys):
    """Scoring must never silently score every pending alert for a
    channel across every generation ever run -- --generation-run-id is
    required, exactly like train's own contract."""
    monkeypatch.setattr(
        "src.fraud_intel.scoring.dispatch.score_channel",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "score", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"
    assert "--generation-run-id" in result["message"]


def test_score_lifecycle_and_data_access_share_the_same_explicit_database(monkeypatch, capsys):
    captured = {}

    def _fake_score_channel(*, channel, generation_run_id, lifecycle, data_access, get_operational_bundle, artifact_loader, alert_queue_store):
        captured["lifecycle_database"] = lifecycle._store._database if hasattr(lifecycle._store, "_database") else None
        return {"run_id": 1, "channel": channel, "bundle_id": 2, "bundle_version": 1, "records_processed": 0, "records_rejected": 0, "alerts": []}

    def _fake_create_default_scoring_data_access(database):
        captured["data_access_database"] = database
        return object()

    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _fake_score_channel)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", _fake_create_default_scoring_data_access)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: object())
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(
        ["fraud-intel", "score", "--channel", "online_banking", "--generation-run-id", "genrun-1",
         "--database", "aidp_test", "--json"]
    )

    assert captured["data_access_database"] == "aidp_test"
    assert captured["lifecycle_database"] == "aidp_test"
    assert captured["data_access_database"] == captured["lifecycle_database"]


def test_score_accepts_a_non_reference_channel_now_that_all_seven_are_registered(monkeypatch, capsys):
    """Phase 7A: 'wire' used to be rejected by _require_implemented_channel
    (Phase 6 corrective pass, reference channel only) -- FRAUD_INTEL_IMPLEMENTED_CHANNELS
    is now widened to all 7 registered channels, so this reaches score_channel()."""
    def _fake_score_channel(*, channel, **kwargs):
        return {"run_id": 1, "channel": channel, "bundle_id": 1, "bundle_version": 1, "records_processed": 0, "records_rejected": 0, "alerts": []}

    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _fake_score_channel)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: object())
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(
        ["fraud-intel", "score", "--channel", "wire", "--generation-run-id", "genrun-1", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["channel"] == "wire"


def test_score_unregistered_channel_name_is_rejected_at_the_argparse_level(capsys):
    """A genuinely unsupported/unknown channel name (not one of the 7
    registered channels at all) still fails cleanly -- argparse's own
    --channel choices reject it before any handler runs, same as every
    other fixed-choice CLI argument in this codebase (--status,
    --priority-band, ...)."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "score", "--channel", "not_a_real_channel", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("exc_cls_path", [
    "src.fraud_intel.scoring.dispatch.NoOperationalBundleError",
    "src.fraud_intel.scoring.dispatch.BundlePolicyMismatchError",
    "src.fraud_intel.cli_data_access.UnknownGenerationRunError",
    "src.fraud_intel.cli_data_access.GenerationRunChannelMismatchError",
    "src.fraud_intel.cli_data_access.GenerationRunDatasetVersionError",
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
        cli_main.main(
            ["fraud-intel", "score", "--channel", "online_banking", "--generation-run-id", "genrun-1",
             "--database", "aidp_test", "--json"]
        )
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

    cli_main.main(
        ["fraud-intel", "score", "--channel", "online_banking", "--generation-run-id", "genrun-1",
         "--database", "aidp_test", "--json"]
    )

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
        cli_main._handle_fraud_intel_evaluate,
        cli_main._evaluate_candidate_cold_start,
        cli_main._evaluate_live,
        cli_main._handle_fraud_intel_promote,
        cli_main._handle_fraud_intel_model_show,
        cli_main._handle_fraud_intel_labels_assess,
    ):
        tree = ast.parse(inspect.getsource(handler))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                func = node.exc.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                assert name != "NotImplementedError", f"{handler.__name__} still raises NotImplementedError"


def test_no_phase7b_required_handler_returns_a_phase_scope_placeholder_dict():
    """The old Phase 6 evaluate() stub returned a literal dict containing
    a "scope"/"reference_channel_phaseN_only" placeholder marker instead
    of doing real work -- proves no Phase 7B-required handler still
    contains that pattern (a dict literal with a 'scope' key) anywhere in
    its own source."""
    import ast
    import inspect

    for handler in (
        cli_main._handle_fraud_intel_generate,
        cli_main._handle_fraud_intel_train,
        cli_main._handle_fraud_intel_score,
        cli_main._handle_fraud_intel_evaluate,
        cli_main._evaluate_candidate_cold_start,
        cli_main._evaluate_live,
        cli_main._handle_fraud_intel_promote,
        cli_main._handle_fraud_intel_model_show,
        cli_main._handle_fraud_intel_labels_assess,
    ):
        tree = ast.parse(inspect.getsource(handler))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                assert "scope" not in keys, f"{handler.__name__} still returns a 'scope' placeholder dict: {keys}"


# ---- fraud-intel evaluate: real Phase 7A evaluation (corrective pass) -----------------


def _outcome_fixtures():
    import uuid

    from src.fraud_intel.evaluation.cross_channel import AlertOutcome

    ids = [uuid.uuid4() for _ in range(4)]
    labels = ["RESOLVED_FRAUD", "RESOLVED_FRAUD", "RESOLVED_LEGITIMATE", "RESOLVED_LEGITIMATE"]
    scores = [0.9, 0.7, 0.3, 0.1]
    bands = ["HIGH", "MEDIUM", "LOW", "LOW"]
    return [
        AlertOutcome(
            source_alert_id=ids[i], channel="online_banking", event_timestamp=T0, operational_priority_score=scores[i],
            baseline_priority_score=scores[i], priority_band=bands[i], resolved_label=labels[i],
        )
        for i in range(4)
    ]


def test_evaluate_returns_operational_and_baseline_sections_without_a_candidate(monkeypatch, capsys):
    outcomes = _outcome_fixtures()
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2", "--recall-target", "0.8", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["channel"] == "online_banking"
    assert result["operational_bundle"] == {"bundle_id": _bundle().bundle_id, "bundle_version": _bundle().bundle_version}
    assert "rules_only_baseline" not in result["operational_evaluation"]  # clearly separated out, not nested
    assert result["rules_only_baseline"]["pr_auc"]["status"] in ("ok", "non_computable_single_class", "non_computable_empty")
    assert result["shadow_candidate_comparison"] is None
    assert result["operational_evaluation"]["total_source_alerts"] == 4


def test_evaluate_includes_shadow_candidate_comparison_when_requested(monkeypatch, capsys):
    """CLI-level test: mocks the real I/O boundaries (bundle lookup,
    pinned-policy loading, artifact loading, population loading, and the
    actual in-memory scoring call) and lets the CLI's own orchestration
    and the real compare_operational_vs_candidate() run."""
    from src.fraud_intel.evaluation.shadow_candidate import CandidateShadowScore

    outcomes = _outcome_fixtures()
    candidate_scores = [
        CandidateShadowScore(
            source_alert_id=o.source_alert_id, channel="online_banking", event_timestamp=T0,
            candidate_priority_score=o.operational_priority_score, candidate_priority_band=o.priority_band,
        )
        for o in outcomes
    ]
    candidate_bundle = _bundle(bundle_id=5, bundle_version=3)

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            assert channel == "online_banking"
            assert bundle_version == 3
            return candidate_bundle

    class _FakeArtifactLoader:
        def load(self, bundle_record):
            return "fake-loaded-candidate-bundle"

    def _fake_score_candidate_shadow(scoring_inputs, *, bundle, rule_provider, ensemble_policy, graph_policy, **kwargs):
        assert bundle == "fake-loaded-candidate-bundle"
        return candidate_scores, []

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_scoring_contexts", lambda channel, database: [])
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.load_and_validate_pinned_policies", lambda bundle_record: (None, None, None))
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: _FakeArtifactLoader())
    monkeypatch.setattr("src.fraud_intel.evaluation.shadow_candidate.score_candidate_shadow", _fake_score_candidate_shadow)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
         "--recall-target", "0.8", "--candidate-bundle-version", "3", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["shadow_candidate_comparison"] is not None
    assert result["shadow_candidate_comparison"]["candidate_bundle_id"] == 5
    assert result["shadow_candidate_comparison"]["candidate_bundle_version"] == 3
    assert result["shadow_candidate_comparison"]["matched_source_alert_count"] == 4
    assert result["candidate_scoring_errors"] == []


def test_evaluate_surfaces_candidate_scoring_errors_without_failing_the_command(monkeypatch, capsys):
    from src.fraud_intel.evaluation.shadow_candidate import CandidateShadowScoringError

    outcomes = _outcome_fixtures()
    candidate_bundle = _bundle(bundle_id=5, bundle_version=3)
    scoring_error = CandidateShadowScoringError(source_alert_id=outcomes[0].source_alert_id, error_type="ValueError", message="simulated per-alert failure")

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            return candidate_bundle

    class _FakeArtifactLoader:
        def load(self, bundle_record):
            return "fake-loaded-candidate-bundle"

    def _fake_score_candidate_shadow(scoring_inputs, *, bundle, rule_provider, ensemble_policy, graph_policy, **kwargs):
        return [], [scoring_error]

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_scoring_contexts", lambda channel, database: [])
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.load_and_validate_pinned_policies", lambda bundle_record: (None, None, None))
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: _FakeArtifactLoader())
    monkeypatch.setattr("src.fraud_intel.evaluation.shadow_candidate.score_candidate_shadow", _fake_score_candidate_shadow)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
         "--recall-target", "0.8", "--candidate-bundle-version", "3", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert len(result["candidate_scoring_errors"]) == 1
    assert result["candidate_scoring_errors"][0]["error_type"] == "ValueError"
    # zero candidate scores -> comparison still returns cleanly, non-computable rather than crashing
    assert result["shadow_candidate_comparison"] is not None


def test_evaluate_unknown_candidate_bundle_version_is_a_cli_user_error(monkeypatch, capsys):
    outcomes = _outcome_fixtures()

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            raise LookupError(f"no bundle version {bundle_version} for channel {channel!r}")

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
             "--recall-target", "0.8", "--candidate-bundle-version", "999", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_evaluate_no_operational_bundle_is_a_cli_user_error(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: None)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2", "--recall-target", "0.8", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_evaluate_missing_required_capacity_or_recall_flags_fails_at_argparse_level(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "evaluate", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2


def test_evaluate_invalid_capacity_value_is_a_cli_user_error(monkeypatch, capsys):
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: [])
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "-5",
             "--recall-target", "0.8", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_evaluate_scope_marker_no_longer_present(monkeypatch, capsys):
    """The removed Phase 6 placeholder marker must not reappear anywhere
    in real output."""
    outcomes = _outcome_fixtures()
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2", "--recall-target", "0.8", "--database", "aidp_test", "--json"]
    )
    out = capsys.readouterr().out
    assert "reference_channel_phase6_only" not in out


# ---- fraud-intel evaluate: Phase 7B Stage 0 cold-start / live modes -------------------


def _cold_start_report_dict(bundle: ChannelModelBundleRecord, **overrides) -> dict:
    report = dict(
        channel=bundle.channel, training_run_id=bundle.training_run_id, dataset_version=bundle.dataset_version,
        supervised_population_hash=bundle.dataset_version,
        source_generation_run_id="genrun-3e8516ce803e5d82", source_dataset_version="dsv-3e8516ce803e5d82",
        feature_schema_version=bundle.feature_schema_version, gbm_model_version=bundle.gbm_model_version,
        lr_model_version=bundle.lr_model_version, anomaly_model_version=bundle.anomaly_model_version,
        preprocessing_artifact_version=bundle.preprocessing_artifact_version,
        gbm_evaluation={"precision": 0.9, "recall": 0.8, "pr_auc": 0.9, "roc_auc": 0.95, "brier_score": 0.05},
        lr_shadow_evaluation={"precision": 0.7, "recall": 0.6, "pr_auc": 0.7, "roc_auc": 0.8, "brier_score": 0.1},
        eligibility_policy_version="v1", realized_split_fractions={"train": 0.6, "calibration": 0.2, "test": 0.2},
        split_class_counts={
            "train": {"fraud": 10, "legitimate": 10}, "calibration": {"fraud": 10, "legitimate": 10},
            "test": {"fraud": 10, "legitimate": 10},
        },
        gbm_mlflow_run_id="run-gbm-1", lr_mlflow_run_id="run-lr-1", anomaly_mlflow_run_id="run-anomaly-1",
        anomaly_normalization={"method": "none"},
    )
    report.update(overrides)
    return report


def _cold_start_candidate_bundle(**overrides) -> ChannelModelBundleRecord:
    base = dict(
        bundle_id=9, channel="online_banking", bundle_version=4, status="CANDIDATE", created_at=T0,
        gbm_model_version="gbm-9", lr_model_version="lr-9", anomaly_model_version="anomaly-9",
        preprocessing_artifact_version="prep-9", feature_schema_version="fsv-9",
        training_run_id=42, dataset_version="dsv-9",
    )
    base.update(overrides)
    bundle = _bundle(**base)
    import json as _json

    return bundle.model_copy(update={"evaluation_report_ref": _json.dumps(_cold_start_report_dict(bundle))})


def test_evaluate_candidate_only_cold_start_when_no_operational_bundle_exists(monkeypatch, capsys):
    candidate_bundle = _cold_start_candidate_bundle()

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            assert (channel, bundle_version) == ("online_banking", 4)
            return candidate_bundle

    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: None)
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
         "--recall-target", "0.8", "--candidate-bundle-version", "4", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["evaluation_mode"] == "candidate_training_holdout"
    assert result["operational_bundle"] is None
    assert result["shadow_candidate_comparison"] is None
    assert result["candidate_bundle"] == {"bundle_id": 9, "bundle_version": 4}
    assert result["candidate_evaluation"]["training_run_id"] == 42
    assert result["rules_only_baseline"]["test_fraud_prevalence"] == 0.5
    assert result["promotion_gate_result"]["passed"] is True
    assert "NOT live operational evaluation" in result["disclaimer"]


def test_evaluate_cold_start_report_bundle_mismatch_blocks_evaluation(monkeypatch, capsys):
    """A tampered/stale evaluation_report_ref that disagrees with its own
    bundle row must fail evaluation, never silently proceed -- promotion
    stays blocked because no promotion_gate_result is ever produced."""
    import json as _json

    mismatched_bundle = _cold_start_candidate_bundle()
    tampered_report = _cold_start_report_dict(mismatched_bundle, dataset_version="dsv-DIFFERENT")
    mismatched_bundle = mismatched_bundle.model_copy(update={"evaluation_report_ref": _json.dumps(tampered_report)})

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            return mismatched_bundle

    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: None)
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
             "--recall-target", "0.8", "--candidate-bundle-version", "4", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_evaluate_cold_start_missing_report_blocks_evaluation(monkeypatch, capsys):
    candidate_bundle = _cold_start_candidate_bundle().model_copy(update={"evaluation_report_ref": None})

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            return candidate_bundle

    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: None)
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
             "--recall-target", "0.8", "--candidate-bundle-version", "4", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_evaluate_candidate_cold_start_never_promotes():
    """AST-based (Phase 7A convention, avoids docstring-substring false
    positives): _evaluate_candidate_cold_start's own source never calls
    anything named promote_bundle."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cli_main._evaluate_candidate_cold_start))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            assert name != "promote_bundle"


def test_evaluate_live_mode_with_no_eligible_resolved_alerts_is_non_computable_not_a_crash(monkeypatch, capsys):
    """Live evaluation's population is intrinsically gated on eligible,
    RESOLVED label_assessments (load_resolved_alert_outcomes' own WHERE
    clause) -- when none exist yet, evaluation still runs cleanly under
    evaluation_mode=live_resolved_alerts rather than crashing or
    fabricating a result."""
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: [])
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
         "--recall-target", "0.8", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["evaluation_mode"] == "live_resolved_alerts"
    assert result["operational_evaluation"]["total_source_alerts"] == 0
    assert result["candidate_bundle"] is None
    assert result["candidate_evaluation"] is None
    assert result["promotion_gate_result"] is None


# ---- fraud-intel promote ---------------------------------------------------------------


def test_promote_dispatches_and_returns_the_promoted_bundle(monkeypatch, capsys):
    fake_store = object()
    promoted = _bundle(bundle_id=2, bundle_version=1, promoted_by="analyst1", promoted_at=T0)
    captured = {}

    def _fake_create_default_bundle_promotion_store(database):
        captured["store_database"] = database
        return fake_store

    def _fake_promote_bundle(*, channel, bundle_version, promoted_by, database, model_version_verifier, store):
        captured.update(channel=channel, bundle_version=bundle_version, promoted_by=promoted_by, database=database, store=store)
        return promoted

    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", _fake_create_default_bundle_promotion_store)
    monkeypatch.setattr("src.fraud_intel.models.promotion.promote_bundle", _fake_promote_bundle)

    cli_main.main(
        ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "analyst1", "--database", "aidp_test", "--json"]
    )

    assert captured["channel"] == "online_banking"
    assert captured["bundle_version"] == 1
    assert captured["promoted_by"] == "analyst1"
    assert captured["store"] is fake_store
    # Phase 7B Stage 5: the promotion store and promote_bundle()'s own
    # RunLifecycle must be given the SAME explicit database -- there is
    # only one `database` local variable in the handler, used for both.
    assert captured["database"] == "aidp_test"
    assert captured["store_database"] == "aidp_test"
    assert captured["database"] == captured["store_database"]
    result = json.loads(capsys.readouterr().out)
    assert result["bundle_id"] == 2
    assert result["status"] == "OPERATIONAL"


@pytest.mark.parametrize(
    "exc_cls",
    [IncompleteBundleError, BundleVerificationFailedError, BundlePromotionRaceError, ColdStartPromotionGateFailedError],
)
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


def test_promote_maps_cold_start_report_error_to_cli_user_error(monkeypatch, capsys):
    from src.fraud_intel.evaluation.cold_start import ColdStartReportError

    def _raise(**kwargs):
        raise ColdStartReportError("bundle's evaluation_report_ref does not match its own bundle row")

    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.models.promotion.promote_bundle", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "analyst1", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- _MlflowModelVersionVerifier: Phase 7B Stage 5 configure_mlflow() corrective pass ----


class _FakeModelVersion:
    def __init__(self, *, status="READY", run_id="run-1"):
        self.status = status
        self.run_id = run_id


def test_verifier_calls_configure_mlflow_before_constructing_mlflow_client(monkeypatch):
    call_order = []
    monkeypatch.setattr("src.common.mlflow_setup.configure_mlflow", lambda: call_order.append("configure"))

    class _FakeClient:
        def __init__(self):
            call_order.append("client_constructed")

        def get_model_version(self, model_name, version):
            call_order.append("get_model_version")
            return _FakeModelVersion(run_id="run-1")

    import mlflow

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", _FakeClient)

    verifier = cli_main._MlflowModelVersionVerifier()
    result = verifier.verify("some-model", "1", expected_run_id="run-1")

    assert result is True
    assert call_order == ["configure", "client_constructed", "get_model_version"]


def test_verifier_configures_mlflow_on_every_call_never_relying_on_ambient_state(monkeypatch):
    """Proves the verifier does not skip configuration based on some
    cached/ambient 'already configured' flag -- every single call
    independently configures MLflow first, exactly as a genuinely fresh
    process (no other command has run in it yet) would need."""
    configure_calls = []
    monkeypatch.setattr("src.common.mlflow_setup.configure_mlflow", lambda: configure_calls.append(1))

    class _FakeClient:
        def get_model_version(self, model_name, version):
            return _FakeModelVersion(run_id="run-1")

    import mlflow

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", _FakeClient)

    verifier = cli_main._MlflowModelVersionVerifier()
    verifier.verify("some-model", "1", expected_run_id="run-1")
    verifier.verify("some-model", "1", expected_run_id="run-1")

    assert len(configure_calls) == 2


def test_verifier_passes_for_correct_name_version_and_run_id(monkeypatch):
    monkeypatch.setattr("src.common.mlflow_setup.configure_mlflow", lambda: None)

    class _FakeClient:
        def get_model_version(self, model_name, version):
            assert model_name == "fraud-detection-model-fraud-intel-online-banking-gbm"
            assert version == "2"
            return _FakeModelVersion(status="READY", run_id="run-gbm-1")

    import mlflow

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", _FakeClient)

    verifier = cli_main._MlflowModelVersionVerifier()
    assert verifier.verify(
        "fraud-detection-model-fraud-intel-online-banking-gbm", "2", expected_run_id="run-gbm-1"
    ) is True


@pytest.mark.parametrize(
    "returned_version,expected_run_id",
    [
        (_FakeModelVersion(status="READY", run_id="run-WRONG"), "run-gbm-1"),  # wrong run_id
        (_FakeModelVersion(status="PENDING_REGISTRATION", run_id="run-gbm-1"), "run-gbm-1"),  # not READY
    ],
)
def test_verifier_fails_for_wrong_run_id_or_non_ready_status(monkeypatch, returned_version, expected_run_id):
    monkeypatch.setattr("src.common.mlflow_setup.configure_mlflow", lambda: None)

    class _FakeClient:
        def get_model_version(self, model_name, version):
            return returned_version

    import mlflow

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", _FakeClient)

    verifier = cli_main._MlflowModelVersionVerifier()
    assert verifier.verify("some-model", "2", expected_run_id=expected_run_id) is False


def test_configure_mlflow_failure_propagates_and_is_caught_as_a_verification_problem(monkeypatch):
    """A configure_mlflow() failure must not be silently swallowed by the
    verifier itself -- it propagates up to verify_bundle_components(),
    which is the layer responsible for turning it into a reported
    problem (and therefore a refused promotion)."""
    from src.fraud_intel.models.bundle import _FakeChannelModelBundleStore
    from src.fraud_intel.models.promotion import verify_bundle_components

    def _raise():
        raise ConnectionError("simulated MLflow tracking server unreachable")

    monkeypatch.setattr("src.common.mlflow_setup.configure_mlflow", _raise)

    store = _FakeChannelModelBundleStore()
    bundle = store.register_candidate(
        channel="online_banking", gbm_model_version="2", lr_model_version="2", anomaly_model_version="2",
        preprocessing_artifact_version="pp-1", feature_schema_version="v1", rule_set_version="v1",
        graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
        training_run_id=5, dataset_version="cc49865cde175619", evaluation_report_ref="{}",
    )

    problems = verify_bundle_components(bundle, model_version_verifier=cli_main._MlflowModelVersionVerifier())
    assert len(problems) == 3  # gbm, lr-shadow, anomaly all fail the same way
    for p in problems:
        assert "verification raised ConnectionError" in p


def test_verifier_source_never_registers_aliases_tags_or_loads_model_weights():
    """AST-based (Phase 7A convention): verify()'s own source never calls
    anything that would register a model, set an alias/tag, log an
    artifact, or load model weights -- metadata-only, read-only."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cli_main._MlflowModelVersionVerifier.verify)))
    forbidden = {
        "set_registered_model_alias", "set_model_version_tag", "set_registered_model_tag",
        "transition_model_version_stage", "delete_model_version", "create_model_version",
        "log_model", "load_model", "register_model",
    }
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    assert not (identifiers & forbidden), f"verify() must stay read-only, found: {identifiers & forbidden}"


def test_failed_verification_through_promote_bundle_produces_a_failed_model_promotion_run(monkeypatch):
    """End-to-end (still no real MLflow/Postgres): promote_bundle() using
    the REAL _MlflowModelVersionVerifier, whose configure_mlflow() is
    made to fail -- proves the resulting BundleVerificationFailedError
    still produces a FAILED model_promotion RunLifecycle record with the
    original error_type preserved, and never mutates the candidate's
    status."""
    from datetime import datetime, timezone as tz

    from src.control_plane.runs import RunLifecycle as RealRunLifecycle
    from src.control_plane.runs import RunRecord
    from src.fraud_intel.models import promotion as promotion_module
    from src.fraud_intel.models.bundle import _FakeChannelModelBundleStore
    from src.fraud_intel.models.promotion import BundleVerificationFailedError, _FakeBundlePromotionStore, promote_bundle

    class _FakeRunStore:
        def __init__(self):
            self.rows: dict[int, dict] = {}
            self._next_id = 1

        def insert(self, row):
            run_id = self._next_id
            self._next_id += 1
            full = {
                "run_id": run_id, "trigger_source": None, "git_sha": None, "config_snapshot": None,
                "config_hash": None, "dataset_version": None, "model_version": None, "error_type": None,
                "error_message": None, "started_at": datetime.now(tz.utc), "heartbeat_at": None,
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

    run_store = _FakeRunStore()
    monkeypatch.setattr(promotion_module, "RunLifecycle", lambda *a, **k: RealRunLifecycle(store=run_store))
    monkeypatch.setattr("src.common.mlflow_setup.configure_mlflow", lambda: (_ for _ in ()).throw(ConnectionError("simulated")))

    bundle_store = _FakeChannelModelBundleStore()
    bundle_store.register_candidate(
        channel="online_banking", gbm_model_version="2", lr_model_version="2", anomaly_model_version="2",
        preprocessing_artifact_version="pp-1", feature_schema_version="v1", rule_set_version="v1",
        graph_policy_version="v1", ensemble_policy_version="v1", reason_code_version="v1",
        training_run_id=5, dataset_version="cc49865cde175619", evaluation_report_ref="{}",
    )
    promotion_store = _FakeBundlePromotionStore(bundle_store)

    with pytest.raises(BundleVerificationFailedError):
        promote_bundle(
            channel="online_banking", bundle_version=1, promoted_by="analyst1", database="aidp_test",
            model_version_verifier=cli_main._MlflowModelVersionVerifier(), store=promotion_store,
        )

    assert bundle_store.rows[0].status == "CANDIDATE"  # untouched
    (run,) = run_store.rows.values()
    assert run["status"] == "FAILED"
    assert run["error_type"] == "BundleVerificationFailedError"


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


# ---- fraud-intel labels assess (Phase 7B Stage 0) --------------------------------------


def test_labels_assess_dispatches_to_assess_channel_labels_with_an_explicit_database(monkeypatch, capsys):
    captured = {}

    def _fake_assess_channel_labels(*, channel, lifecycle, load_label_bases, store, **kwargs):
        captured.update(channel=channel, lifecycle=lifecycle, load_label_bases=load_label_bases, store=store)
        return {"channel": channel, "run_id": 1, "alerts_considered": 3, "assessments_appended": 3, "mature_count": 3, "immature_count": 0, "eligible_count": 2, "unresolved_count": 1}

    fake_store = object()
    monkeypatch.setattr("src.fraud_intel.labels.eligibility.assess_channel_labels", _fake_assess_channel_labels)
    monkeypatch.setattr("src.fraud_intel.labels.eligibility.create_default_label_assessment_store", lambda database: fake_store)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_alert_label_bases", lambda channel, database: [])

    cli_main.main(["fraud-intel", "labels", "assess", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    assert captured["channel"] == "online_banking"
    assert captured["store"] is fake_store
    result = json.loads(capsys.readouterr().out)
    assert result == {"channel": "online_banking", "run_id": 1, "alerts_considered": 3, "assessments_appended": 3, "mature_count": 3, "immature_count": 0, "eligible_count": 2, "unresolved_count": 1}


def test_labels_assess_never_trains_or_promotes():
    """AST-based (Phase 7A convention): the handler's own source never
    calls anything named train_channel_configured or promote_bundle."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cli_main._handle_fraud_intel_labels_assess))
    forbidden = {"train_channel_configured", "promote_bundle"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            assert name not in forbidden, f"labels assess handler must never call {name}"


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
